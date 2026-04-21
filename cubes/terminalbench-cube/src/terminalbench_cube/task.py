"""Task and TaskConfig for terminalbench-cube."""

import base64
import io
import logging
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.resource import ResourceHandle
from cube.task import Task, TaskConfig, TaskMetadata
from cube.task_infra import launch_task_container
from terminalbench_cube.pytest_parser import PytestParser
from terminalbench_cube.tool import TerminalBenchTool, TerminalBenchToolConfig

logger = logging.getLogger(__name__)


class TerminalBenchTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for Terminal-Bench tasks.

    Public fields shipped in task_metadata.json (available at import time).
    Heavy execution data (instruction, archive) lives in the per-task execution
    cache and is loaded lazily by TerminalBenchTaskConfig.make().
    """

    difficulty: str
    """Task difficulty level: 'easy', 'medium', or 'hard'."""

    category: str
    """Task category, e.g. 'scientific-computing', 'debugging'."""

    tags: list[str]
    """Task tags for fine-grained filtering."""

    max_agent_timeout_sec: int
    """Maximum wall-clock seconds the agent is allowed to run (from task.toml)."""


class TerminalBenchTask(Task):
    """A single Terminal-Bench task with pytest-based validation."""

    metadata: TerminalBenchTaskMetadata  # type: ignore[assignment]

    validate_per_step: bool = False
    accept_agent_stop: bool = True

    # L3 resource handle owned by this task: the per-task Docker container
    # launched in model_post_init via the benchmark's InfraConfig. Closed in
    # close(). Not serialised (PrivateAttr).
    _resource_handle: ResourceHandle | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Launch the per-task container via the benchmark's infra, then build the tool.

        Expected carrier convention (see openspec change `deprecate-container-backend`):
        ``runtime_context["infra"]`` holds an ``InfraConfig`` instance. If present,
        we build a per-task ``DockerServiceConfig`` and hand its container to the tool.
        Falls back to the legacy container_backend path if no infra is provided.
        """
        if self.runtime_context is not None and "infra" in self.runtime_context:
            cc = self.metadata.container_config  # type: ignore[union-attr]
            self._resource_handle, self._container = launch_task_container(
                self.runtime_context,
                name=f"terminalbench-{self.metadata.id}",
                image=cc.image,
                ram_gb=cc.ram_gb,
                cpu_cores=cc.cpu_cores,
            )
            self._tool = self.tool_config.make(container=self._container)
            return

        super().model_post_init(__context)

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        extra = self.metadata.extra_info

        # Extract task archive to a temp dir (kept alive until close())
        self._temp_dir = tempfile.TemporaryDirectory()
        task_path = Path(self._temp_dir.name) / self.metadata.id
        task_path.mkdir(parents=True, exist_ok=True)
        archive = extra["archive"]
        if isinstance(archive, str):
            archive = base64.b64decode(archive)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            tar.extractall(path=task_path, filter="data")
        self._task_path = task_path

        # Oracle mode: upload solution for debugging/baselines
        if extra.get("oracle_mode") and (task_path / "solution").exists():
            assert isinstance(self.tool, TerminalBenchTool)
            self.tool.bash("mkdir -p /solution")
            self.tool.upload_directory(task_path / "solution", "/solution")

        return Observation.from_text(extra["instruction"]), {
            "task_id": self.metadata.id,
            "difficulty": self.metadata.difficulty,
            "category": self.metadata.category,
        }

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        assert isinstance(self.tool, TerminalBenchTool)
        extra = self.metadata.extra_info

        # Upload test harness to the sandbox
        if self._task_path is not None:
            tests_dir = self._task_path / "tests"
            self.tool.bash("mkdir -p /tests /logs/verifier")
            if tests_dir.exists():
                self.tool.upload_directory(tests_dir, "/tests")
                self.tool.bash("chmod +x /tests/test.sh")

        # Run test.sh → pytest → writes reward to /logs/verifier/reward.txt
        output = self.tool.bash(
            "cd /app && bash /tests/test.sh",
            timeout=extra.get("max_test_timeout_sec", 900),
        )
        test_results = self._parse_pytest_output(output)

        # Read reward written by test.sh
        reward_output = self.tool.bash("cat /logs/verifier/reward.txt 2>/dev/null || echo 0")
        try:
            reward = float(reward_output.strip().split()[0])
        except (ValueError, IndexError):
            reward = 0.0

        n_passed = sum(1 for r in test_results.values() if r == "passed")
        return reward, {
            "done": True,
            "passed": n_passed,
            "total": len(test_results),
            "all_passed": len(test_results) > 0 and n_passed == len(test_results),
            "test_results": test_results,
            "output_preview": output[:1000] if output else "",
        }

    def finished(self, obs: Observation | None = None) -> bool:
        return False

    def close(self) -> None:
        if hasattr(self, "_temp_dir") and self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
            self._task_path = None
        super().close()
        # Tear down the per-task container via the resource handle (new path) or
        # directly via Container.stop() (legacy path).
        if self._resource_handle is not None:
            logger.info(f"Closing resource handle for task {self.metadata.id}")
            self._resource_handle.close()
            self._resource_handle = None
            self._container = None
        elif self._container is not None:
            logger.info(f"Stopping container {self._container.id} for task {self.metadata.id}")
            self._container.stop()
            self._container = None

    def _parse_pytest_output(self, output: str) -> dict[str, str]:
        """Parse pytest output, falling back to regex heuristics."""
        try:
            return {name: status.value for name, status in PytestParser().parse(output).items()}
        except ValueError:
            logger.debug("PytestParser failed, falling back to heuristics")

        results: dict[str, str] = {}
        for label, status in [("passed", "passed"), ("failed", "failed")]:
            match = re.search(rf"(\d+)\s+{label}", output)
            if match:
                for i in range(int(match.group(1))):
                    results[f"test_{label}_{i}"] = status
        return results


class TerminalBenchTaskConfig(TaskConfig):
    """Serializable factory that produces a TerminalBenchTask.

    Loads heavy execution data (instruction, archive) from the per-task execution
    cache in make(), so it works correctly in Ray workers.
    """

    oracle_mode: bool = False
    """If True, upload the gold solution to /solution in reset()."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> TerminalBenchTask:
        # Import here to avoid circular import (benchmark imports task)
        from terminalbench_cube.benchmark import TerminalBenchBenchmark

        has_infra = runtime_context is not None and "infra" in runtime_context
        if not has_infra and container_backend is None:
            raise ValueError(
                "TerminalBenchTaskConfig.make() requires runtime_context['infra'] "
                "(preferred) or a legacy container_backend."
            )

        metadata = TerminalBenchBenchmark.task_metadata[self.task_id]
        exec_info = TerminalBenchBenchmark.load_task_execution_info(self.task_id)
        exec_info["oracle_mode"] = self.oracle_mode
        metadata = metadata.model_copy(update={"extra_info": exec_info})

        return TerminalBenchTask(
            metadata=metadata,
            tool_config=self.tool_config or TerminalBenchToolConfig(),
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
