"""Task and TaskConfig for terminalbench-cube."""

import io
import logging
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig
from terminalbench_cube.pytest_parser import PytestParser
from terminalbench_cube.tool import TerminalBenchTool, TerminalBenchToolConfig

logger = logging.getLogger(__name__)


class TerminalBenchTask(Task):
    """A single Terminal-Bench task with pytest-based validation."""

    validate_per_step: bool = False
    accept_agent_stop: bool = True

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        extra = self.metadata.extra_info

        # Extract task archive to a temp dir (kept alive until close())
        self._temp_dir = tempfile.TemporaryDirectory()
        task_path = Path(self._temp_dir.name) / self.metadata.id
        task_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(extra["archive"]), mode="r:gz") as tar:
            tar.extractall(path=task_path, filter="data")
        self._task_path = task_path

        # Oracle mode: upload solution for debugging/baselines
        if extra.get("oracle_mode") and (task_path / "solution").exists():
            assert isinstance(self.tool, TerminalBenchTool)
            self.tool.bash("mkdir -p /solution")
            self.tool.upload_directory(task_path / "solution", "/solution")

        return Observation.from_text(extra["instruction"]), {
            "task_id": self.metadata.id,
            "difficulty": extra.get("difficulty", "unknown"),
            "category": extra.get("category", ""),
        }

    def evaluate(self, obs: Observation) -> tuple[float, dict[str, Any]]:
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

    def finished(self, obs: Observation) -> bool:
        return False

    def close(self) -> None:
        if hasattr(self, "_temp_dir") and self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
            self._task_path = None
        super().close()

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
    """Serializable factory that produces a TerminalBenchTask."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> TerminalBenchTask:
        # Import here to avoid circular import (benchmark imports task)
        from terminalbench_cube.benchmark import TerminalBenchBenchmark

        if container_backend is None:
            raise ValueError("TerminalBenchTaskConfig.make() requires a container_backend")
        return TerminalBenchTask(
            metadata=TerminalBenchBenchmark.task_metadata[self.task_id],
            tool_config=self.tool_config or TerminalBenchToolConfig(),
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
