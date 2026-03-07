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
from cube.task import Task, TaskConfig, TaskMetadata
from terminalbench_cube.pytest_parser import PytestParser
from terminalbench_cube.tool import TerminalBenchTool, TerminalBenchToolConfig

logger = logging.getLogger(__name__)

HIDDEN_FILES = {"tests", "solution", "environment", "task.toml", "instruction.md"}


class TerminalBenchTask(Task):
    """A single Terminal-Bench task — real-world terminal challenges with pytest-based validation."""

    validate_per_step: bool = False
    accept_agent_stop: bool = True

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        """Extract archive, upload initial files, return instruction as observation."""
        self.tool.reset()
        extra = self.metadata.extra_info
        archive_bytes: bytes = extra["archive"]
        instruction: str = extra["instruction"]

        self._temp_dir = tempfile.TemporaryDirectory()
        task_path = Path(self._temp_dir.name) / self.metadata.id
        task_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            tar.extractall(path=task_path, filter="data")
        self._task_path = task_path

        oracle_mode = extra.get("oracle_mode", False)
        if oracle_mode:
            solution_dir = task_path / "solution"
            if solution_dir.exists():
                assert isinstance(self.tool, TerminalBenchTool)
                self.tool.bash("mkdir -p /solution")
                self.tool.upload_directory(solution_dir, "/solution")

        obs = Observation.from_text(instruction)
        return obs, {
            "task_id": self.metadata.id,
            "difficulty": extra.get("difficulty", "unknown"),
            "category": extra.get("category", ""),
            "docker_image": self.metadata.container_config.image if self.metadata.container_config else "unknown",
        }

    def evaluate(self, obs: Observation) -> tuple[float, dict[str, Any]]:
        """Upload tests, run test.sh, parse results, return reward."""
        assert isinstance(self.tool, TerminalBenchTool)
        extra = self.metadata.extra_info

        # Upload test files
        if self._task_path is not None:
            tests_dir = self._task_path / "tests"
            self.tool.bash("mkdir -p /tests /logs/verifier")
            if tests_dir.exists():
                self.tool.upload_directory(tests_dir, "/tests")
                self.tool.bash("chmod +x /tests/test.sh")

        # Run tests
        max_test_timeout = extra.get("max_test_timeout_sec", 900)
        output = self.tool.bash("cd /app && bash /tests/test.sh", timeout=max_test_timeout)

        # Parse pytest results
        test_results = self._parse_pytest_output(output)

        # Read reward written by test.sh
        reward_output = self.tool.bash("cat /logs/verifier/reward.txt 2>/dev/null || echo 0")
        reward_str = reward_output.strip().split()[0] if reward_output.strip() else "0"
        try:
            reward = float(reward_str)
        except ValueError:
            reward = 0.0

        n_passed = sum(1 for r in test_results.values() if r == "passed")
        n_total = len(test_results)

        return reward, {
            "done": True,
            "passed": n_passed,
            "total": n_total,
            "all_passed": n_total > 0 and all(r == "passed" for r in test_results.values()),
            "test_results": test_results,
            "output_preview": output[:1000] if output else "",
        }

    def finished(self, obs: Observation) -> bool:
        """Let the agent decide when to stop via final_step."""
        return False

    def close(self) -> None:
        """Clean up temporary files and tool resources."""
        if hasattr(self, "_temp_dir") and self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
            self._task_path = None
        super().close()

    def _parse_pytest_output(self, output: str) -> dict[str, str]:
        """Parse pytest output to extract individual test results."""
        parser = PytestParser()
        try:
            parsed = parser.parse(output)
            return {name: status.value for name, status in parsed.items()}
        except ValueError:
            logger.debug("PytestParser failed, falling back to heuristics")

        results = {}
        passed_match = re.search(r"(\d+)\s+passed", output)
        failed_match = re.search(r"(\d+)\s+failed", output)
        if passed_match:
            for i in range(int(passed_match.group(1))):
                results[f"test_{i}"] = "passed"
        if failed_match:
            for i in range(int(failed_match.group(1))):
                results[f"test_failed_{i}"] = "failed"
        return results


class TerminalBenchTaskConfig(TaskConfig):
    """Serializable factory that produces a TerminalBenchTask."""

    container_backend: ContainerBackend | None = None

    model_config = {"arbitrary_types_allowed": True}

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> TerminalBenchTask:
        from terminalbench_cube.benchmark import TerminalBenchBenchmark

        task_metadata: TaskMetadata = TerminalBenchBenchmark.task_metadata[self.task_id]
        tool_cfg = self.tool_config or TerminalBenchToolConfig()
        backend = container_backend or self.container_backend
        return TerminalBenchTask(
            metadata=task_metadata,
            tool_config=tool_cfg,
            runtime_context=runtime_context,
            container_backend=backend,
        )
