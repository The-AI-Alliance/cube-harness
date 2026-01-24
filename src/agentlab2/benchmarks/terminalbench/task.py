"""Terminal-Bench task implementation."""

import io
import logging
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from agentlab2.action_spaces.swe_action_space import SWEActionSpace
from agentlab2.benchmarks.terminalbench.pytest_parser import PytestParser
from agentlab2.core import ActionSchema, ActionSubset, Observation, Task
from agentlab2.tools.daytona import DaytonaSWETool

logger = logging.getLogger(__name__)

# Files that should NOT be given to the agent at setup
HIDDEN_FILES = {"tests", "solution.sh", "run-tests.sh", "Dockerfile", "docker-compose.yaml", "task.yaml"}


class TerminalBenchTask(Task):
    """A single Terminal-Bench task.

    Terminal-Bench tasks evaluate agents on real-world terminal tasks like:
    - Compiling code, training models, setting up servers
    - Debugging systems, file operations, data processing

    Scoring is binary: all tests must pass for reward=1.0.
    """

    validate_per_step: bool = False
    supported_actions: ActionSubset = (
        SWEActionSpace.bash,
        SWEActionSpace.read_file,
        SWEActionSpace.write_file,
    )
    _tool: DaytonaSWETool

    def __init__(
        self,
        id: str,
        instruction: str,
        archive: bytes,
        difficulty: str,
        category: str,
        tags: list[str],
        max_agent_timeout_sec: int,
        max_test_timeout_sec: int,
    ) -> None:
        self.id = id
        self.instruction = instruction
        self.archive = archive
        self.difficulty = difficulty
        self.category = category
        self.tags = tags
        self.max_agent_timeout_sec = max_agent_timeout_sec
        self.max_test_timeout_sec = max_test_timeout_sec
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._task_path: Path | None = None

    def _extract_archive(self) -> Path:
        """Extract the task archive to a temporary directory."""
        self._temp_dir = tempfile.TemporaryDirectory()
        task_path = Path(self._temp_dir.name) / self.id
        task_path.mkdir(parents=True, exist_ok=True)

        with tarfile.open(fileobj=io.BytesIO(self.archive), mode="r:gz") as tar:
            tar.extractall(path=task_path, filter="data")

        self._task_path = task_path
        return task_path

    def _get_initial_files(self) -> list[Path]:
        """Get files to upload at setup (agent can see these)."""
        if self._task_path is None:
            raise RuntimeError("Task archive not extracted")

        initial_files = []
        for item in self._task_path.iterdir():
            if item.name not in HIDDEN_FILES:
                initial_files.append(item)
        return initial_files

    def _upload_initial_files(self) -> None:
        """Upload initial task files to /app in the sandbox."""
        initial_files = self._get_initial_files()

        for file_path in initial_files:
            if file_path.is_file():
                content = file_path.read_text(encoding="utf-8", errors="replace")
                remote_path = f"/app/{file_path.name}"
                self._tool.write_file(remote_path, content)
                logger.debug(f"Uploaded {file_path.name} to {remote_path}")
            elif file_path.is_dir():
                # Upload directory contents recursively
                self._upload_directory(file_path, f"/app/{file_path.name}")

    def _upload_directory(self, local_dir: Path, remote_dir: str) -> None:
        """Recursively upload a directory to the sandbox."""
        self._tool.bash(f"mkdir -p {remote_dir}")

        for item in local_dir.rglob("*"):
            if item.is_file():
                relative = item.relative_to(local_dir)
                remote_path = f"{remote_dir}/{relative}"
                # Ensure parent directory exists
                remote_parent = str(Path(remote_path).parent)
                self._tool.bash(f"mkdir -p {remote_parent}")
                # Read and upload file
                try:
                    content = item.read_text(encoding="utf-8", errors="replace")
                    self._tool.write_file(remote_path, content)
                except Exception as e:
                    # Handle binary files by base64 encoding
                    logger.warning(f"Could not upload {item} as text: {e}, trying binary")
                    self._upload_binary_file(item, remote_path)

    def _upload_binary_file(self, local_path: Path, remote_path: str) -> None:
        """Upload a binary file using base64 encoding."""
        import base64

        content = local_path.read_bytes()
        b64_content = base64.b64encode(content).decode("ascii")
        self._tool.bash(f"echo '{b64_content}' | base64 -d > {remote_path}")

    def setup(self, tool: DaytonaSWETool) -> tuple[Observation, dict]:
        """Initialize the task environment."""
        logger.info(f"Setting up Terminal-Bench task: {self.id}")

        self._tool = tool
        self._extract_archive()  # extract all the necessary files
        self._tool.bash("mkdir -p /app")
        self._upload_initial_files()
        
        obs = Observation.from_text(self.instruction)

        return obs, {
            "task_id": self.id,
            "difficulty": self.difficulty,
            "category": self.category,
            "tags": self.tags,
            "max_agent_timeout_sec": self.max_agent_timeout_sec,
        }

    def _upload_test_files(self) -> None:
        """Upload test files at validation time."""
        if self._task_path is None:
            raise RuntimeError("Task archive not extracted")

        tests_dir = self._task_path / "tests"
        run_tests_sh = self._task_path / "run-tests.sh"

        # Create /tests directory
        self._tool.bash("mkdir -p /tests")

        # Upload tests directory
        if tests_dir.exists():
            self._upload_directory(tests_dir, "/tests")
            logger.debug("Uploaded tests/ to /tests")

        # Upload run-tests.sh
        if run_tests_sh.exists():
            content = run_tests_sh.read_text(encoding="utf-8")
            self._tool.write_file("/app/run-tests.sh", content)
            self._tool.bash("chmod +x /app/run-tests.sh")
            logger.debug("Uploaded run-tests.sh to /app/run-tests.sh")

    def _parse_pytest_output(self, output: str) -> dict[str, str]:
        """Parse pytest output to extract individual test results.

        Uses Terminal-Bench's PytestParser for robust parsing.
        Falls back to simple heuristics if parser fails.

        Returns dict of {test_name: "passed" | "failed"}.
        """
        parser = PytestParser()

        try:
            parsed = parser.parse(output)
            # Convert UnitTestStatus to string
            return {name: status.value for name, status in parsed.items()}
        except ValueError:
            # Parser failed (no short test summary section)
            # Fall back to simple heuristics
            logger.debug("PytestParser failed, falling back to heuristics")

        results = {}

        # Try to parse from the summary line like "2 passed in 0.01s"
        passed_match = re.search(r"(\d+)\s+passed", output)
        failed_match = re.search(r"(\d+)\s+failed", output)

        if passed_match:
            n_passed = int(passed_match.group(1))
            for i in range(n_passed):
                results[f"test_{i}"] = "passed"
        if failed_match:
            n_failed = int(failed_match.group(1))
            for i in range(n_failed):
                results[f"test_failed_{i}"] = "failed"

        return results

    def validate_task(self, obs: Observation, *args: Any) -> tuple[float, dict]:
        """Validate the task by running tests.

        Terminal-Bench uses binary scoring: all tests must pass for reward=1.0.
        """
        logger.info(f"Validating Terminal-Bench task: {self.id}")

        # Upload test files
        self._upload_test_files()

        # Run tests with TEST_DIR environment variable
        test_command = "cd /app && export TEST_DIR=/tests && bash run-tests.sh"
        output = self._tool.bash(test_command, timeout=self.max_test_timeout_sec)

        # Parse test results
        test_results = self._parse_pytest_output(output)

        # Determine if all tests passed
        all_passed = len(test_results) > 0 and all(r == "passed" for r in test_results.values())
        n_passed = sum(1 for r in test_results.values() if r == "passed")
        n_total = len(test_results)

        # Check exit code in output (from our tool's format)
        exit_code_match = re.search(r"\[exit_code:\s*(\d+)\]", output)
        exit_code = int(exit_code_match.group(1)) if exit_code_match else None

        # Also check for pytest success indicators
        pytest_passed = "passed" in output.lower() and "failed" not in output.lower()

        # Binary reward: 1.0 if all tests pass, 0.0 otherwise
        # Trust test_results if we have them, otherwise fall back to exit code
        if n_total > 0:
            reward = 1.0 if all_passed else 0.0
        elif exit_code is not None:
            reward = 1.0 if exit_code == 0 else 0.0
        else:
            reward = 1.0 if pytest_passed else 0.0

        return reward, {
            "done": True,
            "exit_code": exit_code,
            "passed": n_passed,
            "total": n_total,
            "all_passed": all_passed,
            "test_results": test_results,
            "output_preview": output[:1000] if output else "",
        }

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        """Filter to only SWE actions."""
        supported_names = {action.__name__ for action in self.supported_actions}
        filtered = [a for a in actions if a.name in supported_names]
        logger.debug(f"Filtered to {len(filtered)} SWE actions for Terminal-Bench")
        return filtered

    def finished(self) -> bool:
        """Check if task is complete (after validation)."""
        return False  # Let the environment handle this via final_step

    def teardown(self) -> None:
        """Clean up temporary files."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
            self._task_path = None
