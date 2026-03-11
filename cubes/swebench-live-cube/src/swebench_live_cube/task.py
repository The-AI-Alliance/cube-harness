"""Task and TaskConfig for swebench-live-cube.

Extends the SWE-bench Verified task with SWE-bench Live specifics:
- Per-instance test_cmds (no heuristic test command generation needed)
- At least one FAIL_TO_PASS test must pass (not all) on Linux
"""

import base64
import logging
from typing import Any

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata

from swebench_live_cube.tool import SWEBenchTool, SWEBenchToolConfig

logger = logging.getLogger(__name__)

CONDA_ACTIVATE = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"


class SWEBenchLiveTask(Task):
    """A single SWE-bench Live task with test-based validation."""

    validate_per_step: bool = False
    accept_agent_stop: bool = True

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        extra = self.metadata.extra_info

        instruction = extra["problem_statement"]
        if extra.get("include_hints") and extra.get("hints_text"):
            instruction += f"\n\n## Hints\n{extra['hints_text']}"

        return Observation.from_text(instruction), {
            "instance_id": self.metadata.id,
            "repo": extra["repo"],
        }

    def evaluate(self, obs: Observation) -> tuple[float, dict[str, Any]]:
        assert isinstance(self.tool, SWEBenchTool)
        extra = self.metadata.extra_info

        # Apply test patch
        self._apply_patch(extra["test_patch"])

        fail_to_pass = extra["fail_to_pass"]
        pass_to_pass = extra["pass_to_pass"]
        test_cmds = extra.get("test_cmds", [])
        eval_timeout = extra.get("eval_timeout", 1800)

        # Run tests using explicit test_cmds from the dataset
        test_output = self._run_test_cmds(test_cmds, timeout=eval_timeout)

        # Parse results from the log_parser type
        log_parser = extra.get("log_parser", "pytest")
        f2p_passed, p2p_failed = self._check_test_results(test_output, fail_to_pass, pass_to_pass, log_parser)

        # SWE-bench Live Linux: at least one FAIL_TO_PASS must pass, zero PASS_TO_PASS failures
        resolved = f2p_passed > 0 and p2p_failed == 0
        reward = 1.0 if resolved else 0.0

        return reward, {
            "done": True,
            "resolved": resolved,
            "fail_to_pass_passed": f2p_passed,
            "fail_to_pass_total": len(fail_to_pass),
            "pass_to_pass_failed": p2p_failed,
            "pass_to_pass_total": len(pass_to_pass),
            "test_output": test_output[:2000],
        }

    # ── Private helpers ────────────────────────────────────────────

    def _apply_patch(self, patch: str) -> str:
        """Apply a unified diff patch to /testbed using git apply with fallbacks."""
        assert isinstance(self.tool, SWEBenchTool)
        b64 = base64.b64encode(patch.encode()).decode()
        self.tool.bash(f"echo '{b64}' | base64 -d > /tmp/patch.diff")

        result = self.tool.bash("cd /testbed && git apply /tmp/patch.diff 2>&1", timeout=30)
        if "[exit_code:" not in result:
            return result

        result = self.tool.bash("cd /testbed && git apply --reject /tmp/patch.diff 2>&1", timeout=30)
        if "[exit_code:" not in result:
            return result

        return self.tool.bash("cd /testbed && patch --batch --fuzz=5 -p1 -i /tmp/patch.diff 2>&1", timeout=30)

    def _run_test_cmds(self, test_cmds: list[str], timeout: int = 1800) -> str:
        """Run the explicit test commands from the dataset."""
        assert isinstance(self.tool, SWEBenchTool)
        if not test_cmds:
            return "(no test commands)"

        outputs = []
        for cmd in test_cmds:
            full_cmd = f"{CONDA_ACTIVATE} && cd /testbed && {cmd}"
            output = self.tool.bash(full_cmd, timeout=timeout)
            outputs.append(output)
        return "\n".join(outputs)

    @staticmethod
    def _check_test_results(
        output: str,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        log_parser: str,
    ) -> tuple[int, int]:
        """Check test results: count FAIL_TO_PASS successes and PASS_TO_PASS failures.

        Returns:
            (fail_to_pass_passed, pass_to_pass_failed)
        """
        f2p_passed = 0
        p2p_failed = 0

        if log_parser == "pytest":
            # pytest verbose output: "test_name PASSED" or "test_name FAILED"
            for test_id in fail_to_pass:
                if f"{test_id} PASSED" in output or f"{test_id}::PASSED" in output:
                    f2p_passed += 1
            for test_id in pass_to_pass:
                if f"{test_id} FAILED" in output or f"{test_id} ERROR" in output:
                    p2p_failed += 1
        else:
            # Generic fallback: check exit code patterns
            if "[exit_code:" not in output and "[error]" not in output:
                f2p_passed = len(fail_to_pass)
            else:
                p2p_failed = len(pass_to_pass)

        return f2p_passed, p2p_failed


class SWEBenchLiveTaskConfig(TaskConfig):
    """Serializable factory that produces a SWEBenchLiveTask."""

    task_metadata_snapshot: TaskMetadata | None = None

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> SWEBenchLiveTask:
        if container_backend is None:
            raise ValueError("SWEBenchLiveTaskConfig.make() requires a container_backend")

        metadata = self.task_metadata_snapshot
        if metadata is None:
            # Import here to avoid circular import (benchmark imports task)
            from swebench_live_cube.benchmark import SWEBenchLiveBenchmark

            metadata = SWEBenchLiveBenchmark.task_metadata[self.task_id]

        return SWEBenchLiveTask(
            metadata=metadata,
            tool_config=self.tool_config or SWEBenchToolConfig(),
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
