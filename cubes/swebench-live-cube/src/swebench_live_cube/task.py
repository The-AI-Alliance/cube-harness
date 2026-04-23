"""Task and TaskConfig for swebench-live-cube.

Extends the SWE-bench Verified task with SWE-bench Live specifics:
- Per-instance test_cmds (no heuristic test command generation needed)
- At least one FAIL_TO_PASS test must pass (not all) on Linux
"""

import base64
import logging
from typing import Any

from cube.container import ContainerBackend, relocate_if_readonly
from cube.core import Observation
from cube.task import RuntimeContext, Task, TaskConfig, TaskMetadata

from swebench_live_cube.tool import SWEBenchTool, SWEBenchToolConfig

logger = logging.getLogger(__name__)

# POSIX-compatible: use `.` instead of `source`, skip silently if conda is absent.
# Works with both bash (Daytona/Modal/Toolkit backends) and sh/dash (LocalContainer).
CONDA_ACTIVATE = "if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then . /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed; fi"


class SWEBenchLiveTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for SWE-bench Live tasks.

    Public fields shipped in task_metadata.json (available at import time).
    Heavy execution data (problem_statement, patch, test_patch, etc.) lives in
    the per-task execution cache and is loaded lazily by SWEBenchLiveTaskConfig.make().
    """

    repo: str
    """GitHub repository name, e.g. 'django/django'."""

    base_commit: str
    """Git commit hash the agent's solution must be applied on top of."""

    splits: list[str]
    """SWE-bench Live splits this task belongs to, e.g. ['verified', 'full']."""

    log_parser: str
    """Test log parser to use during evaluation, e.g. 'pytest'."""


class SWEBenchLiveTask(Task):
    """A single SWE-bench Live task with test-based validation."""

    metadata: SWEBenchLiveTaskMetadata  # type: ignore[assignment]

    validate_per_step: bool = False
    accept_agent_stop: bool = True

    def _build_tool(self) -> None:
        new_wd = relocate_if_readonly(
            self._container,
            self.tool_config.working_dir,
            "/tmp/testbed",
            extra_setup="git config --global --add safe.directory /tmp/testbed",
        )
        self._tool = self.tool_config.model_copy(update={"working_dir": new_wd}).make(
            container=self._container
        )

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        extra = self.metadata.extra_info

        # Oracle mode: write gold patch for debug/baseline use
        if extra.get("oracle_mode") and extra.get("patch"):
            assert isinstance(self.tool, SWEBenchTool)
            b64 = base64.b64encode(extra["patch"].encode()).decode()
            self.tool.bash(f"echo '{b64}' | base64 -d > /tmp/gold_patch.diff")

        instruction = extra["problem_statement"]
        if extra.get("include_hints") and extra.get("hints_text"):
            instruction += f"\n\n## Hints\n{extra['hints_text']}"

        return Observation.from_text(instruction), {
            "instance_id": self.metadata.id,
            "repo": self.metadata.repo,
        }

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
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

        # Use the typed log_parser field from metadata
        f2p_passed, p2p_failed = self._check_test_results(
            test_output, fail_to_pass, pass_to_pass, self.metadata.log_parser
        )

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
        self.tool.bash_unlimited(f"echo '{b64}' | base64 -d > /tmp/patch.diff")

        # Try git apply first
        # Commands run in tool.working_dir (may be relocated to writable copy).
        result = self.tool.bash_unlimited("git apply /tmp/patch.diff 2>&1", timeout=30)
        if "[exit_code:" not in result and "[error]" not in result:
            return result

        result = self.tool.bash_unlimited("git apply --reject /tmp/patch.diff 2>&1", timeout=30)
        if "[exit_code:" not in result and "[error]" not in result:
            return result

        return self.tool.bash_unlimited("patch --batch --fuzz=5 -p1 -i /tmp/patch.diff 2>&1", timeout=60)

    def _run_test_cmds(self, test_cmds: list[str], timeout: int = 1800) -> str:
        """Run the explicit test commands from the dataset."""
        assert isinstance(self.tool, SWEBenchTool)
        if not test_cmds:
            return "(no test commands)"

        outputs = []
        for cmd in test_cmds:
            full_cmd = f"{CONDA_ACTIVATE} && {cmd}"  # tool.working_dir already set
            output = self.tool.bash_unlimited(full_cmd, timeout=timeout)
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
            # Support multiple pytest output formats:
            #   verbose (-v):  "test_id PASSED [ X%]"   (test_id then status)
            #   summary (-rA): "PASSED test_id"          (status then test_id)
            #   legacy/other:  "test_id::PASSED"
            # test_ids from the dataset may be truncated prefix strings (e.g.
            # "test_validate[Invalid") which still work as substring matches.
            for test_id in fail_to_pass:
                if f"{test_id} PASSED" in output or f"{test_id}::PASSED" in output or f"PASSED {test_id}" in output:
                    f2p_passed += 1
            for test_id in pass_to_pass:
                if (
                    f"{test_id} FAILED" in output
                    or f"{test_id} ERROR" in output
                    or f"FAILED {test_id}" in output
                    or f"ERROR {test_id}" in output
                ):
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

    include_hints: bool = False
    """If True, append hints_text to the problem statement in reset()."""

    oracle_mode: bool = False
    """If True, write the gold patch to /tmp/gold_patch.diff in reset()."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> SWEBenchLiveTask:
        has_infra = runtime_context is not None and "infra" in runtime_context
        if not has_infra and container_backend is None:
            raise ValueError(
                "SWEBenchLiveTaskConfig.make() requires runtime_context['infra'] "
                "(preferred) or a legacy container_backend."
            )

        # Import here to avoid circular import (benchmark imports task)
        from swebench_live_cube.benchmark import SWEBenchLiveBenchmark

        metadata = SWEBenchLiveBenchmark.task_metadata[self.task_id]
        exec_info = SWEBenchLiveBenchmark.load_task_execution_info(self.task_id)
        # Overlay runtime config flags — these are benchmark-level settings forwarded
        # via TaskConfig so they survive Ray worker serialisation.
        exec_info["include_hints"] = self.include_hints
        exec_info["oracle_mode"] = self.oracle_mode
        metadata = metadata.model_copy(update={"extra_info": exec_info})

        return SWEBenchLiveTask(
            metadata=metadata,
            tool_config=self.tool_config or SWEBenchToolConfig(),
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
