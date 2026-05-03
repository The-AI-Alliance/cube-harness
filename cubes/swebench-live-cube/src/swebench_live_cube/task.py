"""Task and TaskConfig for swebench-live-cube.

Extends the SWE-bench Verified task with SWE-bench Live specifics:
- Per-instance test_cmds (no heuristic test command generation needed)
- At least one FAIL_TO_PASS test must pass (not all) on Linux
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from cube.container import ContainerBackend, relocate_if_readonly
from cube.core import ActionSchema, Observation
from cube.task import STOP_ACTION, RuntimeContext, Task, TaskConfig, TaskExecutionInfo, TaskMetadata

from swebench_live_cube.tool import SWEBenchTool, SWEBenchToolConfig

logger = logging.getLogger(__name__)

# POSIX-compatible: use `.` instead of `source`, skip silently if conda is absent.
# Works with both bash (Daytona/Modal/Toolkit backends) and sh/dash (LocalContainer).
CONDA_ACTIVATE = "if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then . /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed; fi"


class SWEBenchLiveTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for SWE-bench Live tasks.

    Public fields shipped in task_metadata.json (available at import time).
    Heavy execution data (problem_statement, patch, test_patch, etc.) lives on
    ``SWEBenchLiveExecutionInfo`` and is loaded lazily by
    ``SWEBenchLiveTaskConfig.make()``.
    """

    repo: str
    """GitHub repository name, e.g. 'django/django'."""

    base_commit: str
    """Git commit hash the agent's solution must be applied on top of."""

    splits: list[str]
    """SWE-bench Live splits this task belongs to, e.g. ['verified', 'full']."""

    log_parser: str
    """Test log parser to use during evaluation, e.g. 'pytest'."""


class SWEBenchLiveExecutionInfo(TaskExecutionInfo):
    """Heavy per-task execution data for SWE-bench Live — populated on the worker.

    Loaded by ``SWEBenchLiveTaskConfig.make()`` from the per-task execution cache
    written by ``SWEBenchLiveBenchmarkConfig.install()``.

    Mirrors the SWE-bench Verified execution info but adds ``test_cmds``: SWE-bench
    Live ships explicit per-instance test commands rather than relying on a
    repo-aware heuristic.
    """

    problem_statement: str
    """The agent-facing GitHub issue text."""

    hints_text: str = ""
    """Optional hint text (only surfaced when ``SWEBenchLiveTaskConfig.include_hints`` is True)."""

    patch: str
    """Gold patch — written to /tmp/gold_patch.diff in oracle_mode."""

    test_patch: str
    """Test patch applied during evaluation."""

    fail_to_pass: list[str]
    """Test directives that must pass after the fix (Live: at least one)."""

    pass_to_pass: list[str]
    """Test directives that must remain passing after the fix (Live: zero failures)."""

    test_cmds: list[str] = []
    """Explicit shell commands to run during evaluation; replaces the
    repo-aware test command heuristic used by SWE-bench Verified."""

    eval_timeout: int = 1800
    """Wall-clock seconds allowed for the evaluation test commands."""


class SWEBenchLiveTask(Task[SWEBenchLiveTaskMetadata]):
    """A single SWE-bench Live task with test-based validation."""

    validate_per_step: bool = False
    accept_agent_stop: bool = True

    include_hints: bool = False
    """If True, append hints_text to the problem statement in reset()."""

    oracle_mode: bool = False
    """If True, write the gold patch to /tmp/gold_patch.diff in reset()."""

    @property
    def _exec(self) -> SWEBenchLiveExecutionInfo:
        """Typed view on execution_info — fails fast if it was not populated."""
        if not isinstance(self.execution_info, SWEBenchLiveExecutionInfo):
            raise RuntimeError(
                f"SWEBenchLiveTask {self.metadata.id!r}: execution_info is "
                f"{type(self.execution_info).__name__}, expected SWEBenchLiveExecutionInfo. "
                f"Construct via SWEBenchLiveTaskConfig.make() so it is populated."
            )
        return self.execution_info

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        # TODO: remove once cube-standard auto-includes STOP_ACTION in Task.action_set
        # (upstream fix: Task.action_set appends STOP_ACTION when accept_agent_stop=True,
        # and STOP_ACTION constant gets the Anthropic-compatible parameters schema).
        stop = ActionSchema(
            name=STOP_ACTION.name,
            description=STOP_ACTION.description,
            parameters={"type": "object", "properties": {}},
        )
        return actions + [stop]

    def _build_tool(self) -> None:
        """Ensure /testbed files are writable and git-safe, then build the tool.

        Two pre-flight fixes applied unconditionally (mirrors swebench-verified-cube):
        1. git safe.directory: Git 2.35.2+ refuses to operate in repos owned by a
           different user. Configure /testbed as safe so agents can run `git diff`.
        2. chmod via cp/mv: some live containers ship root-owned 644 .py files inside
           a world-writable /testbed. mv unlinks via the writable parent and recreates
           with the runtime user's ownership, making every file writable without sudo.
           Running before relocate_if_readonly keeps conda editable-install paths stable.
        """
        self._container.exec(
            f"git config --global --add safe.directory {self.tool_config.working_dir}",
            timeout=30,
        )
        self._container.exec(
            f"find {self.tool_config.working_dir} -not -path '*/.git/*' -name '*.py' ! -writable"
            f' -exec sh -c \'cp "$1" "$1.tmp" && mv "$1.tmp" "$1"\' _ {{}} \\;'
            f" 2>/dev/null || true",
            timeout=120,
        )
        new_wd = relocate_if_readonly(
            self._container,
            self.tool_config.working_dir,
            "/tmp/testbed",
            extra_setup="git config --global --add safe.directory /tmp/testbed",
        )
        if new_wd != self.tool_config.working_dir:
            # After cp -a, the conda editable install still points to the original /testbed.
            # Use the testbed Python to locate the real site-packages and update every
            # .egg-link / .pth file that references the old path, so Python imports from
            # the relocated copy (where patches will be applied).
            orig_wd = self.tool_config.working_dir
            py_script = (
                "import site, os; "
                "dirs = site.getsitepackages() + [site.getusersitepackages()]; "
                "updated = []; "
                "[updated.append(p) or open(p, 'w').write(c.replace(orig, new)) "
                " for d in dirs "
                " for root, _, files in os.walk(d) "
                " for fname in files if fname.endswith(('.egg-link', '.pth')) "
                " for p in [os.path.join(root, fname)] "
                " for c in [open(p).read()] if orig in c]; "
                "print('editable-install paths updated:', len(updated), updated)"
            )
            result = self._container.exec(
                f"{CONDA_ACTIVATE} && python -c \"orig='{orig_wd}'; new='{new_wd}'; {py_script}\" 2>/dev/null || true",
                timeout=30,
            )
            logger.info("Editable-install path update: %s", result.stdout.strip())
        self._tool = self.tool_config.model_copy(update={"working_dir": new_wd}).make(container=self._container)

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()

        # Oracle mode: write gold patch for debug/baseline use
        if self.oracle_mode and self._exec.patch:
            assert isinstance(self.tool, SWEBenchTool)
            b64 = base64.b64encode(self._exec.patch.encode()).decode()
            self.tool.bash(f"echo '{b64}' | base64 -d > /tmp/gold_patch.diff")

        assert isinstance(self.tool, SWEBenchTool)
        instruction = self._exec.problem_statement
        if self.include_hints and self._exec.hints_text:
            instruction += f"\n\n## Hints\n{self._exec.hints_text}"
        instruction += f"\n\n[Working directory: {self.tool._config.working_dir}]"

        return Observation.from_text(instruction), {
            "instance_id": self.metadata.id,
            "repo": self.metadata.repo,
        }

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        assert isinstance(self.tool, SWEBenchTool)

        # Apply test patch
        self._apply_patch(self._exec.test_patch)

        fail_to_pass = self._exec.fail_to_pass
        pass_to_pass = self._exec.pass_to_pass
        test_cmds = self._exec.test_cmds
        eval_timeout = self._exec.eval_timeout

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
        working_dir = self.tool._config.working_dir
        # Prepend working_dir (and src/ for src-layout packages) to PYTHONPATH so that
        # source files patched in working_dir take precedence over any site-packages copy.
        pythonpath = f"export PYTHONPATH={working_dir}:{working_dir}/src:${{PYTHONPATH:-}}"
        for cmd in test_cmds:
            full_cmd = f"{pythonpath}; {CONDA_ACTIVATE} && {cmd}"
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


class SWEBenchLiveTaskConfig(TaskConfig[SWEBenchLiveTaskMetadata]):
    """Serializable factory that produces a SWEBenchLiveTask.

    Loads heavy execution data (problem_statement, patch, test_patch, test_cmds, etc.)
    from the per-task execution cache populated by ``SWEBenchLiveBenchmarkConfig.install()``.
    """

    include_hints: bool = False
    """If True, append hints_text to the problem statement in reset()."""

    oracle_mode: bool = False
    """If True, write the gold patch to /tmp/gold_patch.diff in reset()."""

    def verify_installed(self) -> None:
        """Fail fast if the per-task execution cache is empty."""
        cache_dir = type(self).task_execution_cache_dir()
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
            raise RuntimeError(
                f"SWE-bench Live per-task execution cache is empty at {cache_dir}. "
                f"Run `cube install swebench-live-cube` (or "
                f"`SWEBenchLiveBenchmarkConfig.install()`) on this worker first."
            )

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> SWEBenchLiveTask:
        if runtime_context is None or "infra" not in runtime_context:
            if container_backend is None:
                raise ValueError(
                    "SWEBenchLiveTaskConfig.make() requires runtime_context['infra'] "
                    "(preferred) or a legacy container_backend."
                )

        self.verify_installed()
        raw = self.load_task_execution_info()
        execution_info = SWEBenchLiveExecutionInfo.model_validate(raw)

        return SWEBenchLiveTask(
            metadata=self.metadata,
            execution_info=execution_info,
            tool_config=self.tool_config or SWEBenchToolConfig(),
            runtime_context=runtime_context,
            container_backend=container_backend,
            include_hints=self.include_hints,
            oracle_mode=self.oracle_mode,
        )
