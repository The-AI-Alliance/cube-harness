"""Task and TaskConfig for swebench-verified-cube."""

import base64
import json
import logging
import shlex
from typing import Any

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata

from swebench_verified_cube.tool import SWEBenchTool, SWEBenchToolConfig

logger = logging.getLogger(__name__)

# POSIX-compatible: use `.` instead of `source`, skip silently if conda is absent.
# Works with both bash (Daytona/Modal/Toolkit backends) and sh/dash (LocalContainer).
CONDA_ACTIVATE = "if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then . /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed; fi"


class SWEBenchVerifiedTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for SWE-bench Verified tasks.

    Public fields shipped in task_metadata.json (available at import time).
    Heavy execution data (problem_statement, patch, test_patch, etc.) lives in
    the per-task execution cache and is loaded lazily by SWEBenchVerifiedTaskConfig.make().
    """

    repo: str
    """GitHub repository, e.g. 'django/django'."""

    difficulty: str
    """Estimated fix time, e.g. '15 min - 1 hour'."""

    version: str
    """Repository version string, e.g. '4.3'."""

    base_commit: str
    """Git SHA of the base commit the agent starts from."""


class SWEBenchVerifiedTask(Task):
    """A single SWE-bench Verified task with test-based validation."""

    metadata: SWEBenchVerifiedTaskMetadata  # type: ignore[assignment]

    validate_per_step: bool = False
    accept_agent_stop: bool = True

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
            "difficulty": self.metadata.difficulty,
        }

    def evaluate(self, obs: Observation) -> tuple[float, dict[str, Any]]:
        assert isinstance(self.tool, SWEBenchTool)
        extra = self.metadata.extra_info

        # Apply test patch
        self._apply_patch(extra["test_patch"])

        # Parse test lists
        fail_to_pass = json.loads(extra["fail_to_pass"])
        pass_to_pass = json.loads(extra["pass_to_pass"])
        eval_timeout = extra.get("eval_timeout", 1800)

        # Run FAIL_TO_PASS tests — these must all pass for resolution
        f2p_passed, f2p_output = self._run_tests(self.metadata.repo, fail_to_pass, timeout=eval_timeout)

        # Run PASS_TO_PASS tests — these must remain passing
        p2p_passed = True
        p2p_output = ""
        if pass_to_pass:
            p2p_passed, p2p_output = self._run_tests(self.metadata.repo, pass_to_pass, timeout=eval_timeout)

        resolved = f2p_passed and p2p_passed
        reward = 1.0 if resolved else 0.0

        return reward, {
            "done": True,
            "resolved": resolved,
            "fail_to_pass_passed": f2p_passed,
            "pass_to_pass_passed": p2p_passed,
            "fail_to_pass_output": f2p_output[:2000],
            "pass_to_pass_output": p2p_output[:2000],
        }

    def close(self) -> None:
        super().close()
        if self._container is not None:
            logger.info(f"Stopping container {self._container.id} for task {self.metadata.id}")
            self._container.stop()
            self._container = None

    # ── Private helpers ────────────────────────────────────────────

    def _apply_patch(self, patch: str) -> str:
        """Apply a unified diff patch to /testbed using git apply with fallbacks."""
        assert isinstance(self.tool, SWEBenchTool)
        b64 = base64.b64encode(patch.encode()).decode()
        self.tool.bash_unlimited(f"echo '{b64}' | base64 -d > /tmp/patch.diff")

        # Try git apply first
        result = self.tool.bash_unlimited("cd /testbed && git apply /tmp/patch.diff 2>&1", timeout=30)
        if "[exit_code:" not in result and "[error]" not in result:
            return result

        # Fallback: git apply --reject
        result = self.tool.bash_unlimited("cd /testbed && git apply --reject /tmp/patch.diff 2>&1", timeout=30)
        if "[exit_code:" not in result and "[error]" not in result:
            return result

        # Final fallback: patch
        return self.tool.bash_unlimited("cd /testbed && patch --batch --fuzz=5 -p1 -i /tmp/patch.diff 2>&1", timeout=60)

    def _run_tests(self, repo: str, test_directives: list[str], timeout: int = 1800) -> tuple[bool, str]:
        """Run test directives and return (all_passed, output)."""
        assert isinstance(self.tool, SWEBenchTool)
        if not test_directives:
            return True, ""

        test_cmd = self._build_test_cmd(repo, test_directives)
        cmd = f"{CONDA_ACTIVATE} && cd /testbed && {test_cmd}"
        output = self.tool.bash_unlimited(cmd, timeout=timeout)

        all_passed = "[exit_code:" not in output and "[error]" not in output
        return all_passed, output

    @staticmethod
    def _normalize_django_directive(directive: str) -> str:
        """Convert SWE-bench unittest verbose format to Django runtests.py format.

        SWE-bench stores test directives in Python unittest verbose output format:
            "test_method (module.path.ClassName)"
        Django's runtests.py expects:
            "module.path.ClassName.test_method"
        """
        import re

        m = re.match(r"^(\w+)\s+\(([^)]+)\)$", directive.strip())
        if m:
            method, class_path = m.group(1), m.group(2)
            return f"{class_path}.{method}"
        return directive  # already in the right format or unrecognised — pass through

    @staticmethod
    def _build_test_cmd(repo: str, test_directives: list[str]) -> str:
        """Build the test command based on repo's test framework."""
        if "django" in repo:
            normalized = [SWEBenchVerifiedTask._normalize_django_directive(t) for t in test_directives]
            tests = " ".join(shlex.quote(t) for t in normalized)
            return f"./tests/runtests.py --verbosity 2 {tests}"
        if "sympy" in repo:
            tests = " ".join(shlex.quote(t) for t in test_directives)
            return f"bin/test -C --verbose {tests}"
        tests = " ".join(shlex.quote(t) for t in test_directives)
        return f"python -m pytest --no-header -rN -p no:cacheprovider {tests}"


class SWEBenchVerifiedTaskConfig(TaskConfig):
    """Serializable factory that produces a SWEBenchVerifiedTask.

    Loads heavy execution data (problem_statement, patch, test_patch, etc.) from
    the per-task execution cache in make(), so it works correctly in Ray workers.
    """

    include_hints: bool = False
    oracle_mode: bool = False

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> SWEBenchVerifiedTask:
        if container_backend is None:
            raise ValueError("SWEBenchVerifiedTaskConfig.make() requires a container_backend")

        # Import here to avoid circular import (benchmark imports task)
        from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmark

        metadata = SWEBenchVerifiedBenchmark.task_metadata[self.task_id]
        exec_info = SWEBenchVerifiedBenchmark.load_task_execution_info(self.task_id)
        metadata = metadata.model_copy(
            update={
                "extra_info": {
                    **exec_info,
                    "include_hints": self.include_hints,
                    "oracle_mode": self.oracle_mode,
                }
            }
        )

        return SWEBenchVerifiedTask(
            metadata=metadata,
            tool_config=self.tool_config or SWEBenchToolConfig(),
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
