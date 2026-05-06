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

from cube.container import ContainerBackend, relocate_if_readonly
from cube.core import Observation
from cube.task import STOP_ACTION, RuntimeContext, Task, TaskConfig, TaskExecutionInfo, TaskMetadata
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


class TerminalBenchExecutionInfo(TaskExecutionInfo):
    """Heavy per-task execution data for TerminalBench — populated on the worker."""

    instruction: str
    archive: str
    max_test_timeout_sec: int = 900


class TerminalBenchTask(Task[TerminalBenchTaskMetadata]):
    """A single Terminal-Bench task with pytest-based validation."""

    metadata: TerminalBenchTaskMetadata  # type: ignore[assignment]

    validate_per_step: bool = False
    accept_agent_stop: bool = True
    oracle_mode: bool = False

    @property
    def action_set(self) -> list:
        return super().action_set + [STOP_ACTION]

    # Container-side paths — always under /tmp so logic works uniformly on root
    # and non-root backends (EAI Toolkit images have /tmp mode 1777).
    _solution_dir: str = PrivateAttr(default="/tmp/solution")
    _tests_dir: str = PrivateAttr(default="/tmp/tests")
    _logs_verifier_dir: str = PrivateAttr(default="/tmp/logs/verifier")

    @property
    def _exec(self) -> TerminalBenchExecutionInfo:
        """Typed view on execution_info — fails fast if it was not populated."""
        if not isinstance(self.execution_info, TerminalBenchExecutionInfo):
            raise RuntimeError(
                f"TerminalBenchTask {self.metadata.id!r}: execution_info is "
                f"{type(self.execution_info).__name__}, expected TerminalBenchExecutionInfo. "
                f"Construct via TerminalBenchTaskConfig.make() so it is populated."
            )
        return self.execution_info

    def _build_tool(self) -> None:
        new_wd = relocate_if_readonly(
            self._container,
            self.tool_config.working_dir,
            "/tmp/app",
            # Git refuses dirs whose ownership differs ('dubious ownership').
            # '*' disables the check globally — safe in this test-runner context.
            # uid 13011 (Toolkit) has no /etc/passwd entry, so git can't
            # auto-detect committer identity without explicit config.
            extra_setup=(
                "git config --global --add safe.directory '*' && "
                "git config --global user.email 'cube-harness@example.com' && "
                "git config --global user.name 'Cube Harness'"
            ),
        )
        self._tool = self.tool_config.model_copy(update={"working_dir": new_wd}).make(container=self._container)

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()

        # Fast-fail if the container sandbox is broken (all bash commands return
        # exit_code=1 with no output). Avoids wasting 100 agent steps on a dead env.
        health = self.tool.bash("echo ok", timeout=15)
        if "ok" not in health:
            raise RuntimeError(f"Container health check failed for {self.metadata.id!r}: {health!r}")

        # Extract task archive to a temp dir (kept alive until close())
        self._temp_dir = tempfile.TemporaryDirectory()
        task_path = Path(self._temp_dir.name) / self.metadata.id
        task_path.mkdir(parents=True, exist_ok=True)
        archive = self._exec.archive
        if isinstance(archive, str):
            archive = base64.b64decode(archive)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            tar.extractall(path=task_path, filter="data")
        self._task_path = task_path

        # Oracle mode: upload solution for debugging/baselines
        if self.oracle_mode and (task_path / "solution").exists():
            assert isinstance(self.tool, TerminalBenchTool)
            app_dir = self.tool._config.working_dir  # type: ignore[attr-defined]
            solution_dir = task_path / "solution"
            if app_dir != "/app":
                self._rewrite_files_locally(solution_dir, {"/app/": f"{app_dir}/"})
            self.tool.bash(f"mkdir -p {self._solution_dir}")
            self.tool.upload_directory(solution_dir, self._solution_dir)
            # Pre-install python3 + uv so oracle solve.sh scripts work on minimal
            # images (e.g. bare LaTeX) that ship without python3.  In non-oracle
            # runs the agent installs its own deps; we don't add overhead there.
            self._ensure_uv_preinstalled()

        instruction = self._exec.instruction
        # Mirror the test-script rewrite: when /app is read-only and we relocated
        # the working dir to /tmp/app, the agent's instructions still reference
        # /app verbatim (from terminal-bench task.yaml).  Rewrite so the agent
        # writes to the same path the evaluator checks.
        assert isinstance(self.tool, TerminalBenchTool)
        app_dir = self.tool._config.working_dir  # type: ignore[attr-defined]
        if app_dir != "/app":
            instruction = re.sub(r"/app(?=[\s/.,;:)\]'\"]|$)", app_dir, instruction)

        return Observation.from_text(instruction), {
            "task_id": self.metadata.id,
            "difficulty": self.metadata.difficulty,
            "category": self.metadata.category,
        }

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        assert isinstance(self.tool, TerminalBenchTool)

        # Upload test harness to the sandbox
        if self._task_path is not None:
            tests_dir = self._task_path / "tests"
            self.tool.bash(f"mkdir -p {self._tests_dir} {self._logs_verifier_dir}")
            if tests_dir.exists():
                # Rewrite hardcoded paths in local test files before uploading.
                # Done in Python (not via sed) to avoid shell-quoting pitfalls
                # (e.g. single quotes inside sed expressions) and GNU sed
                # re-scanning surprises that produce double '/tmp/' prefixes.
                assert isinstance(self.tool, TerminalBenchTool)
                app_dir = self.tool._config.working_dir  # type: ignore[attr-defined]
                path_subs: dict[str, str] = {
                    "/logs/verifier": self._logs_verifier_dir,
                    "/tests/": self._tests_dir + "/",
                    "/tests ": self._tests_dir + " ",
                    # Path("/tests") — no trailing slash, quote-boundary match
                    '"/tests"': f'"{self._tests_dir}"',
                    "'/tests'": f"'{self._tests_dir}'",
                }
                if app_dir != "/app":
                    path_subs["/app/"] = f"{app_dir}/"
                    path_subs['"/app"'] = f'"{app_dir}"'
                    path_subs["'/app'"] = f"'{app_dir}'"
                self._rewrite_files_locally(tests_dir, path_subs)
                self.tool.upload_directory(tests_dir, self._tests_dir)
                self.tool.bash(f"chmod +x {self._tests_dir}/test.sh")

        # Pre-install `uv` + fake HOME so test.sh's
        #   curl https://astral.sh/uv/…/install.sh | sh  →  source $HOME/.local/bin/env
        # succeeds even when astral.sh is unreachable (EAI Toolkit returns 403
        # Forbidden on that host) and when $HOME is a read-only mount.
        # pypi is reachable on Toolkit; pip installs uv in ~10 s.
        self._ensure_uv_preinstalled()

        # Run test.sh → pytest → writes reward.txt in the logs-verifier dir.
        # Tool's working_dir is already set (may be /tmp/app after relocation).
        output = self.tool.bash(
            f"export HOME=/tmp/fakehome && bash {self._tests_dir}/test.sh",
            timeout=self._exec.max_test_timeout_sec,
        )
        test_results = self._parse_pytest_output(output)

        # Read reward written by test.sh
        reward_output = self.tool.bash(f"cat {self._logs_verifier_dir}/reward.txt 2>/dev/null || echo 0")
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

    @staticmethod
    def _rewrite_files_locally(directory: Path, subs: dict[str, str]) -> None:
        """Apply string substitutions to *.sh and *.py files under ``directory`` in-place.

        Preferred over sed-in-container: avoids shell-quoting pitfalls (e.g. single
        quotes inside sed expressions) and GNU sed re-scanning surprises.
        """
        for f in directory.rglob("*"):
            if f.suffix in (".sh", ".py") and f.is_file():
                text = f.read_text(errors="replace")
                new_text = text
                for k, v in subs.items():
                    new_text = new_text.replace(k, v)
                if new_text != text:
                    f.write_text(new_text)

    def _ensure_uv_preinstalled(self) -> None:
        """Pre-install ``uv`` so test.sh's ``source $HOME/.local/bin/env`` works.

        Terminal-Bench task test.sh files bootstrap ``uv`` via
            curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
            source $HOME/.local/bin/env

        On some backends (EAI Toolkit in particular), ``astral.sh`` returns HTTP
        403 (cluster IP range rejected by Cloudflare) AND ``curl`` isn't even in
        the image AND ``$HOME`` is read-only.  All three failures cascade: the
        curl is rc=127, the source finds nothing, uvx is missing, pytest can't
        run, reward=0.

        Fix: ensure python3 is present (some minimal images like LaTeX ship
        without it — install via apt if needed), then install ``uv`` via ``pip``
        from PyPI into ``/tmp/fakehome/.local/bin``, create the env file
        test.sh expects, and override ``HOME=/tmp/fakehome`` when running test.sh.

        Non-root fallback: when running as non-root (e.g. EAI Toolkit uid 13011),
        apt-get requires root.  Fall back to downloading the python3 packages
        via ``apt-get download`` (works without root, writes to /tmp) and
        extracting them with ``dpkg-deb --extract``, then use that python3 to
        bootstrap pip (via get-pip.py with SSL verification disabled for the
        bootstrap step only) and finally ``pip install uv``.
        """
        marker = "/tmp/fakehome/.local/bin/uv"
        probe = self.tool.bash(f"test -x {marker} && echo EXISTS || echo MISSING", timeout=15)
        if "EXISTS" in probe:
            return

        # Fast path: cube_assets EAI data mount (see ToolkitInfraConfig.assets_data).
        # When the harness mounts /opt/cube-assets/ with a uv binary, copy it
        # directly — bypasses the python3-bootstrap path that fails on minimal
        # images lacking python3, curl, AND apt sources.  Note: EAI data mounts
        # are read-only and strip the execute bit (mode 0600), so we use ``-f``
        # for the probe and ``chmod +x`` after copying into the writable HOME.
        assets_probe = self.tool.bash(
            "test -f /opt/cube-assets/uv && test -f /opt/cube-assets/uvx && echo YES || echo NO",
            timeout=15,
        )
        if "YES" in assets_probe:
            logger.info("Using mounted /opt/cube-assets/uv for uv install")
            self.tool.bash(
                "export HOME=/tmp/fakehome && "
                "mkdir -p $HOME/.local/bin && "
                "cp /opt/cube-assets/uv /opt/cube-assets/uvx $HOME/.local/bin/ && "
                "chmod +x $HOME/.local/bin/uv $HOME/.local/bin/uvx && "
                "printf 'export PATH=\"$HOME/.local/bin:$PATH\"\\n' > $HOME/.local/bin/env",
                timeout=30,
            )
            return

        # Some minimal images (e.g. bare LaTeX) ship without python3.
        # Try root apt-get first (works on Docker/local backends).
        has_python = self.tool.bash("python3 --version 2>/dev/null && echo HAS_PYTHON || echo NO_PYTHON", timeout=15)
        if "NO_PYTHON" in has_python:
            logger.info("python3 not found — trying apt-get install (root path)")
            self.tool.bash(
                "apt-get update -qq && apt-get install -y --no-install-recommends python3 python3-pip 2>&1",
                timeout=120,
            )
            has_python = self.tool.bash(
                "python3 --version 2>/dev/null && echo HAS_PYTHON || echo NO_PYTHON", timeout=15
            )

        if "NO_PYTHON" in has_python:
            # Root apt-get failed (non-root container).  Download packages without
            # root and extract them to /tmp/python3_pkg.
            logger.info("root apt-get failed — trying non-root apt download + dpkg-deb extract")
            self._install_python3_nonroot()
            has_python = self.tool.bash(
                "test -x /tmp/python3_pkg/usr/bin/python3.12 && echo HAS_PYTHON || echo NO_PYTHON", timeout=10
            )

        if "NO_PYTHON" in has_python:
            logger.warning("python3 unavailable — skipping uv pre-install; test.sh will fall back to curl")
            return

        logger.info("Pre-installing uv into /tmp/fakehome/.local/bin (backend-portable workaround)")

        use_extracted = "exists" in self.tool.bash(
            "test -x /tmp/python3_pkg/usr/bin/python3.12 && echo exists || echo missing", timeout=5
        )

        if use_extracted:
            # Bootstrap pip via get-pip.py (SSL verification disabled for this
            # one-time download from bootstrap.pypa.io; pip itself uses certifi).
            self.tool.write_file(
                "/tmp/_dl_pip.py",
                "import ssl, urllib.request as R\n"
                "ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)\n"
                "ctx.check_hostname=False\n"
                "ctx.verify_mode=ssl.CERT_NONE\n"
                "open('/tmp/get-pip.py','wb').write(R.urlopen('https://bootstrap.pypa.io/get-pip.py',context=ctx).read())\n",
            )
            cmd = (
                "export LD_LIBRARY_PATH=/tmp/python3_pkg/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH && "
                "export HOME=/tmp/fakehome && "
                "mkdir -p $HOME/.local/bin && "
                "/tmp/python3_pkg/usr/bin/python3.12 /tmp/_dl_pip.py 2>&1 && "
                "/tmp/python3_pkg/usr/bin/python3.12 /tmp/get-pip.py --target /tmp/pip_pkg -q 2>&1 && "
                "PYTHONPATH=/tmp/pip_pkg /tmp/python3_pkg/usr/bin/python3.12 "
                "-m pip install --quiet --target /tmp/uv_pkg uv==0.9.5 2>&1 && "
                "cp /tmp/uv_pkg/bin/uv /tmp/uv_pkg/bin/uvx $HOME/.local/bin/ && "
                "printf 'export PATH=\"$HOME/.local/bin:$PATH\"\\n' > $HOME/.local/bin/env"
            )
        else:
            cmd = (
                "export HOME=/tmp/fakehome && "
                "mkdir -p $HOME/.local/bin && "
                # --trusted-host covers images where ca-certificates is absent (e.g. bare LaTeX)
                "python3 -m pip install --quiet --target /tmp/uv_pkg "
                "--trusted-host pypi.org --trusted-host files.pythonhosted.org uv && "
                "cp /tmp/uv_pkg/bin/uv /tmp/uv_pkg/bin/uvx $HOME/.local/bin/ && "
                "printf 'export PATH=\"$HOME/.local/bin:$PATH\"\\n' > $HOME/.local/bin/env"
            )

        result = self.tool.bash(cmd, timeout=300)
        if not result or "error" in result.lower():
            logger.warning("uv pre-install may have failed; test.sh will fall back to curl: %s", result[:200])

    def _install_python3_nonroot(self) -> None:
        """Download python3.12 packages via apt and extract with dpkg-deb (no root needed)."""
        logger.info("Downloading python3.12 packages via apt (non-root) and extracting to /tmp/python3_pkg")
        apt_opts = "-o Dir::State::Lists=/tmp/apt/lists -o Dir::Cache::Archives=/tmp/apt/archives"
        cmd = (
            "mkdir -p /tmp/apt/lists/partial /tmp/apt/archives/partial /tmp/python3_pkg && "
            f"apt-get {apt_opts} update -qq 2>/dev/null || true && "
            f"cd /tmp && apt-get {apt_opts} download "
            "python3.12-minimal libpython3.12-minimal libpython3.12-stdlib python3-minimal 2>&1 && "
            'for deb in /tmp/*.deb; do dpkg-deb --extract "$deb" /tmp/python3_pkg/; done'
        )
        result = self.tool.bash(cmd, timeout=180)
        logger.info("python3 nonroot install: %s", (result or "")[-300:])

    def finished(self, obs: Observation | None = None) -> bool:
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


class TerminalBenchTaskConfig(TaskConfig[TerminalBenchTaskMetadata]):
    """Serializable factory that produces a TerminalBenchTask.

    Loads heavy execution data (instruction, archive) from the per-task execution
    cache in make(), so it works correctly in Ray workers.
    """

    oracle_mode: bool = False
    """If True, upload the gold solution to /solution in reset()."""

    def verify_installed(self) -> None:
        """Fail fast if the per-task execution cache is empty."""
        cache_dir = type(self).task_execution_cache_dir()
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
            raise RuntimeError(
                f"TerminalBench per-task execution cache is empty at {cache_dir}. "
                f"Run `cube install terminalbench-cube` (or "
                f"`TerminalBenchBenchmarkConfig.install()`) on this worker first."
            )

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> TerminalBenchTask:
        has_infra = runtime_context is not None and "infra" in runtime_context
        if not has_infra and container_backend is None:
            raise ValueError(
                "TerminalBenchTaskConfig.make() requires runtime_context['infra'] "
                "(preferred) or a legacy container_backend."
            )

        self.verify_installed()
        raw = self.load_task_execution_info()
        execution_info = TerminalBenchExecutionInfo.model_validate(raw)

        return TerminalBenchTask(
            metadata=self.metadata,
            execution_info=execution_info,
            tool_config=self.tool_config or TerminalBenchToolConfig(),
            runtime_context=runtime_context,
            container_backend=container_backend,
            oracle_mode=self.oracle_mode,
        )
