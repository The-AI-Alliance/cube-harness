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


def _maybe_relocate_app(container, tool_config: TerminalBenchToolConfig) -> TerminalBenchToolConfig:
    """If ``tool_config.working_dir`` is read-only in the container, copy it to a
    writable path and return an updated ToolConfig.

    Pattern mirrors swebench's ``_maybe_relocate_testbed``.  Should eventually
    be promoted to a shared ``cube.container.Container`` utility.
    """
    wd = tool_config.working_dir
    probe = container.exec(f"test -w {wd} && echo W || echo R", timeout=30)
    if "R" not in probe.stdout:
        return tool_config
    new_wd = "/tmp/app"
    logger.info("%s not writable by runtime user — copying to %s", wd, new_wd)
    container.exec(
        f"cp -a {wd} {new_wd} && "
        # Git refuses to run in dirs whose ownership differs from the caller
        # ('dubious ownership' warning).  '*' disables the check globally —
        # safe in this test-runner context.
        "git config --global --add safe.directory '*' && "
        # Toolkit runs as uid 13011 with no /etc/passwd entry, so git can't
        # auto-detect a committer identity.  Tasks that make commits (e.g.
        # fix-git's `git merge`) fail with exit 128 "Committer identity unknown"
        # without this.  No-op on backends where identity is already configured.
        "git config --global user.email 'cube-harness@example.com' && "
        "git config --global user.name 'Cube Harness'",
        timeout=300,
    )
    return tool_config.model_copy(update={"working_dir": new_wd})


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

    # Container-side paths — we always put these under /tmp so the logic works
    # uniformly on root and non-root backends.  /tmp is universally writable
    # (it's a tmpfs on every POSIX container, and EAI Toolkit images also have
    # it mode 1777).  The task images still have read-only /testbed or /app
    # dirs owned by root, which is why we sometimes relocate those — but the
    # dirs we CREATE are always in /tmp.
    _solution_dir: str = PrivateAttr(default="/tmp/solution")
    _tests_dir: str = PrivateAttr(default="/tmp/tests")
    _logs_verifier_dir: str = PrivateAttr(default="/tmp/logs/verifier")

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
            # Some terminal-bench images chown /app to root mode 755.  When the
            # backend runs as an unprivileged user (EAI Toolkit: uid 13011
            # toolkit), the agent's git operations in /app fail silently.
            # Detect and fall back to a writable copy.  All the dirs we CREATE
            # (solution/, tests/, logs/verifier) are already under /tmp by
            # default — see PrivateAttr defaults on this class.
            tool_config = _maybe_relocate_app(self._container, self.tool_config)
            self._tool = tool_config.make(container=self._container)
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
            timeout=extra.get("max_test_timeout_sec", 900),
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
                "python3 -m pip install --quiet --target /tmp/uv_pkg uv && "
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
            "for deb in /tmp/*.deb; do dpkg-deb --extract \"$deb\" /tmp/python3_pkg/; done"
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
