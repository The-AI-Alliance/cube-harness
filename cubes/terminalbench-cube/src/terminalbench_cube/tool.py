"""Terminal tool for TerminalBench — extends TerminalTool with file upload helpers."""

from __future__ import annotations

import base64
import io
import logging
import shlex
import tarfile
from pathlib import Path

from cube.container import Container
from cube.tools.terminal import TerminalTool, TerminalToolConfig

logger = logging.getLogger(__name__)


class TerminalBenchToolConfig(TerminalToolConfig):
    """Config for TerminalBench — bash + file actions, 900s timeout cap."""

    working_dir: str = "/app"
    max_timeout: int | None = 900
    enable_file_actions: bool = True

    def make(self, container: Container | None = None) -> "TerminalBenchTool":
        if container is None:
            raise ValueError("TerminalBenchTool requires a container")
        return TerminalBenchTool(config=self, container=container)


class TerminalBenchTool(TerminalTool):
    """TerminalTool extended with directory/file upload helpers for task setup.

    Inherits bash(), read_file(), write_file(), bash_unlimited() from TerminalTool.
    Overrides bash() to normalise LLM-supplied millisecond timeouts (timeout > 3600
    is divided by 1000).  upload_file() and upload_directory() are internal helpers
    used by the Task during reset() — not exposed as agent actions.
    """

    def bash(self, command: str, timeout: int = 120) -> str:
        """Execute a bash command, normalising ms → s for LLM-supplied timeouts."""
        if timeout > 3600:
            timeout = timeout // 1000
        return super().bash(command, timeout=timeout)

    # ── Internal helpers (used by Task, not exposed to agent) ─────

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload a local file to the container."""
        try:
            self.write_file(remote_path, local_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
            self._container.exec(
                f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}", workdir=self._config.working_dir
            )
            self._container.exec(
                f"printf '%s' {shlex.quote(b64)} | base64 -d > {shlex.quote(remote_path)}",
                workdir=self._config.working_dir,
            )

    def upload_directory(self, local_dir: Path, remote_dir: str) -> None:
        """Upload a local directory tree to the container in a single exec.

        Packs ``local_dir`` into an in-memory tar.gz, writes the base64 string
        to a temp file via multi-chunk ``printf >> file`` (shell-quoting-safe
        even through nested eai CLI → remote bash layers), then decodes and
        extracts.  Uses only base64+tar which every POSIX task image ships —
        no dependency on python3 in the target image.
        """
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(local_dir, arcname=".")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        remote_q = shlex.quote(remote_dir)
        # Write the base64 payload in 8 KB chunks — short chunks are robust
        # through multiple shell layers (observed with eai CLI + bash -lc).
        chunk_size = 8192
        staging = "/tmp/cube-upload.tar.gz.b64"
        self._container.exec(f": > {staging}", workdir=self._config.working_dir)
        for i in range(0, len(b64), chunk_size):
            self._container.exec(
                f"printf %s {shlex.quote(b64[i : i + chunk_size])} >> {staging}", workdir=self._config.working_dir
            )
        self._container.exec(
            f"mkdir -p {remote_q} && base64 -d < {staging} | tar -xzf - -C {remote_q} && rm -f {staging}",
            workdir=self._config.working_dir,
        )
