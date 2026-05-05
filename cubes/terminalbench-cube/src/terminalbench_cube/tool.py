"""Tool layer — bash, read_file, write_file backed by a CUBE Container."""

import base64
import io
import logging
import shlex
import tarfile
from pathlib import Path
from typing import Any

from cube.container import Container, ExecResult
from cube.tool import Tool, ToolConfig, tool_action

logger = logging.getLogger(__name__)

MAX_OUTPUT_BYTES = 100_000


class TerminalBenchToolConfig(ToolConfig):
    """Config for the terminal-bench tool."""

    working_dir: str = "/app"
    max_output_bytes: int = MAX_OUTPUT_BYTES

    def make(self, container: Container | None = None) -> "TerminalBenchTool":
        if container is None:
            raise ValueError("TerminalBenchTool requires a container")
        return TerminalBenchTool(config=self, container=container)


class TerminalBenchTool(Tool):
    """Agent-facing tool — delegates all execution to a CUBE Container."""

    def __init__(self, config: TerminalBenchToolConfig, container: Container) -> None:
        self._config = config
        self._container = container

    def reset(self) -> None:
        pass

    def _exec(self, command: str, **kwargs: Any) -> ExecResult:
        """Run a command in the container with default workdir."""
        kwargs.setdefault("workdir", self._config.working_dir)
        return self._container.exec(command, **kwargs)

    # ── Agent actions ──────────────────────────────────────────────
    # bash() impl mirrors BashOnlySWEBenchTool (feat/swebench-verified-improvements).
    # TODO: extract to a shared cube-tools package.

    _MAX_BASH_TIMEOUT: int = 900

    def _run_bash(self, command: str, timeout: int = 120) -> str:
        result = self._container.exec(command, timeout=timeout, workdir=self._config.working_dir)
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(result.stderr)
        if result.exit_code == 124:
            parts.append(f"[timed out after {timeout}s]")
        elif result.exit_code != 0:
            parts.append(f"[exit_code: {result.exit_code}]")
        return "\n".join(parts) if parts else "(no output)"

    @tool_action
    def bash(self, command: str, timeout: int = 120) -> str:
        """Execute a bash command in the sandbox and return its output.

        Args:
            command: Shell command. cwd=/app.
            timeout: Seconds (NOT milliseconds), max 900.
        """
        if timeout > 3600:  # agent sent ms instead of seconds (e.g. 120000)
            timeout = timeout // 1000
        timeout = min(timeout, self._MAX_BASH_TIMEOUT)
        output = self._run_bash(command, timeout=timeout)
        encoded = output.encode("utf-8")
        if len(encoded) <= self._config.max_output_bytes:
            return output
        head = encoded[:5000].decode("utf-8", errors="ignore")
        tail = encoded[-5000:].decode("utf-8", errors="ignore")
        return f"{head}\n[... {len(encoded) - 10000} bytes elided ...]\n{tail}"

    def bash_unlimited(self, command: str, timeout: int = 120) -> str:
        """Unlimited bash for internal use (evaluate, test harness)."""
        return self._run_bash(command, timeout=timeout)

    @tool_action
    def read_file(self, path: str) -> str:
        """Read the contents of a file in the sandbox."""
        result = self._exec(f"cat {shlex.quote(path)}")
        if result.exit_code != 0:
            return f"Error reading {path}: {result.stderr or result.stdout}"
        encoded = result.stdout.encode("utf-8")
        if len(encoded) <= self._config.max_output_bytes:
            return result.stdout
        head = encoded[:5000].decode("utf-8", errors="ignore")
        tail = encoded[-5000:].decode("utf-8", errors="ignore")
        return f"{head}\n[... {len(encoded) - 10000} bytes elided ...]\n{tail}"

    @tool_action
    def write_file(self, path: str, content: str) -> str:
        """Write content to a file in the sandbox."""
        mkdir = self._exec(f"mkdir -p {shlex.quote(str(Path(path).parent))}")
        if mkdir.exit_code != 0:
            return f"Error creating parent dir for {path}: {mkdir.stderr or mkdir.stdout}"
        escaped = content.replace("'", "'\\''")
        write = self._exec(f"printf '%s' '{escaped}' > {shlex.quote(path)}")
        if write.exit_code != 0:
            return f"Error writing {path}: {write.stderr or write.stdout}"
        return f"Wrote {len(content)} bytes to {path}"

    # ── Internal helpers (used by Task, not exposed to agent) ─────

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload a local file to the container."""
        try:
            self.write_file(remote_path, local_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
            self._exec(f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}")
            self._exec(f"printf '%s' {shlex.quote(b64)} | base64 -d > {shlex.quote(remote_path)}")

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
        # Write the base64 payload in 8 KB chunks.  Single-arg printf through
        # multiple shell layers can mangle long strings (observed with
        # eai CLI + bash -lc); short chunks are robust.
        chunk_size = 8192
        staging = "/tmp/cube-upload.tar.gz.b64"
        self._exec(f": > {staging}")
        for i in range(0, len(b64), chunk_size):
            self._exec(f"printf %s {shlex.quote(b64[i : i + chunk_size])} >> {staging}")
        self._exec(f"mkdir -p {remote_q} && base64 -d < {staging} | tar -xzf - -C {remote_q} && rm -f {staging}")
