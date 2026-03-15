"""Tool layer — bash, read_file, write_file backed by a CUBE Container."""

import base64
import logging
import shlex
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

    @tool_action
    def bash(self, command: str, timeout: int = 120) -> str:
        """Execute a bash command in the sandbox and return its output."""
        result = self._exec(command, timeout=timeout)
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(result.stderr)
        if result.exit_code == 124:
            parts.append(f"[error] Command timed out after {timeout}s")
        elif result.exit_code != 0:
            parts.append(f"[exit_code: {result.exit_code}]")
        output = "\n".join(parts) if parts else "(no output)"
        encoded = output.encode("utf-8")
        if len(encoded) <= self._config.max_output_bytes:
            return output
        return encoded[: self._config.max_output_bytes].decode("utf-8", errors="ignore") + "\n[truncated]"

    @tool_action
    def read_file(self, path: str) -> str:
        """Read the contents of a file in the sandbox."""
        result = self._exec(f"cat {shlex.quote(path)}")
        if result.exit_code != 0:
            return f"Error reading {path}: {result.stderr or result.stdout}"
        return result.stdout

    @tool_action
    def write_file(self, path: str, content: str) -> str:
        """Write content to a file in the sandbox."""
        self._exec(f"mkdir -p {shlex.quote(str(Path(path).parent))}")
        escaped = content.replace("\\", "\\\\").replace("'", "'\\''")
        self._exec(f"printf '%s' '{escaped}' > {shlex.quote(path)}")
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
        """Upload a local directory tree to the container."""
        self._exec(f"mkdir -p {shlex.quote(remote_dir)}")
        for item in local_dir.rglob("*"):
            if item.is_file():
                remote_path = f"{remote_dir}/{item.relative_to(local_dir)}"
                self._exec(f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}")
                self.upload_file(item, remote_path)
