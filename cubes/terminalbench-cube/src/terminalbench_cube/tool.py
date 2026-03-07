"""Tool layer — bash, read_file, write_file actions backed by a CUBE Container."""

import io
import logging
import shlex
import tarfile
from pathlib import Path

from cube.container import Container, ExecResult
from cube.tool import Tool, ToolConfig, tool_action

logger = logging.getLogger(__name__)

MAX_OUTPUT_BYTES = 100_000


def _truncate(output: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    """Truncate output to a maximum UTF-8 byte length."""
    encoded = output.encode("utf-8")
    if len(encoded) <= max_bytes:
        return output
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return f"{truncated}\n[truncated at {max_bytes} bytes]"


class TerminalBenchToolConfig(ToolConfig):
    """Serializable config for the terminal-bench tool."""

    working_dir: str = "/app"
    max_output_bytes: int = MAX_OUTPUT_BYTES

    def make(self, container: Container | None = None) -> "TerminalBenchTool":
        if container is None:
            raise ValueError("TerminalBenchTool requires a container")
        return TerminalBenchTool(config=self, container=container)


class TerminalBenchTool(Tool):
    """Agent-facing tool that wraps a CUBE Container for terminal tasks."""

    def __init__(self, config: TerminalBenchToolConfig, container: Container) -> None:
        self._config = config
        self._container = container

    def reset(self) -> None:
        pass

    @tool_action
    def bash(self, command: str, timeout: int = 120) -> str:
        """Execute a bash command in the sandbox and return its output."""
        result: ExecResult = self._container.exec(
            command,
            timeout=timeout,
            workdir=self._config.working_dir,
        )
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
        return _truncate(output, self._config.max_output_bytes)

    @tool_action
    def read_file(self, path: str) -> str:
        """Read the contents of a file in the sandbox."""
        result = self._container.exec(f"cat {shlex.quote(path)}", workdir=self._config.working_dir)
        if result.exit_code != 0:
            return f"Error reading {path}: {result.stderr or result.stdout}"
        return result.stdout

    @tool_action
    def write_file(self, path: str, content: str) -> str:
        """Write content to a file in the sandbox."""
        parent = str(Path(path).parent)
        self._container.exec(f"mkdir -p {shlex.quote(parent)}", workdir=self._config.working_dir)
        # Use heredoc to avoid shell escaping issues with content
        escaped = content.replace("\\", "\\\\").replace("'", "'\\''")
        self._container.exec(
            f"printf '%s' '{escaped}' > {shlex.quote(path)}",
            workdir=self._config.working_dir,
        )
        return f"Wrote {len(content)} bytes to {path}"

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload a local file to the container (not an agent action)."""
        try:
            content = local_path.read_text(encoding="utf-8")
            self.write_file(remote_path, content)
        except UnicodeDecodeError:
            self._upload_binary(local_path, remote_path)

    def upload_directory(self, local_dir: Path, remote_dir: str) -> None:
        """Recursively upload a local directory to the container (not an agent action)."""
        self._container.exec(f"mkdir -p {shlex.quote(remote_dir)}", workdir=self._config.working_dir)
        for item in local_dir.rglob("*"):
            if item.is_file():
                relative = item.relative_to(local_dir)
                remote_path = f"{remote_dir}/{relative}"
                remote_parent = str(Path(remote_path).parent)
                self._container.exec(f"mkdir -p {shlex.quote(remote_parent)}", workdir=self._config.working_dir)
                self.upload_file(item, remote_path)

    def _upload_binary(self, local_path: Path, remote_path: str) -> None:
        """Upload a binary file using base64 encoding."""
        import base64
        b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
        parent = str(Path(remote_path).parent)
        self._container.exec(f"mkdir -p {shlex.quote(parent)}", workdir=self._config.working_dir)
        self._container.exec(
            f"printf '%s' {shlex.quote(b64)} | base64 -d > {shlex.quote(remote_path)}",
            workdir=self._config.working_dir,
        )
