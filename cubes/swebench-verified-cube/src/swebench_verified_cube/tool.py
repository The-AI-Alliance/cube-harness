"""Tool layer — bash, read_file, write_file backed by a CUBE Container."""

import logging
import shlex
from pathlib import Path
from typing import Any

from cube.container import Container, ExecResult
from cube.tool import Tool, ToolConfig, tool_action

logger = logging.getLogger(__name__)

MAX_OUTPUT_BYTES = 100_000


class SWEBenchToolConfig(ToolConfig):
    """Config for the SWE-bench tool."""

    working_dir: str = "/testbed"
    max_output_bytes: int = MAX_OUTPUT_BYTES

    def make(self, container: Container | None = None) -> "SWEBenchTool":
        if container is None:
            raise ValueError("SWEBenchTool requires a container")
        return SWEBenchTool(config=self, container=container)


class SWEBenchTool(Tool):
    """Agent-facing tool — delegates all execution to a CUBE Container."""

    def __init__(self, config: SWEBenchToolConfig, container: Container) -> None:
        self._config = config
        self._container = container

    def reset(self) -> None:
        pass

    def _exec(self, command: str, **kwargs: Any) -> ExecResult:
        """Run a command in the container with default workdir."""
        kwargs.setdefault("workdir", self._config.working_dir)
        return self._container.exec(command, **kwargs)

    # ── Agent actions ──────────────────────────────────────────────

    def _run_bash(self, command: str, timeout: int = 120) -> str:
        """Execute a command and return the full output (no truncation)."""
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
        return "\n".join(parts) if parts else "(no output)"

    @tool_action
    def bash(self, command: str, timeout: int = 120) -> str:
        """Execute a bash command in the sandbox and return its output.

        Args:
            command: Shell command to run. The working directory is /testbed
                (the cloned repo). Use absolute paths or assume cwd=/testbed.
            timeout: Wall-clock seconds (NOT milliseconds). Default 120s.
                Use larger values (600-1800) for test suites. Max 3600s.
        """
        timeout = min(timeout, 3600)
        output = self._run_bash(command, timeout=timeout)
        encoded = output.encode("utf-8")
        if len(encoded) <= self._config.max_output_bytes:
            return output
        return encoded[: self._config.max_output_bytes].decode("utf-8", errors="ignore") + "\n[truncated]"

    def bash_unlimited(self, command: str, timeout: int = 120) -> str:
        """Like bash() but without output truncation — for internal use (e.g. evaluate())."""
        result = self._container.exec(
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
        return "\n".join(parts) if parts else "(no output)"

    @tool_action
    def read_file(self, path: str, line_start: int | None = None, line_end: int | None = None) -> str:
        """Read the contents of a file in the sandbox.

        Args:
            path: Path to the file. Relative paths resolve against /testbed
                (e.g. 'django/core/validators.py'). Absolute paths also work.
            line_start: First line to return (1-indexed, inclusive). Omit to read from the start.
            line_end: Last line to return (1-indexed, inclusive). Omit to read to the end.
        """
        if line_start is not None or line_end is not None:
            start = max(1, line_start or 1)
            end = str(line_end) if line_end is not None else "$"
            cmd = f"sed -n '{start},{end}p' {shlex.quote(path)}"
        else:
            cmd = f"cat {shlex.quote(path)}"
        result = self._exec(cmd)
        if result.exit_code != 0:
            return f"Error reading {path}: {result.stderr or result.stdout}"
        return result.stdout

    @tool_action
    def write_file(self, path: str, content: str) -> str:
        """Write content to a file in the sandbox (overwrites any existing file).

        Args:
            path: Destination path. Parent directories are created as needed.
                Relative paths resolve against /testbed.
            content: Full file contents to write. Pass the entire new file body —
                this is not a patch tool. For incremental edits, use bash with
                sed / patch / git apply.
        """
        self._exec(f"mkdir -p {shlex.quote(str(Path(path).parent))}")
        escaped = content.replace("'", "'\\''")
        self._exec(f"printf '%s' '{escaped}' > {shlex.quote(path)}")
        return f"Wrote {len(content)} bytes to {path}"
