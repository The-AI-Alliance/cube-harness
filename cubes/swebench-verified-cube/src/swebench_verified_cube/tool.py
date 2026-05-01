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
_VIEW_WINDOW = 100  # lines shown by default in view()


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

    _MAX_BASH_TIMEOUT: int = 600

    @tool_action
    def bash(self, command: str, timeout: int = 120) -> str:
        """Execute a bash command in the sandbox and return its output.

        Args:
            command: Shell command to run. The working directory is /testbed
                (the cloned repo). Use absolute paths or assume cwd=/testbed.
            timeout: Wall-clock seconds (NOT milliseconds). Default 120s, capped at 600s.
        """
        timeout = min(timeout, self._MAX_BASH_TIMEOUT)
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
    def view(self, path: str, line_start: int = 1, window: int = _VIEW_WINDOW) -> str:
        """View a file with line numbers in a scrollable window.

        Shows up to `window` lines starting at `line_start`, each prefixed with its
        line number. Reports how many lines exist above and below the window so you
        know where to scroll next. Prefer this over `bash cat` for source files —
        it keeps context manageable for large files.

        Args:
            path: File path. Relative paths resolve against /testbed.
            line_start: First line to show (1-indexed). Default 1. Use grep output
                to find the right starting line, then scroll with line_start.
            window: Number of lines to show. Default 100.
        """
        # Count total lines
        wc_result = self._exec(f"wc -l < {shlex.quote(path)}")
        if wc_result.exit_code != 0:
            return f"Error reading {path}: {wc_result.stderr or wc_result.stdout}"
        try:
            total = int(wc_result.stdout.strip())
        except ValueError:
            total = 0

        start = max(1, line_start)
        end = min(start + window - 1, total)

        # Read lines with awk to get line numbers
        cmd = f"awk 'NR>={start} && NR<={end} {{printf \"%6d\\t%s\\n\", NR, $0}}' {shlex.quote(path)}"
        result = self._exec(cmd)
        if result.exit_code != 0:
            return f"Error reading {path}: {result.stderr or result.stdout}"

        above = start - 1
        below = max(0, total - end)
        header = f"[{path}] lines {start}-{end} of {total}"
        if above:
            header += f"  ({above} lines above)"
        if below:
            header += f"  ({below} lines below — use line_start={end + 1} to continue)"
        return f"{header}\n{result.stdout}"

    @tool_action
    def read_file(self, path: str, line_start: int | None = None, line_end: int | None = None) -> str:
        """Read the contents of a file in the sandbox (no line numbers).

        For navigating large files use `view` instead — it shows line numbers and
        window context. Use `read_file` when you need the raw text for copy-paste
        into `str_replace`.

        Args:
            path: Path to the file. Relative paths resolve against /testbed
                (e.g. 'django/core/validators.py'). Absolute paths also work.
            line_start: First line to return (1-indexed, inclusive). Omit to read from the start.
            line_end: Last line to return (1-indexed, inclusive). Omit to read to the end.
        """
        if line_start is not None or line_end is not None:
            start = max(1, int(line_start) if line_start is not None else 1)
            end = str(int(line_end)) if line_end is not None else "$"
            cmd = f"sed -n '{start},{end}p' {shlex.quote(path)}"
        else:
            cmd = f"cat {shlex.quote(path)}"
        result = self._exec(cmd)
        if result.exit_code != 0:
            return f"Error reading {path}: {result.stderr or result.stdout}"
        return result.stdout

    def _write_content(self, path: str, content: str) -> None:
        """Write content to path using base64 to avoid shell escaping issues."""
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        self._exec(f"mkdir -p {shlex.quote(str(Path(path).parent))}")
        # base64 content is alphanumeric + +/= — safe to put in single quotes
        self._exec(f"printf '%s' '{encoded}' | base64 -d > {shlex.quote(path)}")

    def _lint_python(self, path: str) -> str | None:
        """Run py_compile on a .py file; return error string or None if clean."""
        if not path.endswith(".py"):
            return None
        result = self._exec(f"python -m py_compile {shlex.quote(path)} 2>&1")
        if result.exit_code != 0:
            output = (result.stderr or result.stdout or "").strip()
            return f"[syntax error] {output}"
        return None

    @tool_action
    def write_file(self, path: str, content: str) -> str:
        """Write content to a file in the sandbox (overwrites any existing file).

        For .py files, runs a syntax check after writing and reports any errors.

        Args:
            path: Destination path. Parent directories are created as needed.
                Relative paths resolve against /testbed.
            content: Full file contents to write. Pass the entire new file body.
                For targeted edits prefer str_replace — it is safer than overwriting
                the whole file.
        """
        self._write_content(path, content)
        msg = f"Wrote {len(content)} bytes to {path}"
        lint_error = self._lint_python(path)
        return f"{msg}\n{lint_error}" if lint_error else msg

    @tool_action
    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        """Replace an exact string in a file (safer than sed for targeted edits).

        Fails clearly if old_str is not found or appears more than once, so you
        always know whether the edit landed. For .py files, runs a syntax check
        after the replacement. Prefer this over bash+sed for source code changes.

        Args:
            path: File to edit. Relative paths resolve against /testbed.
            old_str: The exact text to find (including surrounding context to make
                it unique). Must appear exactly once in the file.
            new_str: Replacement text. Use an empty string to delete old_str.
        """
        result = self._exec(f"cat {shlex.quote(path)}")
        if result.exit_code != 0:
            return f"Error reading {path}: {result.stderr or result.stdout}"
        content = result.stdout
        if old_str not in content:
            if new_str in content:
                return f"Already applied: new content already present in {path}. No changes made."
            return f"Error: old_str not found in {path}. No changes made."
        count = content.count(old_str)
        if count > 1:
            return (
                f"Error: old_str appears {count} times in {path} — add more surrounding "
                "context to make it unique. No changes made."
            )
        new_content = content.replace(old_str, new_str, 1)
        self._write_content(path, new_content)
        msg = f"Replaced 1 occurrence in {path}"
        lint_error = self._lint_python(path)
        return f"{msg}\n{lint_error}" if lint_error else msg
