"""Tool layer — bash-only and full-surface tools backed by a CUBE Container."""

import logging
import re
import shlex
from pathlib import Path
from typing import Any

from cube.container import Container, ExecResult
from cube.tool import Tool, ToolConfig, tool_action

logger = logging.getLogger(__name__)

MAX_OUTPUT_BYTES = 100_000


def _parse_line_arg(val: Any) -> int | None:
    """Parse a line-number argument, handling list/range-style strings.

    Handles edge cases from models that pass ranges as a single argument:
    - ``[200, 210]`` or ``"[200, 210]"`` → first number (200) as start, last (210) as end
    - ``"200"`` or ``200`` → 200

    Returns None for None input or unparseable values.
    Callers that need both start and end from a range-style value should use
    ``_parse_line_range``.
    """
    if val is None:
        return None
    if isinstance(val, int):
        return val
    s = str(val).strip().strip("[](){}")
    parts = re.split(r"[,\s\-:]+", s)
    try:
        return int(parts[0])
    except (ValueError, IndexError):
        return None


def _parse_line_range(line_start: Any, line_end: Any) -> tuple[int | None, int | None]:
    """Parse line_start / line_end arguments, recovering range-style inputs.

    When a model passes ``line_start="[200, 210]"`` with no line_end, this
    extracts both endpoints from that single value so the intent is preserved.
    """
    if isinstance(line_start, str) and line_end is None:
        s = line_start.strip().strip("[](){}")
        parts = re.split(r"[,\s\-:]+", s)
        if len(parts) >= 2:
            try:
                return int(parts[0]), int(parts[-1])
            except (ValueError, IndexError):
                pass
    return _parse_line_arg(line_start), _parse_line_arg(line_end)


class SWEBenchToolConfig(ToolConfig):
    """Config for the SWE-bench tool (bash + read_file + write_file)."""

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
                Use larger values (600-1800) for test suites.
        """
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
                Also accepts range notation like ``[200, 210]`` when line_end is omitted.
            line_end: Last line to return (1-indexed, inclusive). Omit to read to the end.
        """
        parsed_start, parsed_end = _parse_line_range(line_start, line_end)
        if parsed_start is not None or parsed_end is not None:
            start = max(1, parsed_start if parsed_start is not None else 1)
            end = str(parsed_end) if parsed_end is not None else "$"
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


class BashOnlySWEBenchToolConfig(ToolConfig):
    """Config for bash-only SWE-bench tool — matches mini-SWE-agent's tool surface."""

    working_dir: str = "/testbed"
    max_output_bytes: int = MAX_OUTPUT_BYTES

    def make(self, container: Container | None = None) -> "BashOnlySWEBenchTool":
        if container is None:
            raise ValueError("BashOnlySWEBenchTool requires a container")
        return BashOnlySWEBenchTool(config=self, container=container)


class BashOnlySWEBenchTool(Tool):
    """Bash-only tool: a single bash() action.

    Agents edit files via inline Python, heredoc, or sed — the same pattern
    used by mini-SWE-agent. Simpler surface reduces parsing failures and
    removes lint-feedback noise from the observation stream.
    """

    _MAX_BASH_TIMEOUT: int = 600

    # Verbatim from upstream swebench.yaml environment.env — prevents interactive
    # pagers (git diff/log) and progress bars (pip, tqdm) polluting bash output.
    _ENV: dict[str, str] = {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
    }

    def __init__(self, config: BashOnlySWEBenchToolConfig, container: Container) -> None:
        self._config = config
        self._container = container

    def reset(self) -> None:
        pass

    def _run_bash(self, command: str, timeout: int = 120) -> str:
        result = self._container.exec(command, timeout=timeout, workdir=self._config.working_dir, env=self._ENV)
        output = "\n".join(filter(None, [result.stdout, result.stderr]))
        if result.exit_code == 124:
            return f"<returncode>124</returncode>\n<output>\n{output}\n[timed out after {timeout}s]\n</output>"
        return (
            f"<returncode>{result.exit_code}</returncode>\n<output>\n{output if output else '(no output)'}\n</output>"
        )

    @tool_action
    def bash(self, command: str, timeout: int = 120) -> str:
        """Execute a bash command in the sandbox (/testbed) and return its output.

        Args:
            command: Shell command. cwd=/testbed. env changes do not persist between calls.
            timeout: Seconds, max 600.
        """
        timeout = min(timeout, self._MAX_BASH_TIMEOUT)
        output = self._run_bash(command, timeout=timeout)
        encoded = output.encode("utf-8")
        if len(encoded) <= self._config.max_output_bytes:
            return output
        # Preserve <returncode> header; truncate only the <output> body.
        output_start = output.find("<output>")
        header = output[:output_start] if output_start != -1 else ""
        body_bytes = encoded[len(header.encode("utf-8")) :]
        half = self._config.max_output_bytes // 2
        head = body_bytes[:half].decode("utf-8", errors="ignore")
        tail = body_bytes[-half:].decode("utf-8", errors="ignore")
        elided = len(body_bytes) - self._config.max_output_bytes
        return f"{header}{head}\n[... {elided} bytes elided ...]\n{tail}"

    def bash_unlimited(self, command: str, timeout: int = 120) -> str:
        """Unlimited bash for internal use (evaluate, apply_patch)."""
        return self._run_bash(command, timeout=timeout)
