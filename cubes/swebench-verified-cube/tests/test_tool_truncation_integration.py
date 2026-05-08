"""Integration tests for bash output truncation via the real SWEBenchTool.bash() path.

Unlike test_tool_truncation.py which tests the truncation logic in isolation,
these tests call bash() through the real code path: bash() → _run_bash() →
container.exec() → _truncate_output(). The container is a MagicMock that
returns controlled ExecResult objects — no Docker needed.

This catches regressions where the logic is correct but the call path breaks
(e.g. bash() stops calling _truncate_output, or _run_bash output format changes).
"""

from __future__ import annotations

from unittest.mock import MagicMock


from cube.container import ExecResult
from swebench_verified_cube.tool import SWEBenchTool, SWEBenchToolConfig


def _make_tool(max_output_bytes: int = 100) -> tuple[SWEBenchTool, MagicMock]:
    """Return (tool, container_mock) with configured max_output_bytes."""
    config = SWEBenchToolConfig(max_output_bytes=max_output_bytes)
    container = MagicMock()
    tool = SWEBenchTool(config=config, container=container)
    return tool, container


def _set_output(container: MagicMock, stdout: str, stderr: str = "", exit_code: int = 0) -> None:
    container.exec.return_value = ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)


class TestBashTruncationIntegration:
    """End-to-end: bash() → container.exec() → truncation in one call."""

    def test_short_output_passes_through_unchanged(self) -> None:
        """Output under the limit is returned verbatim from bash()."""
        tool, container = _make_tool(max_output_bytes=200)
        _set_output(container, stdout="hello world")
        result = tool.bash("echo hello world")
        assert result == "hello world"

    def test_oversized_stdout_is_truncated(self) -> None:
        """bash() truncates stdout that exceeds max_output_bytes."""
        tool, container = _make_tool(max_output_bytes=100)
        _set_output(container, stdout="a" * 300)
        result = tool.bash("cat big_file")
        assert "bytes elided" in result
        assert len(result.encode("utf-8")) < 300

    def test_tail_content_present_after_truncation(self) -> None:
        """The tail of long stdout is preserved — not just the head."""
        tool, container = _make_tool(max_output_bytes=100)
        tail_marker = "PYTEST_SUMMARY_MARKER"
        _set_output(container, stdout="x" * 500 + tail_marker)
        result = tool.bash("pytest")
        assert tail_marker in result

    def test_stderr_appended_before_truncation(self) -> None:
        """stderr is joined with stdout before truncation — both can be visible."""
        tool, container = _make_tool(max_output_bytes=200)
        _set_output(container, stdout="out " * 10, stderr="ERR_MARKER", exit_code=1)
        result = tool.bash("bad_cmd")
        assert "ERR_MARKER" in result

    def test_nonzero_exit_code_annotation_visible(self) -> None:
        """[exit_code: N] annotation is appended — survives truncation at tail."""
        tool, container = _make_tool(max_output_bytes=100)
        # Long stdout + non-zero exit: the exit_code annotation is appended last,
        # so it lands in the tail portion.
        _set_output(container, stdout="x" * 500, exit_code=2)
        result = tool.bash("failing_cmd")
        assert "[exit_code: 2]" in result

    def test_container_called_with_correct_workdir(self) -> None:
        """bash() passes the configured working_dir to container.exec."""
        tool, container = _make_tool()
        _set_output(container, stdout="ok")
        tool.bash("ls")
        call_kwargs = container.exec.call_args
        assert call_kwargs.kwargs.get("workdir") == "/testbed"

    def test_combined_stdout_stderr_total_truncation(self) -> None:
        """When stdout+stderr together exceed the limit both get truncated correctly."""
        tool, container = _make_tool(max_output_bytes=100)
        _set_output(container, stdout="S" * 300, stderr="E" * 300, exit_code=1)
        result = tool.bash("cmd")
        assert "bytes elided" in result
        # Both contributions are visible (head has S, tail has exit_code)
        assert result[0] == "S"

    def test_empty_output_returns_no_output_placeholder(self) -> None:
        """Empty stdout+stderr with exit_code=0 returns '(no output)'."""
        tool, container = _make_tool()
        _set_output(container, stdout="", stderr="")
        result = tool.bash("true")
        assert result == "(no output)"

    def test_timeout_exit_code_annotation(self) -> None:
        """exit_code=124 produces the timeout annotation, not [exit_code: 124]."""
        tool, container = _make_tool(max_output_bytes=1000)
        _set_output(container, stdout="partial output", exit_code=124)
        result = tool.bash("sleep 999", timeout=1)
        assert "timed out" in result
        assert "[exit_code: 124]" not in result
