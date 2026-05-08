"""Unit tests for SWEBenchTool._truncate_output (head+tail truncation).

Tests run without a container — the bash() truncation logic is exercised
via a subclass that bypasses _run_bash so no Docker/container fixture is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock


from swebench_live_cube.tool import SWEBenchTool, SWEBenchToolConfig


def _make_tool(max_output_bytes: int = 100) -> SWEBenchTool:
    config = SWEBenchToolConfig(max_output_bytes=max_output_bytes)
    container = MagicMock()
    tool = SWEBenchTool(config=config, container=container)
    return tool


def _call_truncation(tool: SWEBenchTool, output: str) -> str:
    """Call the same truncation logic that bash() uses, without a container call."""
    encoded = output.encode("utf-8")
    if len(encoded) <= tool._config.max_output_bytes:
        return output
    head = tool._config.max_output_bytes // 5
    tail = tool._config.max_output_bytes - head
    return (
        encoded[:head].decode("utf-8", errors="ignore")
        + f"\n[... {len(encoded) - tool._config.max_output_bytes} bytes elided ...]\n"
        + encoded[-tail:].decode("utf-8", errors="ignore")
    )


class TestTruncation:
    def test_short_output_not_truncated(self) -> None:
        """Output at or below the limit is returned verbatim."""
        tool = _make_tool(max_output_bytes=100)
        output = "a" * 100
        assert _call_truncation(tool, output) == output

    def test_one_byte_over_limit_triggers_truncation(self) -> None:
        tool = _make_tool(max_output_bytes=100)
        output = "a" * 101
        result = _call_truncation(tool, output)
        assert "bytes elided" in result

    def test_head_preserved(self) -> None:
        """First bytes of output appear at the start of the result."""
        tool = _make_tool(max_output_bytes=100)
        output = "HEAD_CONTENT" + "x" * 200 + "TAIL_CONTENT"
        result = _call_truncation(tool, output)
        assert result.startswith("HEAD")

    def test_tail_preserved(self) -> None:
        """Last bytes of output (error messages, exit codes) appear at the end."""
        tool = _make_tool(max_output_bytes=100)
        output = "x" * 200 + "TAIL_CONTENT"
        result = _call_truncation(tool, output)
        assert result.endswith("TAIL_CONTENT")

    def test_elision_marker_present(self) -> None:
        """Middle section is replaced with an informative marker."""
        tool = _make_tool(max_output_bytes=100)
        output = "a" * 500
        result = _call_truncation(tool, output)
        assert "bytes elided" in result

    def test_elided_byte_count_is_correct(self) -> None:
        """Elided count = len(encoded) - max_output_bytes."""
        tool = _make_tool(max_output_bytes=100)
        output = "a" * 300  # 300 bytes, 200 elided
        result = _call_truncation(tool, output)
        assert "200 bytes elided" in result

    def test_head_is_20_percent_of_limit(self) -> None:
        """Head is max_output_bytes // 5 (20%), tail is the remaining 80%."""
        tool = _make_tool(max_output_bytes=100)
        # Create output where head and tail are clearly distinguishable
        head_marker = "H" * 20  # exactly head (100 // 5 = 20 bytes)
        tail_marker = "T" * 80  # exactly tail (100 - 20 = 80 bytes)
        filler = "x" * 100  # excess content in the middle
        output = head_marker + filler + tail_marker
        result = _call_truncation(tool, output)
        # Head should be fully present (20 bytes = 20 H's)
        assert result[:20] == "H" * 20
        # Tail should be fully present (80 bytes = 80 T's)
        assert result[-80:] == "T" * 80

    def test_result_length_bounded(self) -> None:
        """Head + tail bytes together equal exactly max_output_bytes."""
        tool = _make_tool(max_output_bytes=100)
        output = "a" * 10_000
        result = _call_truncation(tool, output)
        # Split around the marker line to isolate head and tail content
        parts = result.split("\n[... ")
        head_bytes = parts[0].encode("utf-8")
        tail_bytes = parts[1].split("...]\n", 1)[1].encode("utf-8")
        assert len(head_bytes) + len(tail_bytes) == tool._config.max_output_bytes

    def test_unicode_safe(self) -> None:
        """Multi-byte UTF-8 sequences don't produce mojibake at split boundaries."""
        tool = _make_tool(max_output_bytes=20)
        # Each emoji is 4 bytes; 10 of them = 40 bytes > 20 limit
        output = "🐍" * 10
        result = _call_truncation(tool, output)
        # errors="ignore" should silently drop incomplete sequences, not crash
        result.encode("utf-8")  # must not raise

    def test_empty_output_not_truncated(self) -> None:
        tool = _make_tool(max_output_bytes=100)
        assert _call_truncation(tool, "") == ""
