"""Unit tests for TerminalBenchTool I/O hardening (PR #362).

Covers:
- bash() ms→s timeout normalisation + 900s cap
- bash() head+tail truncation for large output
- write_file() error propagation when mkdir or write fails
"""

from __future__ import annotations

from unittest.mock import MagicMock

from cube.container import ExecResult

from terminalbench_cube.tool import TerminalBenchTool, TerminalBenchToolConfig


def _make_tool(max_output_bytes: int = 100_000) -> tuple[TerminalBenchTool, MagicMock]:
    container = MagicMock()
    config = TerminalBenchToolConfig(working_dir="/app", max_output_bytes=max_output_bytes)
    return TerminalBenchTool(config=config, container=container), container


def _ok(stdout: str = "", stderr: str = "") -> ExecResult:
    return ExecResult(stdout=stdout, stderr=stderr, exit_code=0, duration_seconds=0.0)


def _fail(stderr: str = "error") -> ExecResult:
    return ExecResult(stdout="", stderr=stderr, exit_code=1, duration_seconds=0.0)


# ── bash() timeout normalisation ─────────────────────────────────────────────


def test_bash_millisecond_timeout_converted_to_seconds() -> None:
    """LLM passes timeout=120000 (ms); bash() divides by 1000 → 120s."""
    tool, container = _make_tool()
    container.exec.return_value = _ok("ok")

    tool.bash("echo ok", timeout=120_000)

    _, kwargs = container.exec.call_args
    assert kwargs["timeout"] == 120


def test_bash_timeout_capped_at_900s() -> None:
    """Timeout > 900s (but ≤ 3600s so not ms-converted) is clamped to 900s."""
    tool, container = _make_tool()
    container.exec.return_value = _ok("ok")

    tool.bash("sleep 1800", timeout=1_800)

    _, kwargs = container.exec.call_args
    assert kwargs["timeout"] == 900


def test_bash_reasonable_timeout_unchanged() -> None:
    """Timeout of 60s is passed through unchanged."""
    tool, container = _make_tool()
    container.exec.return_value = _ok("ok")

    tool.bash("sleep 5", timeout=60)

    _, kwargs = container.exec.call_args
    assert kwargs["timeout"] == 60


# ── bash() head+tail truncation ──────────────────────────────────────────────


def test_bash_small_output_returned_verbatim() -> None:
    """Output under max_output_bytes is returned as-is."""
    tool, container = _make_tool(max_output_bytes=1000)
    container.exec.return_value = _ok("hello")

    result = tool.bash("echo hello")

    assert result == "hello"


def test_bash_large_output_uses_head_tail() -> None:
    """Output > max_output_bytes shows both head and tail, not just head."""
    tool, container = _make_tool(max_output_bytes=100)
    big = "A" * 6_000 + "B" * 6_000
    container.exec.return_value = _ok(stdout=big)

    result = tool.bash("cat big_file")

    assert "AAAA" in result  # head preserved
    assert "BBBB" in result  # tail preserved — the key regression check
    assert "bytes elided" in result


# ── write_file() error propagation ───────────────────────────────────────────


def test_write_file_error_on_mkdir_failure() -> None:
    """Failed mkdir surfaces as an error string, not 'Wrote N bytes'."""
    tool, container = _make_tool()
    container.exec.return_value = _fail("Read-only filesystem")

    result = tool.write_file("/app/out.py", "print('hi')")

    assert "Error creating parent dir" in result
    assert "Wrote" not in result


def test_write_file_error_on_write_failure() -> None:
    """Failed write (after successful mkdir) also surfaces as an error string."""
    tool, container = _make_tool()
    mkdir_ok = ExecResult(stdout="", stderr="", exit_code=0, duration_seconds=0.0)
    write_fail = ExecResult(stdout="", stderr="Permission denied", exit_code=1, duration_seconds=0.0)
    container.exec.side_effect = [mkdir_ok, write_fail]

    result = tool.write_file("/app/out.py", "print('hi')")

    assert "Error writing" in result
    assert "Wrote" not in result


def test_write_file_success_reports_bytes() -> None:
    """Successful write returns 'Wrote N bytes to path'."""
    tool, container = _make_tool()
    container.exec.return_value = ExecResult(stdout="", stderr="", exit_code=0, duration_seconds=0.0)

    result = tool.write_file("/tmp/out.py", "x = 1\n")

    assert "Wrote" in result
    assert "/tmp/out.py" in result
