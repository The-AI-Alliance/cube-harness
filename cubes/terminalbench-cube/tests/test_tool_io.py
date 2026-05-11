"""Unit tests for TerminalBenchTool — covers terminalbench-specific behaviour only.

General bash/truncation/write tests live in cube-standard's test_terminal_tool.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from cube.container import ExecResult

from terminalbench_cube.tool import TerminalBenchTool, TerminalBenchToolConfig


def _make_tool() -> tuple[TerminalBenchTool, MagicMock]:
    container = MagicMock()
    container.exec.return_value = ExecResult(stdout="ok", stderr="", exit_code=0, duration_seconds=0.0)
    config = TerminalBenchToolConfig()
    return TerminalBenchTool(config=config, container=container), container


# ── ms → s timeout normalisation (terminalbench-specific) ────────────────────


def test_bash_millisecond_timeout_normalised() -> None:
    """LLM passes timeout=120000 (ms); bash() divides by 1000 → 120s."""
    tool, container = _make_tool()
    tool.bash("echo ok", timeout=120_000)
    _, kwargs = container.exec.call_args
    assert kwargs["timeout"] == 120


def test_bash_reasonable_timeout_unchanged() -> None:
    tool, container = _make_tool()
    tool.bash("sleep 5", timeout=60)
    _, kwargs = container.exec.call_args
    assert kwargs["timeout"] == 60


def test_bash_timeout_still_capped_by_max_timeout() -> None:
    """max_timeout=900 from TerminalBenchToolConfig clamps large timeouts."""
    tool, container = _make_tool()
    tool.bash("sleep 1800", timeout=1_800)
    _, kwargs = container.exec.call_args
    assert kwargs["timeout"] == 900
