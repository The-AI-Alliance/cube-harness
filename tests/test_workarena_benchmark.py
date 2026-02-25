"""Unit tests for WorkArenaBenchmark tool config normalization."""

from __future__ import annotations

import pytest

from agentlab2.benchmarks.workarena import WorkArenaBenchmark
from agentlab2.tools.browsergym import BrowsergymConfig
from agentlab2.tools.chat import ChatConfig
from agentlab2.tools.toolbox import ToolboxConfig


def test_workarena_benchmark_wraps_browsergym_with_chat_tool() -> None:
    benchmark = WorkArenaBenchmark(tool_config=BrowsergymConfig())

    assert isinstance(benchmark.tool_config, ToolboxConfig)
    assert any(isinstance(config, BrowsergymConfig) for config in benchmark.tool_config.tool_configs)
    assert any(isinstance(config, ChatConfig) for config in benchmark.tool_config.tool_configs)


def test_workarena_benchmark_appends_chat_to_existing_toolbox() -> None:
    benchmark = WorkArenaBenchmark(tool_config=ToolboxConfig(tool_configs=[BrowsergymConfig()]))

    assert isinstance(benchmark.tool_config, ToolboxConfig)
    assert sum(isinstance(config, ChatConfig) for config in benchmark.tool_config.tool_configs) == 1


def test_workarena_benchmark_requires_browsergym_in_toolbox() -> None:
    with pytest.raises(ValueError, match="requires BrowsergymConfig"):
        WorkArenaBenchmark(tool_config=ToolboxConfig(tool_configs=[ChatConfig()]))
