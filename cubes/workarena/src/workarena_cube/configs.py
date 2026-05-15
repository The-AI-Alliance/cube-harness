"""Canonical WorkArena benchmark configs.

    from workarena_cube import WORKARENA_CONFIGS
    benchmark = WORKARENA_CONFIGS["l1"]

"default" is all levels with the real browser tool. "l1"/"l2"/"l3" are the
canonical WorkArena difficulty splits; l2/l3 add the infeasible-task tool,
matching how the benchmark is run in practice.
"""

from cube.core import ConfigRegistry
from cube.tool import ToolboxConfig
from workarena_cube.benchmark import WorkArenaBenchmarkConfig
from workarena_cube.tools import WorkArenaInfeasibleToolConfig, WorkarenaBrowserToolConfig


def _browser() -> WorkarenaBrowserToolConfig:
    return WorkarenaBrowserToolConfig()


def _browser_with_infeasible() -> ToolboxConfig:
    return ToolboxConfig(tool_configs=[WorkarenaBrowserToolConfig(), WorkArenaInfeasibleToolConfig()])


WORKARENA_CONFIGS: ConfigRegistry[WorkArenaBenchmarkConfig] = ConfigRegistry(
    {
        "default": WorkArenaBenchmarkConfig(tool_config=_browser()),
        "l1": WorkArenaBenchmarkConfig(tool_config=_browser()).named_subset("l1"),
        "l2": WorkArenaBenchmarkConfig(tool_config=_browser_with_infeasible()).named_subset("l2"),
        "l3": WorkArenaBenchmarkConfig(tool_config=_browser_with_infeasible()).named_subset("l3"),
    }
)
