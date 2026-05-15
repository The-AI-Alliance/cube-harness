"""Canonical WorkArena benchmark configs.

    from workarena_cube import WORKARENA_CONFIGS
    benchmark = WORKARENA_CONFIGS["l1"]

Runs on BrowserGym with axtree + screenshot (the canonical web observation).
"default" is all levels; "l1"/"l2"/"l3" are the canonical WorkArena
difficulty splits; l2/l3 add the infeasible-task tool, matching how the
benchmark is run in practice.
"""

# TEMPORARY (tools-architecture Phase 1): BrowsergymConfig still lives in
# cube-harness. This cube -> cube-harness import is a known, accepted
# exception until it moves to cube-tools/cube-browsergym-tool. Functionally
# safe (no import cycle); do not copy this pattern to new cubes.
from cube.core import ConfigRegistry
from cube.tool import ToolboxConfig
from cube_harness.tools.browsergym import BrowsergymConfig
from workarena_cube.benchmark import WorkArenaBenchmarkConfig
from workarena_cube.tools import WorkArenaInfeasibleToolConfig


def _browser() -> BrowsergymConfig:
    return BrowsergymConfig(use_html=False, use_axtree=True, use_screenshot=True)


def _browser_with_infeasible() -> ToolboxConfig:
    return ToolboxConfig(tool_configs=[_browser(), WorkArenaInfeasibleToolConfig()])


WORKARENA_CONFIGS: ConfigRegistry[WorkArenaBenchmarkConfig] = ConfigRegistry(
    {
        "default": WorkArenaBenchmarkConfig(tool_config=_browser()),
        "l1": WorkArenaBenchmarkConfig(tool_config=_browser()).named_subset("l1"),
        "l2": WorkArenaBenchmarkConfig(tool_config=_browser_with_infeasible()).named_subset("l2"),
        "l3": WorkArenaBenchmarkConfig(tool_config=_browser_with_infeasible()).named_subset("l3"),
    }
)
