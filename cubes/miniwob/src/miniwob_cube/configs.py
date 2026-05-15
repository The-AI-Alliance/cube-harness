"""Canonical MiniWoB benchmark configs.

    from miniwob_cube import MINIWOB_CONFIGS
    benchmark = MINIWOB_CONFIGS["default"]

MiniWoB runs on BrowserGym with html + screenshot (html, not axtree, is the
canonical MiniWoB observation).
"""

# TEMPORARY (tools-architecture Phase 1): BrowsergymConfig still lives in
# cube-harness. This cube -> cube-harness import is a known, accepted
# exception until it moves to cube-tools/cube-browsergym-tool. Functionally
# safe (no import cycle); do not copy this pattern to new cubes.
from cube.core import ConfigRegistry
from cube_harness.tools.browsergym import BrowsergymConfig
from miniwob_cube.benchmark import MiniWobBenchmarkConfig

MINIWOB_CONFIGS: ConfigRegistry[MiniWobBenchmarkConfig] = ConfigRegistry(
    {
        "default": MiniWobBenchmarkConfig(
            tool_config=BrowsergymConfig(use_html=True, use_axtree=False, use_screenshot=True)
        ),
    }
)
