"""Canonical WebArena-Verified benchmark configs.

    from webarena_verified_cube import WEBARENA_CONFIGS
    benchmark = WEBARENA_CONFIGS["default"]

WEBARENA_ALL is the primary resource (one provision/launch for all sites).
Runs on BrowserGym with axtree + screenshot (the canonical web observation).
"""

# TEMPORARY (tools-architecture Phase 1): BrowsergymConfig still lives in
# cube-harness. This cube -> cube-harness import is a known, accepted
# exception until it moves to cube-tools/cube-browsergym-tool. Functionally
# safe (no import cycle); do not copy this pattern to new cubes.
from cube.core import ConfigRegistry
from cube_harness.tools.browsergym import BrowsergymConfig
from webarena_verified_cube.benchmark import WebArenaVerifiedBenchmarkConfig
from webarena_verified_cube.resources import WEBARENA_ALL

WEBARENA_CONFIGS: ConfigRegistry[WebArenaVerifiedBenchmarkConfig] = ConfigRegistry(
    {
        "default": WebArenaVerifiedBenchmarkConfig(
            tool_config=BrowsergymConfig(use_html=False, use_axtree=True, use_screenshot=True),
            resources=[WEBARENA_ALL],
        ),
    }
)
