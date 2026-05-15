"""Canonical WebArena-Verified benchmark configs.

    from webarena_verified_cube import WEBARENA_CONFIGS
    benchmark = WEBARENA_CONFIGS["default"]

WEBARENA_ALL is the primary resource (one provision/launch for all sites).
The benchmark's default tool_config is used; clone the recipe to swap it.
"""

from webarena_verified_cube.benchmark import WebArenaVerifiedBenchmarkConfig
from webarena_verified_cube.resources import WEBARENA_ALL

WEBARENA_CONFIGS: dict[str, WebArenaVerifiedBenchmarkConfig] = {
    "default": WebArenaVerifiedBenchmarkConfig(resources=[WEBARENA_ALL]),
}
