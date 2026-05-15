"""Canonical WorkArena benchmark configs.

    from workarena_cube import WORKARENA_CONFIGS
    benchmark = WORKARENA_CONFIGS["default"]

Uses the real browser tool (not the debug cheat tool). Clone the recipe to
pick a level subset, e.g. `WORKARENA_CONFIGS["default"].named_subset("l1")`.
"""

from cube.core import ConfigRegistry
from workarena_cube.benchmark import WorkArenaBenchmarkConfig
from workarena_cube.tools import WorkarenaBrowserToolConfig

WORKARENA_CONFIGS: ConfigRegistry[WorkArenaBenchmarkConfig] = ConfigRegistry(
    {
        "default": WorkArenaBenchmarkConfig(tool_config=WorkarenaBrowserToolConfig()),
    }
)
