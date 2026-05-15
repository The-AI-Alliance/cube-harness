"""Canonical MiniWoB benchmark configs.

from miniwob_cube import MINIWOB_CONFIGS
benchmark = MINIWOB_CONFIGS["default"]
"""

from cube.core import ConfigRegistry
from cube_browser_tool import PlaywrightConfig
from miniwob_cube.benchmark import MiniWobBenchmarkConfig

MINIWOB_CONFIGS: ConfigRegistry[MiniWobBenchmarkConfig] = ConfigRegistry(
    {
        "default": MiniWobBenchmarkConfig(tool_config=PlaywrightConfig(use_screenshot=True, headless=True)),
    }
)
