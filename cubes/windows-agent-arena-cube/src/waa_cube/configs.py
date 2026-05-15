"""Canonical Windows-Agent-Arena benchmark configs.

from waa_cube import WAA_CONFIGS
benchmark = WAA_CONFIGS["default"]
"""

from waa_cube.benchmark import WAABenchmark
from waa_cube.computer import ComputerConfig

WAA_CONFIGS: dict[str, WAABenchmark] = {
    "default": WAABenchmark(default_tool_config=ComputerConfig()),
}
