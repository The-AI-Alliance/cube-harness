"""Canonical OSWorld benchmark configs.

    from osworld_cube import OSWORLD_CONFIGS
    benchmark = OSWORLD_CONFIGS["default"]

Zero-arg OSWorldBenchmarkConfig defaults to ComputerConfig() and the
OSWORLD_UBUNTU resource.
"""

from cube.core import ConfigRegistry
from osworld_cube.benchmark import OSWorldBenchmarkConfig

OSWORLD_CONFIGS: ConfigRegistry[OSWorldBenchmarkConfig] = ConfigRegistry(
    {
        "default": OSWorldBenchmarkConfig(),
    }
)
