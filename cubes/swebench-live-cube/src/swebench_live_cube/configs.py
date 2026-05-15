"""Canonical SWE-bench Live benchmark configs.

from swebench_live_cube import SWEBENCH_LIVE_CONFIGS
benchmark = SWEBENCH_LIVE_CONFIGS["default"]
"""

from swebench_live_cube.benchmark import SWEBenchLiveBenchmarkConfig

SWEBENCH_LIVE_CONFIGS: dict[str, SWEBenchLiveBenchmarkConfig] = {
    "default": SWEBenchLiveBenchmarkConfig(),
}
