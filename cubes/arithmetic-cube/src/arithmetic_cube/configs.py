"""Canonical Arithmetic benchmark configs.

from arithmetic_cube import ARITHMETIC_CONFIGS
benchmark = ARITHMETIC_CONFIGS["default"]
"""

from arithmetic_cube.benchmark import ArithmeticBenchmarkConfig

ARITHMETIC_CONFIGS: dict[str, ArithmeticBenchmarkConfig] = {
    "default": ArithmeticBenchmarkConfig(),
}
