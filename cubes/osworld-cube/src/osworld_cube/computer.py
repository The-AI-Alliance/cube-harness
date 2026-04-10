"""
Computer tool re-exports for osworld-cube.

The implementation has moved to cube-computer-tool (cube_computer_tool).
This module re-exports everything for backwards compatibility and sets
ComputerConfig.cache_dir to the osworld-cube cache directory.
"""

from osworld_cube._paths import OSWORLD_CACHE_DIR
from cube_computer_tool.computer import ActionSpace, Computer13, ComputerBase, PyAutoGUIComputer
from cube_computer_tool.computer import ComputerConfig as _BaseComputerConfig


class ComputerConfig(_BaseComputerConfig):
    """ComputerConfig with osworld-cube cache default."""

    cache_dir: str = str(OSWORLD_CACHE_DIR)


__all__ = [
    "ActionSpace",
    "Computer13",
    "ComputerBase",
    "ComputerConfig",
    "PyAutoGUIComputer",
]
