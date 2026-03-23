"""
Computer tool re-exports for osworld-cube.

The implementation has moved to cube-computer-tool (cube_computer_tool).
This module re-exports everything for backwards compatibility and adds
the osworld-cube cache root so ComputerConfig.cache_dir defaults sensibly.
"""

import cube

from cube_computer_tool.computer import ActionSpace, Computer13, ComputerBase, PyAutoGUIComputer
from cube_computer_tool.computer import ComputerConfig as _BaseComputerConfig

_CUBE_CACHE_ROOT = cube.get_cache_dir("osworld-cube")


class ComputerConfig(_BaseComputerConfig):
    """ComputerConfig with osworld-cube cache default."""

    cache_dir: str = str(_CUBE_CACHE_ROOT / "cache")


__all__ = [
    "ActionSpace",
    "Computer13",
    "ComputerBase",
    "ComputerConfig",
    "PyAutoGUIComputer",
    "_CUBE_CACHE_ROOT",
]
