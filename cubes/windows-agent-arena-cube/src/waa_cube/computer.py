"""ComputerConfig re-export with WAA-specific cache defaults."""

from pathlib import Path

import cube

from cube_computer_tool.computer import ActionSpace, Computer13, ComputerBase, ComputerConfig as _BaseComputerConfig, PyAutoGUIComputer

_CUBE_CACHE_ROOT = cube.get_cache_dir("waa-cube")


class ComputerConfig(_BaseComputerConfig):
    """ComputerConfig pre-configured with waa-cube cache directory."""

    cache_dir: str = str(_CUBE_CACHE_ROOT / "cache")


__all__ = ["ActionSpace", "Computer13", "ComputerBase", "ComputerConfig", "PyAutoGUIComputer"]
