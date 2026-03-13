"""VM backend for OSWorld: bare QEMU/KVM replacing desktop_env."""

from pathlib import Path
from typing import Literal

import cube
from pydantic import BaseModel

from osworld_cube.vm_backend.qemu_manager import QEMUConfig, QEMUManager, ensure_base_image

_CUBE_CACHE_ROOT = cube.get_cache_dir("osworld-cube")

# A running VM instance — callers hold this handle and pass it wherever needed.
VMInstance = QEMUManager


class VMConfig(BaseModel):
    """Configuration for launching a standalone VM instance.

    Parameters
    ----------
    vm_dir : str
        Directory holding the base qcow2 image and per-task overlays.
    path_to_vm : str | None
        Explicit path to the base qcow2 image. If None, the image is
        auto-downloaded to vm_dir on first use.
    os_type : "Ubuntu" | "Windows"
        OS type used for base image download when path_to_vm is None.
    memory : str
        RAM allocation passed to QEMU ``-m`` (e.g. ``"4G"``).
    cpus : int
        Number of vCPUs.
    headless : bool
        Suppress the graphical display.
    screen_size : tuple[int, int]
        (width, height) resolution.
    """

    vm_dir: str = str(_CUBE_CACHE_ROOT / "vm_data")
    path_to_vm: str | None = None
    os_type: Literal["Ubuntu", "Windows"] = "Ubuntu"
    memory: str = "4G"
    cpus: int = 4
    headless: bool = True
    screen_size: tuple[int, int] = (1920, 1080)


def launch(vm_config: VMConfig) -> VMInstance:
    """Launch a VM and return the running instance handle.

    Example::

        vm = launch(VMConfig(memory="4G", cpus=4))
        computer = ComputerConfig(...).make(vm=vm)
        # ...
        vm.stop()

    Parameters
    ----------
    vm_config : VMConfig
        VM configuration.

    Returns
    -------
    VMInstance
        A started QEMUManager. The caller owns the lifecycle (call ``vm.stop()``
        when done).
    """
    vm_dir = Path(vm_config.vm_dir)
    if vm_config.path_to_vm is not None:
        base_image = Path(vm_config.path_to_vm)
    else:
        base_image = ensure_base_image(vm_dir, vm_config.os_type)

    qemu_config = QEMUConfig(
        base_image=base_image,
        overlay_dir=vm_dir / "overlays",
        memory=vm_config.memory,
        cpus=vm_config.cpus,
        headless=vm_config.headless,
        screen_width=vm_config.screen_size[0],
        screen_height=vm_config.screen_size[1],
    )
    vm = QEMUManager(qemu_config)
    vm.start()
    return vm
