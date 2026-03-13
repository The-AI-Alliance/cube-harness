"""VM backend for OSWorld — re-exports from cube-standard and cube-vm-backend.

The concrete QEMU implementation now lives in cube_vm_backend (cube-tools).
This module re-exports the public surface for backwards compatibility.

Usage::

    from osworld_cube.vm_backend import LocalQEMUVMBackend, VMConfig
    backend = LocalQEMUVMBackend(cache_dir="/path/to/cache")
    vm = backend.launch(VMConfig(os_type="Ubuntu"))
    # ... use vm ...
    vm.stop()
"""

from cube.vm import VM, VMBackend, VMConfig
from cube_vm_backend import LocalQEMUVM, LocalQEMUVMBackend
from cube_vm_backend.qemu_manager import QEMUConfig, QEMUManager

# Backwards-compatibility alias — old code referenced VMInstance as QEMUManager
VMInstance = QEMUManager

__all__ = [
    "VM",
    "VMBackend",
    "VMConfig",
    "VMInstance",
    "LocalQEMUVM",
    "LocalQEMUVMBackend",
    "QEMUConfig",
    "QEMUManager",
]
