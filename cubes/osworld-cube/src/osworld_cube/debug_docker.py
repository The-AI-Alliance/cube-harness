"""
Debug agent for testing OSWorldTask end-to-end with the Docker backend.

Mirrors osworld_cube.debug but uses OSWorldDockerVMBackend (QEMU-in-Docker)
instead of OSWorldQEMUVMBackend. Works on macOS via Docker Desktop.

Usage::

    # Run all debug tasks with the Docker backend
    python -m osworld_cube.debug_docker
"""

import logging
import sys
import types

import osworld_cube.debug as _debug_mod
from cube.testing import run_debug_suite
from osworld_cube.vm_backend import OSWorldDockerVMBackend

logger = logging.getLogger(__name__)


def get_debug_benchmark() -> object:
    """Return the debug benchmark wired to the Docker backend."""
    return _debug_mod.get_debug_benchmark(vm_backend=OSWorldDockerVMBackend())


def make_debug_agent(task_id: str) -> _debug_mod.DebugAgent:
    """Return a fresh DebugAgent for the given task_id."""
    return _debug_mod.make_debug_agent(task_id)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    _mod = types.ModuleType("osworld_cube.debug_docker")
    _mod.get_debug_benchmark = get_debug_benchmark
    _mod.make_debug_agent = make_debug_agent

    results = run_debug_suite("osworld-cube", _mod)
    failed = [r for r in results if r.get("error") or not r.get("done") or r.get("reward", 0) <= 0]
    sys.exit(1 if failed else 0)
