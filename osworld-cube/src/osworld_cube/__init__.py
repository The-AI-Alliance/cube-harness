"""osworld-cube: OSWorld benchmark integration for cube-standard and agentlab2."""

from osworld_cube.benchmark import OSWORLD_SYSTEM_PROMPT_COMPUTER_13, OSWorldBenchmark
from osworld_cube.config import OSWorldComputerConfig
from osworld_cube.task import OSWorldTask
from osworld_cube.vm_utils import OSWORLD_DOCKER_IMAGE, get_osworld_vm_image

__all__ = [
    "OSWorldBenchmark",
    "OSWorldTask",
    "OSWorldComputerConfig",
    "OSWORLD_SYSTEM_PROMPT_COMPUTER_13",
    "OSWORLD_DOCKER_IMAGE",
    "get_osworld_vm_image",
]
