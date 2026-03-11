"""OSWorldComputerConfig: agentlab2-compatible config that uses cube-computer-tool.

Bridges agentlab2's ToolConfig interface (required by Benchmark base class) with
cube-computer-tool's Computer implementation. Provides OSWorld-specific defaults
(Docker image, auto-download of qcow2).
"""

from __future__ import annotations

from typing import Optional

from agentlab2.tool import ToolConfig
from cube.vm import VMConfig
from cube_computer_tool import ComputerConfig
from cube_computer_tool.backends.local_qemu import LocalQEMUVMBackend

from osworld_cube.vm_utils import OSWORLD_DOCKER_IMAGE, get_osworld_vm_image


class OSWorldComputerConfig(ToolConfig):
    """agentlab2-compatible config for running OSWorld tasks via cube-computer-tool.

    Provides sensible defaults for OSWorld:
    - Docker image: ``happysixd/osworld-docker``
    - qcow2 auto-downloaded from HuggingFace if vm_image_path is not set

    Usage::

        config = OSWorldComputerConfig()  # auto-download qcow2 on make()
        # or
        config = OSWorldComputerConfig(vm_image_path="/path/to/ubuntu.qcow2")

        tool = config.make()  # launches Docker container + returns Computer
    """

    vm_image_path: Optional[str] = None
    """Path to the Ubuntu qcow2 disk image. Auto-downloaded if None."""

    docker_image: str = OSWORLD_DOCKER_IMAGE
    """Docker image that bundles QEMU + the Flask control server."""

    headless: bool = True
    """Run QEMU without a graphical display."""

    require_a11y_tree: bool = True
    """Include accessibility tree in observations."""

    require_terminal: bool = False
    """Include terminal output in observations."""

    observe_after_action: bool = True
    """Capture full observation after each agent action."""

    def make(self, container=None) -> "Computer":  # type: ignore[override]
        """Launch the Docker+QEMU VM and return a ready Computer tool.

        Args:
            container: Ignored (OSWorld manages its own VM lifecycle).

        Returns:
            A Computer instance connected to the running VM.
        """
        from cube_computer_tool.computer import Computer

        vm_image = self.vm_image_path or get_osworld_vm_image()

        return ComputerConfig(
            vm_backend=LocalQEMUVMBackend(
                vm_image_path=vm_image,
                docker_image=self.docker_image,
                headless=self.headless,
            ),
            vm_config=VMConfig(),
            require_a11y_tree=self.require_a11y_tree,
            require_terminal=self.require_terminal,
            observe_after_action=self.observe_after_action,
        ).make()
