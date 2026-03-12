"""
Computer tool — wraps cube-computer-tool for OSWorld VM-based desktop automation.

Replaces the desktop_env dependency with cube-computer-tool (LocalQEMUVMBackend +
HTTP Flask API).  The public interface is kept identical to the original so that
task.py, benchmark.py and debug.py require no changes.

    config = ComputerConfig()          # auto-downloads qcow2 on first make()
    tool   = config.make()             # launches Docker container, waits for VM
    obs    = tool.setup_task(task)     # restores snapshot, runs setup scripts
    reward = tool.evaluate_task()      # runs evaluator inside VM
    tool.close()                       # stops container
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import cube
from cube.tool import ToolConfig
from cube.vm import VMConfig
from cube_computer_tool.backends.local_qemu import LocalQEMUVMBackend
from cube_computer_tool.computer import Computer as _CCTComputer
from cube_computer_tool.computer import ComputerConfig as _CCTComputerConfig

logger = logging.getLogger(__name__)

# Root cache directory shared with benchmark.py
_CUBE_CACHE_ROOT = Path(cube.get_cache_dir("osworld-cube"))


# ---------------------------------------------------------------------------
# _OSWorldComputerProps — exposes config.os_type read by task.py
# ---------------------------------------------------------------------------


class _OSWorldComputerProps:
    """Minimal config object so task.py can access self._computer.config.os_type."""

    def __init__(self, os_type: str) -> None:
        self.os_type = os_type


# ---------------------------------------------------------------------------
# ComputerBase — the main tool class
# ---------------------------------------------------------------------------


class ComputerBase(_CCTComputer):
    """OSWorld-specific Computer that extends cube-computer-tool's Computer.

    Adds three things task.py needs that the base Computer doesn't have:
      - ``config`` property  → exposes ``os_type`` for axtree platform selection
      - ``update_marks()``   → stores SoM element bounding-box marks
      - ``evaluate_task()``  → implements ``check_include_exclude`` evaluator
                               (and can be extended for other OSWorld evaluators)
    """

    def __init__(
        self,
        config: _CCTComputerConfig,
        vm,
        os_type: str = "Ubuntu",
    ) -> None:
        super().__init__(config=config, vm=vm)
        self._osworld_props = _OSWorldComputerProps(os_type=os_type)
        self._marks: list = []

    # ------------------------------------------------------------------
    # Properties expected by task.py
    # ------------------------------------------------------------------

    @property
    def config(self) -> _OSWorldComputerProps:  # type: ignore[override]
        """Return a config-like object with .os_type for axtree linearization."""
        return self._osworld_props

    def update_marks(self, marks: list) -> None:
        """Store Set-of-Marks element bounding boxes (used by SoM annotation)."""
        self._marks = marks

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_task(self) -> float:
        """Run the task evaluator and return reward in [0.0, 1.0].

        Currently implements the ``check_include_exclude`` evaluator used by
        most OSWorld tasks: run a shell command inside the VM and check that
        the output contains all ``include`` strings and none of the ``exclude``
        strings.

        For evaluators that cannot be handled locally the method returns 0.0
        and logs a warning.  Extend this method (or override in a subclass) to
        support additional OSWorld evaluator types.
        """
        if self._task_config is None:
            logger.warning("evaluate_task() called before setup_task() — returning 0.0")
            return 0.0

        evaluator = self._task_config.get("evaluator", {})
        if not evaluator:
            logger.warning("Task %r has no evaluator — returning 0.0", self._task_config.get("id"))
            return 0.0

        func = evaluator.get("func", "")
        if func == "check_include_exclude":
            return self._eval_check_include_exclude(evaluator)

        logger.warning("Unsupported evaluator func %r — returning 0.0", func)
        return 0.0

    def _eval_check_include_exclude(self, evaluator: dict) -> float:
        result_spec = evaluator.get("result", {})
        expected = evaluator.get("expected", {})

        if result_spec.get("type") != "vm_command_line":
            logger.warning("check_include_exclude: unsupported result type %r", result_spec.get("type"))
            return 0.0

        output = self.run_shell_command(result_spec["command"])
        logger.info("Evaluator command %r → %r", result_spec["command"], output[:200])

        rules = expected.get("rules", {})
        reward = 1.0
        for s in rules.get("include", []):
            if s not in output:
                reward = 0.0
                break
        for s in rules.get("exclude", []):
            if s in output:
                reward = 0.0
                break
        return reward


# Aliases kept for __init__.py and external callers that import these names
Computer13 = ComputerBase
PyAutoGUIComputer = ComputerBase


# ---------------------------------------------------------------------------
# ComputerConfig — serialisable config; compatible with debug.py and task.py
# ---------------------------------------------------------------------------


class ComputerConfig(ToolConfig):
    """Serialisable configuration for ComputerBase.

    On ``make()``, auto-downloads the OSWorld qcow2 from HuggingFace (if not
    already cached), then launches the Docker+QEMU container and returns a
    ready ``ComputerBase`` tool.

    Fields kept compatible with the previous desktop_env-based interface:
        path_to_vm:          path to the Ubuntu.qcow2 disk image.  Auto-downloaded
                             from HuggingFace if None.
        os_type:             OS running in the VM ("Ubuntu" or "Windows").
        headless:            run QEMU without a graphical display.
        require_a11y_tree:   include accessibility tree in observations.
        require_terminal:    include terminal output in observations.
        observe_after_action: capture full observation after every action.
    """

    os_type: Literal["Ubuntu", "Windows"] = "Ubuntu"
    path_to_vm: Optional[str] = None
    headless: bool = True
    require_a11y_tree: bool = True
    require_terminal: bool = False
    observe_after_action: bool = True

    def make(self, container=None) -> ComputerBase:  # type: ignore[override]
        """Launch the Docker+QEMU VM and return a ready ComputerBase tool."""
        from osworld_cube.vm_utils import OSWORLD_DOCKER_IMAGE, get_osworld_vm_image

        vm_path = self.path_to_vm or get_osworld_vm_image()

        cct_config = _CCTComputerConfig(
            vm_backend=LocalQEMUVMBackend(
                vm_image_path=vm_path,
                docker_image=OSWORLD_DOCKER_IMAGE,
                headless=self.headless,
            ),
            vm_config=VMConfig(),
            require_a11y_tree=self.require_a11y_tree,
            require_terminal=self.require_terminal,
            observe_after_action=self.observe_after_action,
        )
        vm = cct_config.vm_backend.launch(cct_config.vm_config)
        return ComputerBase(config=cct_config, vm=vm, os_type=self.os_type)
