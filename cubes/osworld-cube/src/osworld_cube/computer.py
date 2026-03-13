"""
Computer tool — CUBE tool wrapping the vm_backend for VM-based desktop automation.

Two variants selected by ComputerConfig.action_space:
    Computer13        — 13 mouse/keyboard primitives + wait/done/fail
    PyAutoGUIComputer — run_pyautogui() code execution + wait/done/fail
"""

import logging
import time
from enum import Enum
from io import BytesIO
from pathlib import Path

from PIL import Image

import cube
from cube.container import Container
from cube.core import Action, Content, ImageContent, Observation, StepError, TextContent
from cube.tool import Tool, ToolConfig, tool_action

from osworld_cube.vm_backend import VMConfig, VMInstance
from osworld_cube.vm_backend.evaluator import Evaluator
from osworld_cube.vm_backend.guest_agent import GuestAgent
from osworld_cube.vm_backend.pyautogui_utils import fix_pyautogui_less_than_bug
from osworld_cube.vm_backend.qemu_manager import QEMUConfig, QEMUManager, ensure_base_image
from osworld_cube.vm_backend.setup_controller import SetupController

logger = logging.getLogger(__name__)

_CUBE_CACHE_ROOT = cube.get_cache_dir("osworld-cube")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ActionSpace(str, Enum):
    """Action space variants for the Computer tool."""

    COMPUTER_13 = "computer_13"
    PYAUTOGUI = "pyautogui"


# ---------------------------------------------------------------------------
# Config classes
# ---------------------------------------------------------------------------


class ComputerConfig(ToolConfig):
    """
    Serializable configuration for Computer tool variants.

    vm_config    — VM parameters (image path, memory, CPUs, screen size, etc.).
    cache_dir    — per-task reference file cache.

    action_space selects the tool variant:
      "computer_13" → Computer13 (13 mouse/keyboard primitives + wait/done/fail)
      "pyautogui"   → PyAutoGUIComputer (run_pyautogui + wait/done/fail)
    """

    action_space: ActionSpace = ActionSpace.COMPUTER_13
    vm_config: VMConfig = VMConfig()
    snapshot_name: str = "init_state"  # kept for API compatibility; overlays replace snapshots
    cache_dir: str = str(_CUBE_CACHE_ROOT / "cache")
    require_a11y_tree: bool = True
    require_terminal: bool = False
    observe_after_action: bool = True

    def make(self, container: Container | None = None, vm: VMInstance | None = None) -> "ComputerBase":
        if container is not None:
            logger.warning(
                "ComputerConfig.make() received a cube Container, but the OSWorld "
                "Computer tool manages its own VM via QEMUManager. "
                "The container argument will be ignored."
            )
        if self.action_space == ActionSpace.PYAUTOGUI:
            return PyAutoGUIComputer(self, vm=vm)
        return Computer13(self, vm=vm)


# ---------------------------------------------------------------------------
# ComputerBase — shared VM lifecycle and task helpers
# ---------------------------------------------------------------------------


class ComputerBase(Tool):
    """
    Shared base for Computer13 and PyAutoGUIComputer.

    Provides VM lifecycle (init, setup_task, evaluate_task, get_observation, close)
    and the three terminal @tool_action signals shared by both action spaces:
    wait, done, fail.

    Subclasses add the action-space-specific @tool_action methods.
    """

    def __init__(self, config: ComputerConfig, vm: VMInstance | None = None) -> None:
        self.config = config
        self._current_task_config: dict | None = None
        self._last_marks: list[list[int]] = []
        self._is_done: bool = False
        self._action_history: list = []

        if vm is not None:
            self._qemu = vm
        else:
            vc = config.vm_config
            vm_dir = Path(vc.vm_dir)
            base_image = Path(vc.path_to_vm) if vc.path_to_vm else ensure_base_image(vm_dir, vc.os_type)
            qemu_config = QEMUConfig(
                base_image=base_image,
                overlay_dir=vm_dir / "overlays",
                memory=vc.memory,
                cpus=vc.cpus,
                headless=vc.headless,
                screen_width=vc.screen_size[0],
                screen_height=vc.screen_size[1],
            )
            self._qemu = QEMUManager(qemu_config)
            self._qemu.start()

        self._guest = GuestAgent(host="localhost", port=self._qemu.server_port)

        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

        self._setup_ctrl = SetupController(
            guest=self._guest,
            chromium_port=self._qemu.chromium_port,
            vlc_port=self._qemu.vlc_port,
            cache_dir=config.cache_dir,
            screen_width=config.vm_config.screen_size[0],
            screen_height=config.vm_config.screen_size[1],
        )
        self._evaluator = Evaluator(
            guest=self._guest,
            cache_dir_base=Path(config.cache_dir),
            chromium_port=self._qemu.chromium_port,
            vlc_port=self._qemu.vlc_port,
            server_port=self._qemu.server_port,
        )

    def execute_action(self, action: Action) -> Observation | StepError:
        """Execute action; append full VM observation if observe_after_action=True."""
        action_obs = super().execute_action(action)

        if self.config.observe_after_action and action.name not in ("done", "fail"):
            action_obs += self.get_observation()

        return action_obs

    def setup_task(self, task_config: dict, seed: int | None = None) -> Observation:
        """
        Restore VM to initial state, run setup scripts, wait for stabilisation,
        then return the initial observation.

        Called by OSWorldTask.reset().

        Parameters
        ----------
        task_config : dict
            Task configuration dict with keys id, instruction, config, evaluator,
            snapshot, related_apps.
        seed : int | None
            Optional random seed (not used by QEMU backend; kept for API compat).
        """
        logger.info("Setting up task: %s", task_config.get("id", "unknown"))
        logger.info("Instruction: %s", task_config.get("instruction", ""))

        # Reset VM to clean state (stop → delete overlay → create overlay → boot)
        self._qemu.reset()

        # Run task-specific setup steps
        setup_steps = task_config.get("config") or []
        task_cache_dir = str(Path(self.config.cache_dir) / task_config.get("id", "task"))
        Path(task_cache_dir).mkdir(parents=True, exist_ok=True)
        self._setup_ctrl.reset_cache_dir(task_cache_dir)
        if setup_steps:
            self._setup_ctrl.setup(setup_steps)

        logger.info("Waiting 60s for VM to stabilise...")
        time.sleep(60)

        self._is_done = False
        self._action_history = []
        self._current_task_config = task_config
        return self.get_observation()

    def evaluate_task(self) -> float:
        """
        Run the task evaluator and return reward ∈ [0.0, 1.0].

        Called by OSWorldTask.evaluate(). Partial credit is preserved.
        """
        if self._current_task_config is None:
            logger.error("evaluate_task() called before setup_task()")
            return 0.0
        try:
            reward = self._evaluator.evaluate(self._current_task_config, self._action_history)
            logger.info("Task evaluation result: %f", reward)
            return reward
        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            return 0.0

    def get_observation(self) -> Observation:
        """Read current screen state from the VM and return as Observation."""
        raw_obs = {
            "screenshot": self._guest.get_screenshot(),
            "accessibility_tree": self._guest.get_accessibility_tree() if self.config.require_a11y_tree else None,
            "terminal": self._guest.get_terminal_output() if self.config.require_terminal else None,
        }
        return self._convert_observation(raw_obs)

    def _convert_observation(self, raw_obs: dict) -> Observation:
        """
        Convert VM observation dict to a cube Observation.

        Keys:
            "screenshot"         → bytes (PNG) → PIL.Image stored as ImageContent
            "accessibility_tree" → XML string  → TextContent (named "accessibility_tree")
            "terminal"           → str         → TextContent (named "terminal")
        """
        contents: list[Content] = []

        if raw_obs.get("screenshot"):
            img = Image.open(BytesIO(raw_obs["screenshot"])).convert("RGB")
            contents.append(ImageContent(data=img, name="screenshot"))

        if raw_obs.get("accessibility_tree"):
            contents.append(TextContent(data=raw_obs["accessibility_tree"], name="accessibility_tree"))

        if raw_obs.get("terminal"):
            contents.append(TextContent(data=raw_obs["terminal"], name="terminal"))

        return Observation(contents=contents)

    def _execute_desktop_action(self, action_dict: dict | str) -> str:
        """Send an action to the guest VM and return a success string."""
        if isinstance(action_dict, dict):
            self._guest.execute_action(action_dict)
        else:
            self._guest.execute_python_command(str(action_dict))
        self._action_history.append(action_dict)
        return "Success"

    def update_marks(self, marks: list[list[int]]) -> None:
        """Store SoM bounding-box marks for tag_N variable resolution in run_pyautogui."""
        self._last_marks = marks

    def reset(self) -> None:
        """Reset tool state between tasks (cube AbstractTool.reset() override)."""
        self._last_marks = []
        self._is_done = False
        self._action_history = []

    def close(self) -> None:
        """Shut down the VM and release resources."""
        logger.info("Closing desktop environment")
        self._qemu.stop()

    @tool_action
    def wait(self) -> str:
        """Wait one step without taking any action."""
        self._action_history.append("WAIT")
        return "Success"

    @tool_action
    def done(self) -> str:
        """Signal that the task has been completed successfully."""
        self._is_done = True
        self._action_history.append("DONE")
        return "Task marked as done"

    @tool_action
    def fail(self) -> str:
        """Signal that the task cannot be completed (infeasible or failed)."""
        self._is_done = True
        self._action_history.append("FAIL")
        return "Task marked as failed"


# ---------------------------------------------------------------------------
# Computer13 — 13 mouse/keyboard primitives
# ---------------------------------------------------------------------------


class Computer13(ComputerBase):
    """
    Desktop/VM computer tool with the computer_13 action space.

    Exposes 13 mouse/keyboard primitives as @tool_action methods, plus the
    shared wait/done/fail terminal signals inherited from ComputerBase.
    """

    @tool_action
    def click(
        self,
        button: str = "left",
        x: int = -1,
        y: int = -1,
        num_clicks: int = 1,
    ) -> str:
        """Click the mouse button at optional coordinates.

        Parameters
        ----------
        button : str
            Mouse button — "left", "right", or "middle"
        x : int
            X coordinate to click at (-1 = use current cursor position)
        y : int
            Y coordinate to click at (-1 = use current cursor position)
        num_clicks : int
            Number of clicks (1 for single, 2 for double, etc.)
        """
        params: dict = {"button": button, "num_clicks": num_clicks}
        if x >= 0:
            params["x"] = x
        if y >= 0:
            params["y"] = y
        return self._execute_desktop_action({"action_type": "CLICK", "parameters": params})

    @tool_action
    def double_click(self, x: int = -1, y: int = -1) -> str:
        """Double-click the mouse at optional coordinates.

        Parameters
        ----------
        x : int
            X coordinate (-1 = use current cursor position)
        y : int
            Y coordinate (-1 = use current cursor position)
        """
        return self.click(x=x, y=y, num_clicks=2)

    @tool_action
    def right_click(self, x: int = -1, y: int = -1) -> str:
        """Right-click the mouse at optional coordinates.

        Parameters
        ----------
        x : int
            X coordinate (-1 = use current cursor position)
        y : int
            Y coordinate (-1 = use current cursor position)
        """
        return self.click(button="right", x=x, y=y)

    @tool_action
    def mouse_down(self, button: str = "left") -> str:
        """Press and hold a mouse button.

        Parameters
        ----------
        button : str
            Mouse button — "left", "right", or "middle"
        """
        return self._execute_desktop_action({"action_type": "MOUSE_DOWN", "parameters": {"button": button}})

    @tool_action
    def mouse_up(self, button: str = "left") -> str:
        """Release a held mouse button.

        Parameters
        ----------
        button : str
            Mouse button — "left", "right", or "middle"
        """
        return self._execute_desktop_action({"action_type": "MOUSE_UP", "parameters": {"button": button}})

    @tool_action
    def move_to(self, x: int, y: int) -> str:
        """Move the mouse cursor to pixel coordinates without clicking.

        Parameters
        ----------
        x : int
            Target X coordinate
        y : int
            Target Y coordinate
        """
        return self._execute_desktop_action({"action_type": "MOVE_TO", "parameters": {"x": x, "y": y}})

    @tool_action
    def drag_to(self, x: int, y: int) -> str:
        """Click-and-drag from the current cursor position to (x, y).

        Parameters
        ----------
        x : int
            Target X coordinate
        y : int
            Target Y coordinate
        """
        return self._execute_desktop_action({"action_type": "DRAG_TO", "parameters": {"x": x, "y": y}})

    @tool_action
    def scroll(self, dx: int, dy: int) -> str:
        """Scroll the mouse wheel.

        Parameters
        ----------
        dx : int
            Horizontal scroll amount (positive = right)
        dy : int
            Vertical scroll amount (positive = down)
        """
        return self._execute_desktop_action({"action_type": "SCROLL", "parameters": {"dx": dx, "dy": dy}})

    @tool_action
    def typing(self, text: str) -> str:
        """Type text into the currently focused element.

        Parameters
        ----------
        text : str
            The text to type
        """
        return self._execute_desktop_action({"action_type": "TYPING", "parameters": {"text": text}})

    @tool_action
    def press(self, key: str) -> str:
        """Press and release a single key.

        Parameters
        ----------
        key : str
            Key name (e.g. "enter", "esc", "tab", "backspace", "space")
        """
        return self._execute_desktop_action({"action_type": "PRESS", "parameters": {"key": key}})

    @tool_action
    def key_down(self, key: str) -> str:
        """Press a key down without releasing it.

        Parameters
        ----------
        key : str
            Key name (e.g. "ctrl", "shift", "alt")
        """
        return self._execute_desktop_action({"action_type": "KEY_DOWN", "parameters": {"key": key}})

    @tool_action
    def key_up(self, key: str) -> str:
        """Release a previously held key.

        Parameters
        ----------
        key : str
            Key name (e.g. "ctrl", "shift", "alt")
        """
        return self._execute_desktop_action({"action_type": "KEY_UP", "parameters": {"key": key}})

    @tool_action
    def hotkey(self, keys: str) -> str:
        """Press a key combination simultaneously (e.g. Ctrl+C).

        Parameters
        ----------
        keys : str
            Key names joined by '+' (e.g. "ctrl+c", "ctrl+shift+t")
        """
        if isinstance(keys, str):
            keys = keys.split("+")
        return self._execute_desktop_action({"action_type": "HOTKEY", "parameters": {"keys": keys}})


# ---------------------------------------------------------------------------
# PyAutoGUIComputer — pyautogui code execution action space
# ---------------------------------------------------------------------------


class PyAutoGUIComputer(ComputerBase):
    """
    Desktop/VM computer tool with the pyautogui action space.

    Exposes run_pyautogui() as a @tool_action method, plus the shared
    wait/done/fail terminal signals inherited from ComputerBase.

    The agent writes Python code using pyautogui; SoM tag_N variables
    (center coordinates of numbered bounding boxes) are prepended automatically
    so agents can reference screen elements by index.
    """

    @tool_action
    def run_pyautogui(self, code: str) -> str:
        """Execute Python code using pyautogui in the VM.

        Parameters
        ----------
        code : str
            Python code to execute (e.g. "pyautogui.click(100, 200)"). If SoM
            bounding boxes are available, tag_1, tag_2, ... variables are
            prepended as center coordinates (e.g. "pyautogui.click(*tag_3)").
        """
        tag_vars = ""
        for i, mark in enumerate(self._last_marks):
            x, y, w, h = mark
            tag_vars += f"tag_{i + 1} = ({int(x + w // 2)}, {int(y + h // 2)})\n"

        fixed_code = fix_pyautogui_less_than_bug(tag_vars + code)
        result = self._guest.execute_python_command(fixed_code)
        time.sleep(2)  # replicate desktop_env.step()'s default pause

        if result:
            returncode = result.get("returncode", 0)
            error = result.get("error", "") or result.get("stderr", "")
            if returncode != 0 and error:
                return f"Error executing code:\n{error.strip()}"

        self._action_history.append(code)
        return "Success"
