"""
Computer tool — CUBE tool wrapping desktop_env for VM-based desktop automation.

Two variants selected by ComputerConfig.action_space:
    Computer13        — 13 mouse/keyboard primitives + wait/done/fail
    PyAutoGUIComputer — run_pyautogui() code execution + wait/done/fail
"""

import logging
import time
from enum import Enum
from io import BytesIO
from typing import Literal
from pathlib import Path

from PIL import Image

import cube
from cube.container import Container
from cube.core import Action, Content, ImageContent, Observation, StepError, TextContent
from cube.tool import Tool, ToolConfig, tool_action
from desktop_env.desktop_env import DesktopEnv
import desktop_env.providers.docker.manager as docker_manager

logger = logging.getLogger(__name__)

_CUBE_CACHE_ROOT = cube.get_cache_dir("osworld-cube")


# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------


class VMProvider(str, Enum):
    """Supported desktop_env VM providers."""

    DOCKER = "docker"
    VMWARE = "vmware"
    VIRTUALBOX = "virtualbox"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ALIYUN = "aliyun"
    VOLCENGINE = "volcengine"


class ActionSpace(str, Enum):
    """desktop_env action space variants.

    Values match the desktop_env action_space string passed to DesktopEnv().
    """

    COMPUTER_13 = "computer_13"
    PYAUTOGUI = "pyautogui"


# ---------------------------------------------------------------------------
# Config classes
# ---------------------------------------------------------------------------


class ComputerConfig(ToolConfig):
    """
    Serializable configuration for Computer tool variants.

    Maps to desktop_env.DesktopEnv constructor arguments.

    cache_dir and vm_dir default to $CUBE_CACHE_DIR/benchmarks/osworld/{cache,vm_data}.
    Override CUBE_CACHE_DIR env var to change the root data directory.

    action_space selects the tool variant:
      "computer_13" → Computer13 (13 mouse/keyboard primitives + wait/done/fail)
      "pyautogui"   → PyAutoGUIComputer (run_pyautogui + wait/done/fail)
    """

    action_space: ActionSpace = ActionSpace.COMPUTER_13
    provider: VMProvider = VMProvider.DOCKER
    region: str | None = None
    path_to_vm: str | None = None
    snapshot_name: str = "init_state"
    cache_dir: str = str(_CUBE_CACHE_ROOT / "cache")
    vm_dir: str = str(_CUBE_CACHE_ROOT /"vm_data")
    screen_size: tuple[int, int] = (1920, 1080)
    headless: bool = True
    require_a11y_tree: bool = True
    require_terminal: bool = False
    os_type: Literal["Ubuntu", "Windows"] = "Ubuntu"
    enable_proxy: bool = False
    observe_after_action: bool = True

    def make(self, container: Container | None = None) -> "ComputerBase":
        if container is not None:
            logger.warning(
                "ComputerConfig.make() received a cube Container, but the OSWorld "
                "Computer tool manages its own VM via desktop_env. The container "
                "argument will be ignored."
            )
        if self.action_space == ActionSpace.PYAUTOGUI:
            return PyAutoGUIComputer(self)
        return Computer13(self)


# ---------------------------------------------------------------------------
# ComputerBase — shared VM lifecycle and task helpers
# ---------------------------------------------------------------------------


class ComputerBase(Tool):
    """
    Shared base for Computer13 and PyAutoGUIComputer.

    Provides VM lifecycle (init, setup_task, evaluate_task, get_observation, close)
    and the three terminal @tool_action signals shared by both action spaces:
    wait, done, fail.

    Subclasses add the action-space-specific @tool_action methods and are
    paired with their own Config class.

    """

    def __init__(self, config: ComputerConfig) -> None:
        """
        Create Computer and initialise desktop_env.DesktopEnv.

        DesktopEnv is created eagerly here (inside Task.model_post_init).
        The VM boots at this point; the snapshot restore happens later in setup_task().
        Marked for revisiting: consider lazy init if boot time is a problem.
        """
        self.config = config
        self._current_task_config: dict | None = None
        self._last_marks: list[list[int]] = []
        self._is_done: bool = False

        if docker_manager is not None:
            vm_dir = Path(config.vm_dir)
            vm_dir.mkdir(parents=True, exist_ok=True)
            docker_manager.VMS_DIR = str(vm_dir)
            logger.info(f"desktop_env will download VMs to: {vm_dir}")

        self._env = DesktopEnv(
            action_space=self._desktop_env_action_space,
            provider_name=config.provider.value,
            region=config.region,
            path_to_vm=config.path_to_vm,
            snapshot_name=config.snapshot_name,
            cache_dir=config.cache_dir,
            screen_size=config.screen_size,
            headless=config.headless,
            require_a11y_tree=config.require_a11y_tree,
            require_terminal=config.require_terminal,
            os_type=config.os_type,
        )

    _desktop_env_action_space: ActionSpace = ActionSpace.COMPUTER_13

    def execute_action(self, action: Action) -> Observation | StepError:
        """Execute action; append full VM observation if observe_after_action=True."""
        action_obs = super().execute_action(action)

        if self.config.observe_after_action and action.name not in ("done", "fail"):
            action_obs += self.get_observation()

        return action_obs

    def setup_task(self, task_config: dict, seed: int | None = None) -> Observation:
        """
        Restore VM snapshot, run setup scripts, wait for stabilisation,
        then return the initial observation.

        Called by OSWorldTask.reset().

        Args:
            task_config: dict with keys id, instruction, config, evaluator,
                         snapshot, related_apps — matches desktop_env format.
            seed: Optional random seed passed through to desktop_env.
        """
        logger.info(f"Setting up task: {task_config.get('id', 'unknown')}")
        logger.info(f"Instruction: {task_config.get('instruction', '')}")

        self._env.reset(task_config=task_config, seed=seed)

        logger.info("Waiting 60s for VM to stabilise...")
        time.sleep(60)

        self._is_done = False
        self._current_task_config = task_config
        return self.get_observation()

    def evaluate_task(self) -> float:
        """
        Run desktop_env's built-in evaluator and return reward ∈ [0.0, 1.0].

        Called by OSWorldTask.evaluate(). Partial credit is preserved.
        """
        try:
            reward = self._env.evaluate()
            logger.info(f"Task evaluation result: {reward}")
            return reward
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0

    def get_observation(self) -> Observation:
        """Read current screen state from desktop_env and return as Observation."""
        raw_obs = self._env._get_obs()
        return self._convert_observation(raw_obs)

    def _convert_observation(self, raw_obs: dict) -> Observation:
        """
        Convert desktop_env observation dict to a cube Observation.

        desktop_env keys:
            "screenshot"         → bytes (PNG) → PIL.Image stored as ImageContent
            "accessibility_tree" → XML string  → TextContent (named "accessibility_tree")
            "terminal"           → str         → TextContent (named "terminal")

        Screenshot bytes are decoded to PIL.Image so that ImageContent can
        serialise them as base64 and pass them to multimodal LLMs.
        """
        contents: list[Content] = []

        if "screenshot" in raw_obs:
            img = Image.open(BytesIO(raw_obs["screenshot"])).convert("RGB")
            contents.append(ImageContent(data=img, name="screenshot"))

        if raw_obs.get("accessibility_tree"):
            contents.append(
                TextContent(data=raw_obs["accessibility_tree"], name="accessibility_tree")
            )

        if raw_obs.get("terminal"):
            contents.append(TextContent(data=raw_obs["terminal"], name="terminal"))

        return Observation(contents=contents)

    def _execute_desktop_action(self, action_dict: dict | str) -> str:
        """Send an action dict to desktop_env and return a success string."""
        self._env.step(action_dict)
        return "Success"

    def update_marks(self, marks: list[list[int]]) -> None:
        """
        Store SoM bounding-box marks produced by tag_screenshot().
        Used by run_pyautogui() to resolve tag_N variables.
        """
        self._last_marks = marks

    def reset(self) -> None:
        """Reset tool state between tasks (cube AbstractTool.reset() override)."""
        self._last_marks = []
        self._is_done = False

    def close(self) -> None:
        """Shut down the VM and release resources."""
        if self._env:
            logger.info("Closing desktop environment")
            self._env.close()

    @tool_action
    def wait(self) -> str:
        """Wait one step without taking any action."""
        return self._execute_desktop_action("WAIT")

    @tool_action
    def done(self) -> str:
        """Signal that the task has been completed successfuly."""
        self._is_done = True
        return "Task marked as done"

    @tool_action
    def fail(self) -> str:
        """Signal that the task cannot be completed (infeasible or failed)."""
        self._is_done = True
        return "Task marked as failed"


# ---------------------------------------------------------------------------
# Computer13 — 13 mouse/keyboard primitives
# ---------------------------------------------------------------------------


class Computer13(ComputerBase):
    """
    Desktop/VM computer tool with the computer_13 action space.

    Exposes 13 mouse/keyboard primitives as @tool_action methods, plus the
    shared wait/done/fail terminal signals inherited from ComputerBase.

    Action methods are decorated with @tool_action so that cube.tool.Tool
    auto-discovers them for action_set and execute_action dispatch.
    """

    _desktop_env_action_space = ActionSpace.COMPUTER_13

    @tool_action
    def click(
        self,
        button: str = "left",
        x: int = -1,
        y: int = -1,
        num_clicks: int = 1,
    ) -> str:
        """Click the mouse button at optional coordinates.

        Args:
            button: Mouse button — "left", "right", or "middle"
            x: X coordinate to click at (-1 = use current cursor position)
            y: Y coordinate to click at (-1 = use current cursor position)
            num_clicks: Number of clicks (1 for single, 2 for double, etc.)
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

        Args:
            x: X coordinate (-1 = use current cursor position)
            y: Y coordinate (-1 = use current cursor position)
        """
        return self.click(x=x, y=y, num_clicks=2)

    @tool_action
    def right_click(self, x: int = -1, y: int = -1) -> str:
        """Right-click the mouse at optional coordinates.

        Args:
            x: X coordinate (-1 = use current cursor position)
            y: Y coordinate (-1 = use current cursor position)
        """
        return self.click(button="right", x=x, y=y)

    @tool_action
    def mouse_down(self, button: str = "left") -> str:
        """Press and hold a mouse button.

        Args:
            button: Mouse button — "left", "right", or "middle"
        """
        return self._execute_desktop_action(
            {"action_type": "MOUSE_DOWN", "parameters": {"button": button}}
        )

    @tool_action
    def mouse_up(self, button: str = "left") -> str:
        """Release a held mouse button.

        Args:
            button: Mouse button — "left", "right", or "middle"
        """
        return self._execute_desktop_action(
            {"action_type": "MOUSE_UP", "parameters": {"button": button}}
        )

    @tool_action
    def move_to(self, x: int, y: int) -> str:
        """Move the mouse cursor to pixel coordinates without clicking.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
        """
        return self._execute_desktop_action(
            {"action_type": "MOVE_TO", "parameters": {"x": x, "y": y}}
        )

    @tool_action
    def drag_to(self, x: int, y: int) -> str:
        """Click-and-drag from the current cursor position to (x, y).

        Args:
            x: Target X coordinate
            y: Target Y coordinate
        """
        return self._execute_desktop_action(
            {"action_type": "DRAG_TO", "parameters": {"x": x, "y": y}}
        )

    @tool_action
    def scroll(self, dx: int, dy: int) -> str:
        """Scroll the mouse wheel.

        Args:
            dx: Horizontal scroll amount (positive = right)
            dy: Vertical scroll amount (positive = down)
        """
        return self._execute_desktop_action(
            {"action_type": "SCROLL", "parameters": {"dx": dx, "dy": dy}}
        )

    @tool_action
    def typing(self, text: str) -> str:
        """Type text into the currently focused element.

        Args:
            text: The text to type
        """
        return self._execute_desktop_action(
            {"action_type": "TYPING", "parameters": {"text": text}}
        )

    @tool_action
    def press(self, key: str) -> str:
        """Press and release a single key.

        Args:
            key: Key name (e.g. "enter", "esc", "tab", "backspace", "space")
        """
        return self._execute_desktop_action(
            {"action_type": "PRESS", "parameters": {"key": key}}
        )

    @tool_action
    def key_down(self, key: str) -> str:
        """Press a key down without releasing it.

        Args:
            key: Key name (e.g. "ctrl", "shift", "alt")
        """
        return self._execute_desktop_action(
            {"action_type": "KEY_DOWN", "parameters": {"key": key}}
        )

    @tool_action
    def key_up(self, key: str) -> str:
        """Release a previously held key.

        Args:
            key: Key name (e.g. "ctrl", "shift", "alt")
        """
        return self._execute_desktop_action(
            {"action_type": "KEY_UP", "parameters": {"key": key}}
        )

    @tool_action
    def hotkey(self, keys: str) -> str:
        """Press a key combination simultaneously (e.g. Ctrl+C).

        Args:
            keys: Key names joined by '+' (e.g. "ctrl+c", "ctrl+shift+t")
        """
        if isinstance(keys, str):
            keys = keys.split("+")
        return self._execute_desktop_action(
            {"action_type": "HOTKEY", "parameters": {"keys": keys}}
        )


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

    _desktop_env_action_space = ActionSpace.PYAUTOGUI

    @tool_action
    def run_pyautogui(self, code: str) -> str:
        """Execute Python code using pyautogui in the VM.

        SoM tag_N variables (center coordinates of numbered bounding boxes) are
        automatically prepended to the code so agents can reference them by index.

        Args:
            code: Python code using pyautogui and optional tag_N variables
                  (e.g. "pyautogui.click(*tag_3)")
        """
        from desktop_env.desktop_env import _fix_pyautogui_less_than_bug

        tag_vars = ""
        for i, mark in enumerate(self._last_marks):
            x, y, w, h = mark
            tag_vars += f"tag_{i + 1} = ({int(x + w // 2)}, {int(y + h // 2)})\n"

        fixed_code = _fix_pyautogui_less_than_bug(tag_vars + code)
        result = self._env.controller.execute_python_command(fixed_code)
        time.sleep(2)  # replicate desktop_env.step()'s default pause

        if result:
            returncode = result.get("returncode", 0)
            error = result.get("error", "") or result.get("stderr", "")
            if returncode != 0 and error:
                return f"Error executing code:\n{error.strip()}"

        return "Success"