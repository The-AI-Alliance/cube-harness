"""Computer tool for desktop/VM interactions via desktop_env.

This module provides a general Computer tool that wraps desktop_env.DesktopEnv
for desktop automation tasks. The tool provides raw observations (screenshots,
accessibility trees, terminal output) that can be post-processed by tasks.
"""

import importlib.util
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from PIL import Image

from agentlab2.action_spaces.computer_action_space import (
    ButtonType,
    ComputerActionSpace,
)
from agentlab2.core import Action, Content, Observation
from agentlab2.tool import Tool, ToolConfig

logger = logging.getLogger(__name__)

# Conditional import for desktop_env (following AgentLab pattern)
spec = importlib.util.find_spec("desktop_env")
if spec is not None:
    from desktop_env.desktop_env import DesktopEnv

    # Override desktop_env's default VM download directory to use AgentLab2 data dir
    try:
        from pathlib import Path

        import desktop_env.providers.docker.manager as docker_manager

        AGENTLAB2_VM_DIR = Path.home() / ".agentlab2" / "benchmarks" / "osworld" / "vm_data"
        AGENTLAB2_VM_DIR.mkdir(parents=True, exist_ok=True)
        docker_manager.VMS_DIR = str(AGENTLAB2_VM_DIR)
        logger.info(f"Desktop_env will download VMs to: {AGENTLAB2_VM_DIR}")
    except Exception as e:
        logger.warning(f"Could not override desktop_env VM directory: {e}")
else:
    DesktopEnv = None


class ComputerConfig(ToolConfig):
    """Configuration for Computer tool."""

    provider: str = "docker"  # vmware, virtualbox, docker, aws, azure, gcp, aliyun, volcengine
    region: Optional[str] = None
    path_to_vm: Optional[str] = None
    snapshot_name: str = "init_state"
    action_space: str = "computer_13"
    cache_dir: str = str(Path.home() / ".agentlab2" / "benchmarks" / "osworld" / "cache")
    screen_size: tuple[int, int] = (1920, 1080)
    headless: bool = True
    require_a11y_tree: bool = True
    require_terminal: bool = False
    os_type: str = "Ubuntu"
    enable_proxy: bool = False
    observe_after_action: bool = False  # Get full observation after each action (adds ~1-2s per action)

    def make(self) -> "Computer":
        """Create a Computer tool instance."""
        return Computer(self)


class Computer(Tool, ComputerActionSpace):
    """Computer use tool for desktop/VM interactions via desktop_env.

    This tool wraps desktop_env.DesktopEnv to provide general desktop automation capabilities.
    It returns raw observations (screenshots, accessibility trees) that can be post-processed
    by tasks for benchmark-specific formatting. Implements the ComputerActionSpace protocol
    with 15 actions for mouse, keyboard, and control operations.
    """

    action_space = ComputerActionSpace

    def __init__(self, config: ComputerConfig) -> None:
        """Initialize Computer tool with configuration.

        Args:
            config: Computer configuration

        Raises:
            ImportError: If desktop_env is not installed
        """
        super().__init__()
        self.config = config

        if DesktopEnv is None:
            raise ImportError(
                "desktop_env is not installed. Please install it to use Computer tool.\n"
                "You can install it with: pip install desktop-env"
            )

        self._env = DesktopEnv(
            action_space=config.action_space,
            provider_name=config.provider,
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
        self._current_task_config: Optional[dict] = None

    def execute_action(self, action: Action) -> Observation:
        """Execute action and return observation.

        By default, only returns the action result (e.g., "Success").
        If config.observe_after_action is True, also gets full observation
        (screenshot + accessibility tree) after each action, which adds ~1-2s overhead.

        Args:
            action: Action to execute

        Returns:
            Observation from action execution, optionally with full VM state
        """
        # Handle special "done" and "fail" actions that don't need VM execution
        if action.name in ["done", "fail"]:
            return Observation.from_text(f"Task marked as {action.name}")

        action_obs = super().execute_action(action)

        # Optionally get full observation after each action (expensive for VMs)
        if self.config.observe_after_action:
            action_obs += self.get_observation()

        return action_obs

    def setup_task(self, task_config: dict, seed: Optional[int] = None) -> Observation:
        """Setup VM for a task.

        This resets the VM to the task's snapshot and waits for it to stabilize.
        Typically called by Task.setup() implementations.

        Args:
            task_config: Task configuration dict (structure depends on desktop_env requirements)
            seed: Random seed for task setup

        Returns:
            Initial observation with screenshot and raw accessibility tree
        """
        logger.info(f"Setting up task: {task_config.get('id', 'unknown')}")
        logger.info(f"Instruction: {task_config.get('instruction', '')}")

        self._env.reset(task_config=task_config, seed=seed)

        # CRITICAL: Wait 60 seconds for VM to stabilize
        # This is non-negotiable per OSWorld documentation
        logger.info("Waiting 60s for VM to stabilize...")
        time.sleep(60)

        self._current_task_config = task_config
        return self.get_observation()

    def get_observation(self) -> Observation:
        """Get current observation from VM.

        Returns:
            Observation with screenshot and accessibility tree
        """
        raw_obs = self._env._get_obs()
        return self._convert_observation(raw_obs)

    def _convert_observation(self, raw_obs: dict) -> Observation:
        """Convert desktop_env observation to AgentLab2 format.

        Args:
            raw_obs: Raw observation dict from desktop_env

        Returns:
            AgentLab2 Observation with screenshot and raw accessibility tree.
            Task-specific processing (like axtree linearization) happens in task.obs_postprocess().
        """
        obs = Observation()

        # Screenshot: bytes -> PIL Image
        if "screenshot" in raw_obs:
            screenshot_bytes = raw_obs["screenshot"]
            img = Image.open(BytesIO(screenshot_bytes)).convert("RGB")
            obs.contents.append(Content(data=img, name="screenshot"))

        # Accessibility tree: keep raw XML for task-specific processing
        if "accessibility_tree" in raw_obs and raw_obs["accessibility_tree"]:
            obs.contents.append(
                Content(data=raw_obs["accessibility_tree"], name="accessibility_tree")
            )

        # Terminal output (if available)
        if raw_obs.get("terminal"):
            obs.contents.append(Content(data=raw_obs["terminal"], name="terminal"))

        return obs

    def _execute_desktop_action(self, action_dict: dict) -> str:
        """Execute action in desktop_env and return result.

        Args:
            action_dict: Action dictionary or string for desktop_env

        Returns:
            Success message
        """
        raw_obs, reward, done, info = self._env.step(action_dict)
        return "Success"

    def evaluate_task(self) -> float:
        """Run task evaluation via desktop_env.

        Returns:
            Reward value (1.0 for success, 0.0 for failure)
        """
        try:
            reward = self._env.evaluate()
            logger.info(f"Task evaluation result: {reward}")
            return reward
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0

    # ========================================================================
    # ComputerActionSpace Protocol Implementation
    # ========================================================================

    def move_to(self, x: int, y: int) -> str:
        """Move the cursor to the specified position."""
        action = {"action_type": "MOVE_TO", "parameters": {"x": x, "y": y}}
        return self._execute_desktop_action(action)

    def click(
        self,
        button: ButtonType = "left",
        x: Optional[int] = None,
        y: Optional[int] = None,
        num_clicks: int = 1,
    ) -> str:
        """Click the mouse button."""
        params = {"button": button, "num_clicks": num_clicks}
        if x is not None:
            params["x"] = x
        if y is not None:
            params["y"] = y
        action = {"action_type": "CLICK", "parameters": params}
        return self._execute_desktop_action(action)

    def mouse_down(self, button: ButtonType = "left") -> str:
        """Press the mouse button down."""
        action = {"action_type": "MOUSE_DOWN", "parameters": {"button": button}}
        return self._execute_desktop_action(action)

    def mouse_up(self, button: ButtonType = "left") -> str:
        """Release the mouse button."""
        action = {"action_type": "MOUSE_UP", "parameters": {"button": button}}
        return self._execute_desktop_action(action)

    def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> str:
        """Right click the mouse."""
        return self.click(button="right", x=x, y=y)

    def double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> str:
        """Double click the mouse."""
        return self.click(x=x, y=y, num_clicks=2)

    def drag_to(self, x: int, y: int) -> str:
        """Drag the cursor to the specified position with the left button pressed."""
        action = {"action_type": "DRAG_TO", "parameters": {"x": x, "y": y}}
        return self._execute_desktop_action(action)

    def scroll(self, dx: int, dy: int) -> str:
        """Scroll the mouse wheel."""
        action = {"action_type": "SCROLL", "parameters": {"dx": dx, "dy": dy}}
        return self._execute_desktop_action(action)

    def typing(self, text: str) -> str:
        """Type the specified text."""
        action = {"action_type": "TYPING", "parameters": {"text": text}}
        return self._execute_desktop_action(action)

    def press(self, key: str) -> str:
        """Press the specified key and release it."""
        action = {"action_type": "PRESS", "parameters": {"key": key}}
        return self._execute_desktop_action(action)

    def key_down(self, key: str) -> str:
        """Press the specified key down (without releasing)."""
        action = {"action_type": "KEY_DOWN", "parameters": {"key": key}}
        return self._execute_desktop_action(action)

    def key_up(self, key: str) -> str:
        """Release the specified key."""
        action = {"action_type": "KEY_UP", "parameters": {"key": key}}
        return self._execute_desktop_action(action)

    def hotkey(self, keys: List[str]) -> str:
        """Press the specified key combination."""
        # Handle if keys is passed as a string like "ctrl+alt+t"
        if isinstance(keys, str):
            keys = keys.split("+")
        action = {"action_type": "HOTKEY", "parameters": {"keys": keys}}
        return self._execute_desktop_action(action)

    def wait(self) -> str:
        """Wait until the next action (no-op)."""
        return self._execute_desktop_action("WAIT")

    def fail(self) -> str:
        """Signal that the task cannot be performed."""
        return "Task marked as failed"

    def done(self) -> str:
        """Signal that the task is complete."""
        return "Task marked as done"

    def close(self) -> None:
        """Clean up VM resources."""
        if self._env:
            logger.info("Closing desktop environment")
            self._env.close()
