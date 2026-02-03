import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from browsergym.core.env import BrowserEnv
from browsergym.core.task import OpenEndedTask
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from PIL import Image
from termcolor import colored

from agentlab2.action_spaces.browser_action_space import BidBrowserActionSpace
from agentlab2.core import Action, Content, Observation
from agentlab2.tool import Tool, ToolConfig

logger = logging.getLogger(__name__)


class BrowsergymConfig(ToolConfig):
    """Configuration for BrowserGym tool."""

    # Task configuration
    task_entrypoint: Callable[[], Any] | None = None
    task_kwargs: dict = {}

    # Browser configuration
    headless: bool = True
    viewport: dict | None = None
    slow_mo: int | None = None
    timeout: int | None = None
    locale: str | None = None
    timezone_id: str | None = None

    # Playwright customization
    pw_chromium_kwargs: dict = {}
    pw_context_kwargs: dict = {}

    # Recording
    record_video_dir: str | None = None

    # Behavioral options
    tags_to_mark: str = "standard_html"  # "all" or "standard_html"
    wait_for_user_message: bool = False
    terminate_on_infeasible: bool = True
    resizeable_window: bool = False
    action_mapping: Callable | None = None
    use_raw_page_output: bool = False
    pre_observation_delay: float = 2.5

    # Observation configuration
    use_html: bool = True
    use_axtree: bool = True
    use_screenshot: bool = True
    prune_html: bool = True
    max_wait: int = 60

    def make(self) -> "BrowsergymTool":
        return BrowsergymTool(self)


class BrowsergymTool(Tool, BidBrowserActionSpace):
    """BrowserGym tool wrapper that adapts BrowserGym's BrowserEnv to the AgentLab2 Tool interface.

    This tool wraps the BrowserGym environment and provides:
    - Action execution via BidBrowserActionSpace protocol (mapped to BrowserGym actions)
    - Observation extraction (HTML, accessibility tree, screenshots)
    - Proper lifecycle management (reset, close)

    The action space is defined by the BidBrowserActionSpace protocol, using BID-based element selection.
    Actions are executed via BrowserGym's env.step() method.
    """

    action_space = BidBrowserActionSpace

    def __init__(self, config: BrowsergymConfig) -> None:
        super().__init__()
        self.config = config
        self._env: BrowserEnv | None = None
        self._last_obs: dict | None = None
        self._last_info: dict | None = None
        self._last_reward: float = 0.0
        self._last_terminated: bool = False

    def _create_env(self) -> BrowserEnv:
        """Create a new BrowserGym environment instance."""
        task_entrypoint = self.config.task_entrypoint if self.config.task_entrypoint is not None else OpenEndedTask
        task_kwargs = (
            self.config.task_kwargs
            if self.config.task_kwargs is not None
            else {"start_url": "about:blank", "goal": None}
        )

        env_kwargs = {
            "task_entrypoint": task_entrypoint,
            "task_kwargs": task_kwargs,
            "headless": self.config.headless,
            "tags_to_mark": self.config.tags_to_mark,
            "wait_for_user_message": self.config.wait_for_user_message,
            "terminate_on_infeasible": self.config.terminate_on_infeasible,
            "resizeable_window": self.config.resizeable_window,
            "use_raw_page_output": self.config.use_raw_page_output,
            "pre_observation_delay": self.config.pre_observation_delay,
        }
        if self.config.viewport is not None:
            env_kwargs["viewport"] = self.config.viewport
        if self.config.slow_mo is not None:
            env_kwargs["slow_mo"] = self.config.slow_mo
        if self.config.timeout is not None:
            env_kwargs["timeout"] = self.config.timeout
        if self.config.locale is not None:
            env_kwargs["locale"] = self.config.locale
        if self.config.timezone_id is not None:
            env_kwargs["timezone_id"] = self.config.timezone_id
        if self.config.pw_chromium_kwargs:
            env_kwargs["pw_chromium_kwargs"] = self.config.pw_chromium_kwargs
        if self.config.pw_context_kwargs:
            env_kwargs["pw_context_kwargs"] = self.config.pw_context_kwargs
        if self.config.record_video_dir is not None:
            env_kwargs["record_video_dir"] = self.config.record_video_dir
        if self.config.action_mapping is not None:
            env_kwargs["action_mapping"] = self.config.action_mapping

        return BrowserEnv(**env_kwargs)

    def _ensure_env(self) -> None:
        """Ensure the environment is created and reset."""
        if self._env is None:
            raise RuntimeError("BrowserGym environment is not initialized. Call reset() first.")

    @property
    def env(self) -> BrowserEnv:
        """Access the underlying BrowserGym environment."""
        self._ensure_env()
        return self._env  # type: ignore

    @property
    def page(self) -> Any:
        """Access the current Playwright page from BrowserGym."""
        self._ensure_env()
        return self._env.page if self._env and hasattr(self._env, "page") else None

    @property
    def last_reward(self) -> float:
        """Get the reward from the last step."""
        return self._last_reward

    @property
    def last_terminated(self) -> bool:
        """Check if the episode terminated on the last step."""
        return self._last_terminated

    def reset(self) -> None:
        """Reset the environment."""
        if self._env is not None:
            self._env.close()
        self._env = self._create_env()
        self._last_obs, self._last_info = self._env.reset()

    def set_gym_task(
        self,
        task_entrypoint: Callable[[], Any],
        task_kwargs: dict | None = None,
        seed: int | None = None,
    ) -> None:
        """Reinitialize the BrowserGym environment with a specific task.

        This method allows tasks (like WorkArenaTask) to configure the BrowserGym
        task after the tool has been created. It closes the existing environment
        and creates a new one with the specified task entrypoint and kwargs.

        Args:
            task_entrypoint: The BrowserGym task class or factory function.
            task_kwargs: Optional kwargs to pass to the task constructor (excluding seed).
            seed: Random seed for the task (passed to env.reset(), not task_kwargs).
        """
        if self._env is not None:
            self._env.close()
            self._env = None

        # Update config with task-specific settings
        self.config = self.config.model_copy(
            update={
                "task_entrypoint": task_entrypoint,
                "task_kwargs": task_kwargs or {},
            }
        )

        # Create and reset the new environment
        self._env = self._create_env()
        self._last_obs, self._last_info = self._env.reset(seed=seed)

    def execute_action(self, action: Action) -> Observation:
        """Execute an action using BidBrowserActionSpace protocol and return the observation."""
        action_obs = super().execute_action(action)
        action_obs += self.page_obs()
        return action_obs

    def _execute_bgym_step(self, action_str: str) -> str:
        """Execute a BrowserGym action string via env.step() and return result message."""
        self._ensure_env()
        logger.info(colored(f"Execute bgym step: {action_str}", "magenta"))
        obs, reward, terminated, truncated, info = self._env.step(action_str)  # type: ignore
        self._last_obs = obs
        self._last_info = info
        self._last_reward = reward
        self._last_terminated = terminated

        result = "Success"
        if terminated:
            result += " (Episode terminated)"
        if truncated:
            result += " (Episode truncated)"
        return result

    # === BidBrowserActionSpace protocol implementation ===
    # Each method maps to a BrowserGym action and executes via env.step()

    def browser_press_key(self, key: str) -> str:
        """Press a key on the keyboard."""
        action_str = f'keyboard_press("{key}")'
        return self._execute_bgym_step(action_str)

    def browser_type(self, bid: str, text: str) -> str:
        """Type text into the element specified by BID."""
        # Escape quotes in the text
        escaped_text = text.replace("\\", "\\\\").replace('"', '\\"')
        action_str = f'fill(bid="{bid}", value="{escaped_text}")'
        return self._execute_bgym_step(action_str)

    def browser_click(self, bid: str) -> str:
        """Click on an element specified by BID."""
        # Get state before click (for checkbox/radio detection)
        state_before = self._get_checkbox_state(bid)

        # Execute standard click
        action_str = f'click(bid="{bid}")'
        result = self._execute_bgym_step(action_str)

        # For checkboxes/radios, verify state changed and use JS fallback if needed
        if state_before is not None:
            state_after = self._get_checkbox_state(bid)
            if state_after == state_before:
                # Click didn't toggle - use JS fallback
                self._toggle_checkbox_js(bid, not state_before)
                state_after_js = self._get_checkbox_state(bid)
                logger.info(colored(f"Checkbox/radio {bid} clicked with JS fallback, state: {state_after_js}", "cyan"))
                result = self._execute_bgym_step("noop()")  # Dummy step to update obs/info

        return result

    def _get_frame_for_bid(self, bid: str) -> Any:
        """Navigate to the correct frame for a BID using BrowserGym's naming convention.

        BIDs like 'a195' encode iframe hierarchy:
        - 'a' is the first iframe
        - 'aA' would be a nested iframe inside 'a'
        - '195' without prefix is in the main frame

        Returns the frame/page where the element with this BID lives.
        """
        current_frame = self.page

        # Parse the BID to find frame prefixes
        i = 0
        while i < len(bid) and not bid[i:].isnumeric():
            i += 1
            # Allow multi-character frame ids like aA, bCD etc.
            while i < len(bid) and bid[i].isalpha() and bid[i].isupper():
                i += 1

            if i > 0:
                frame_bid = bid[:i]  # bid of the next frame to select
                try:
                    frame_elem = current_frame.get_by_test_id(frame_bid)
                    if frame_elem.count() > 0:
                        current_frame = frame_elem.frame_locator(":scope")
                    else:
                        break
                except Exception:
                    break

        return current_frame

    def _get_checkbox_state(self, bid: str) -> bool | None:
        """Get checkbox/radio checked state, or None if not a checkbox/radio.

        Navigates to the correct iframe using BrowserGym's BID naming convention,
        then checks the element's checkbox state.
        """
        try:
            # Navigate to the correct frame for this BID
            frame = self._get_frame_for_bid(bid)
            locator = frame.get_by_test_id(bid)

            if locator.count() == 0:
                return None

            # Get the element's properties via evaluate
            js_code = """
                (elem) => {
                    if (elem.type === 'checkbox' || elem.type === 'radio') {
                        return { isCheckbox: true, checked: elem.checked };
                    }
                    if (elem.getAttribute('data-type') === 'checkbox') {
                        return { isCheckbox: true, checked: elem.value === 'true' };
                    }
                    return { isCheckbox: false };
                }
            """
            result = locator.evaluate(js_code)

            if isinstance(result, dict) and result.get("isCheckbox"):
                return result.get("checked")
            return None
        except Exception:
            return None

    def _toggle_checkbox_js(self, bid: str, checked: bool) -> None:
        """Toggle checkbox state using JavaScript.

        Navigates to the correct iframe using BrowserGym's BID naming convention,
        then toggles the checkbox state.
        """
        try:
            frame = self._get_frame_for_bid(bid)
            locator = frame.get_by_test_id(bid)

            js_code = """
            (elem, checked) => {
                if (elem.type === 'checkbox' || elem.type === 'radio') {
                    elem.checked = checked;
                    elem.dispatchEvent(new Event('click', { bubbles: true }));
                    elem.dispatchEvent(new Event('change', { bubbles: true }));
                    elem.dispatchEvent(new Event('input', { bubbles: true }));
                    return true;
                }
                if (elem.getAttribute('data-type') === 'checkbox') {
                    elem.value = checked ? 'true' : 'false';
                    elem.dispatchEvent(new Event('change', { bubbles: true }));
                    elem.dispatchEvent(new Event('input', { bubbles: true }));
                    return true;
                }
                return false;
            }
            """
            locator.evaluate(js_code, checked)
        except Exception:
            pass

    def browser_drag(self, from_bid: str, to_bid: str) -> str:
        """Drag and drop from one element to another."""
        action_str = f'drag_and_drop(from_bid="{from_bid}", to_bid="{to_bid}")'
        return self._execute_bgym_step(action_str)

    def browser_hover(self, bid: str) -> str:
        """Hover over an element specified by BID."""
        action_str = f'hover(bid="{bid}")'
        return self._execute_bgym_step(action_str)

    def browser_select_option(self, bid: str, value: str) -> str:
        """Select an option from a dropdown element."""
        escaped_value = value.replace("\\", "\\\\").replace('"', '\\"')
        action_str = f'select_option(bid="{bid}", options="{escaped_value}")'
        return self._execute_bgym_step(action_str)

    def browser_mouse_click_xy(self, x: int, y: int) -> str:
        """Click at a given x, y coordinate using the mouse."""
        action_str = f"mouse_click(x={x}, y={y})"
        return self._execute_bgym_step(action_str)

    def browser_wait(self, seconds: int) -> str:
        """Wait for a given number of seconds, up to max_wait."""
        wait_seconds = min(seconds, self.config.max_wait)
        wait_ms = wait_seconds * 1000
        action_str = f"noop(wait_ms={wait_ms})"
        return self._execute_bgym_step(action_str)

    def browser_back(self) -> str:
        """Navigate back in browser history."""
        action_str = "go_back()"
        return self._execute_bgym_step(action_str)

    def browser_forward(self) -> str:
        """Navigate forward in browser history."""
        action_str = "go_forward()"
        return self._execute_bgym_step(action_str)

    def noop(self) -> str:
        """No operation action."""
        action_str = "noop()"
        return self._execute_bgym_step(action_str)

    def _bgym_obs_to_agentlab_obs(self, bgym_obs: dict) -> Observation:
        """Convert BrowserGym observation dict to AgentLab2 Observation.

        BrowserGym provides observations with keys like:
        - 'screenshot': numpy array (converted to PIL Image)
        - 'dom_object': dict with 'documents' and 'strings' (converted to HTML string)
        - 'axtree_object': dict (converted to accessibility tree string)
        """
        obs = Observation()

        # Add HTML if configured (flatten BrowserGym's DOM to HTML string)
        if self.config.use_html and "dom_object" in bgym_obs:
            dom_obj = bgym_obs["dom_object"]
            # Use flatten_dom_to_str to convert BrowserGym's compact DOM format to HTML
            html_str = flatten_dom_to_str(
                dom_obj,
                extra_properties=bgym_obs.get("extra_element_properties", {}),
                with_visible=False,
                filter_visible_only=False,
            )
            if self.config.prune_html:
                html_str = prune_html(html_str)
            # BrowserGym's flatten_dom_to_str already provides a pruned representation
            obs.contents.append(Content(data=html_str, name="pruned_html"))

        # Add accessibility tree if configured
        if self.config.use_axtree and "axtree_object" in bgym_obs:
            axtree_obj = bgym_obs["axtree_object"]
            if axtree_obj:
                axtree_str = flatten_axtree_to_str(axtree_obj)
                obs.contents.append(Content(data=axtree_str, name="axtree_txt"))

        # Add screenshot if configured (convert numpy array to PIL Image)
        if self.config.use_screenshot and "screenshot" in bgym_obs:
            screenshot = bgym_obs["screenshot"]
            if isinstance(screenshot, Image.Image):
                obs.contents.append(Content(data=screenshot, name="screenshot"))
            elif isinstance(screenshot, np.ndarray):
                screenshot_img = Image.fromarray(screenshot)
                obs.contents.append(Content(data=screenshot_img, name="screenshot"))

        return obs

    # === Utility methods for BrowserTaskTool (goto, evaluate_js, page_obs) ===

    def goto(self, url: str) -> None:
        """Navigate to a URL and wait for the page to be fully loaded."""
        self._ensure_env()
        self.page.goto(url)

    def evaluate_js(self, js: str) -> Any:
        """Evaluate JavaScript in the browser context and return the result."""
        self._ensure_env()
        if self.page is not None:
            return self.page.evaluate(js)
        return None

    def page_obs(self) -> Observation:
        """Get the current page observation from BrowserGym.

        This method acts as an adapter, converting BrowserGym's observation format
        to AgentLab2's Observation format.
        """
        self._ensure_env()

        # Get the latest observation from BrowserGym
        # If we have a cached observation from the last step, use it
        # Otherwise, the observation should be available from reset()
        if self._last_obs is None:
            raise RuntimeError("No BrowserGym observation available. Did you call reset()?")

        return self._bgym_obs_to_agentlab_obs(self._last_obs)

    def close(self) -> None:
        """Clean up BrowserGym environment resources."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                logger.warning(f"Error closing BrowserGym environment: {e}")
            finally:
                self._env = None
                self._last_obs = None
                self._last_info = None
                self._last_reward = 0.0
                self._last_terminated = False
