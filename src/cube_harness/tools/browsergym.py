import logging
import time
from typing import Any, Callable

import numpy as np
from browsergym.core.action.base import execute_python_code
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.constants import BROWSERGYM_ID_ATTRIBUTE, EXTRACT_OBS_MAX_TRIES
from browsergym.core.observation import (
    MarkingError,
    _post_extract,
    _pre_extract,
    extract_dom_extra_properties,
    extract_dom_snapshot,
    extract_focused_element_bid,
    extract_merged_axtree,
    extract_screenshot,
)
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from cube.core import Action, Content, Observation, StepError
from cube.tool import BrowserTool, ToolConfig
from cube_browser_playwright.playwright_session import PlaywrightSession, PlaywrightSessionConfig
from PIL import Image
from playwright.sync_api import Error, Frame, Page
from pydantic import Field
from termcolor import colored

from cube_harness.action_spaces.browser_action_space import BidBrowserActionSpace
from cube_harness.tool import ToolWithTelemetry

logger = logging.getLogger(__name__)


class BrowsergymConfig(ToolConfig):
    """Configuration for BrowserGym-style Playwright tool."""

    # Browser configuration (launch parameters)
    browser_config: PlaywrightSessionConfig = Field(default_factory=PlaywrightSessionConfig)

    # Observation behavior
    tags_to_mark: str = "standard_html"  # "all" or "standard_html"
    wait_for_user_message: bool = False
    terminate_on_infeasible: bool = True
    action_mapping: Callable | None = None
    use_raw_page_output: bool = False
    pre_observation_delay: float = 2.5

    # Observation configuration
    use_html: bool = True
    use_axtree: bool = True
    use_screenshot: bool = True
    prune_html: bool = True

    # Action behavior
    max_wait: int = 60

    def make(self, container=None) -> "BrowsergymTool":
        return BrowsergymTool(self)


class BrowsergymTool(ToolWithTelemetry, BrowserTool, BidBrowserActionSpace):
    """BrowserGym tool wrapper that adapts BrowserGym's observation utilities to the cube-harness Tool interface.

    This tool manages the browser lifecycle directly (without BrowserEnv) and provides:
    - Action execution via BidBrowserActionSpace protocol (mapped to BrowserGym action strings)
    - Observation extraction (HTML, accessibility tree, screenshots) via BrowserGym utilities
    - Proper lifecycle management (reset, close) via PlaywrightSession

    The action space is defined by the BidBrowserActionSpace protocol, using BID-based element selection.
    Actions are executed via BrowserGym's execute_python_code() function.
    """

    def __init__(self, config: BrowsergymConfig) -> None:
        super().__init__()
        self.config = config
        self._action_set = HighLevelActionSet()
        self._session: PlaywrightSession | None = None
        self._last_obs: dict | None = None
        self._last_info: dict | None = None
        self._last_reward: float = 0.0
        self._last_terminated: bool = False

    @property
    def session(self) -> PlaywrightSession:
        if self._session is None:
            raise RuntimeError("Browser is not initialized. Call reset() first.")
        return self._session

    @property
    def page(self) -> Page:
        return self.session.page

    @property
    def last_reward(self) -> float:
        """Get the reward from the last step."""
        return self._last_reward

    @property
    def last_terminated(self) -> bool:
        """Check if the episode terminated on the last step."""
        return self._last_terminated

    def _create_runtime(self) -> None:
        self._session = self.config.browser_config.make()
        self._session.playwright.selectors.set_test_id_attribute(BROWSERGYM_ID_ATTRIBUTE)

    def _close_runtime(self) -> None:
        if self._session is not None:
            self.session.stop()
            self._session = None

    def _wait_dom_loaded(self) -> None:
        if self.session is None:
            return
        for page in self.session.context.pages:
            try:
                page.wait_for_load_state("domcontentloaded", timeout=3000)
            except Error:
                pass
            for frame in page.frames:
                try:
                    frame.wait_for_load_state("domcontentloaded", timeout=3000)
                except Error:
                    pass

    def reset(self) -> None:
        self._close_runtime()
        self._create_runtime()
        self._wait_dom_loaded()
        self._last_obs = self._extract_bgym_obs()
        self._last_info = {"source": "reset"}
        self._last_reward = 0.0
        self._last_terminated = False

    def _execute_action(self, action: Action) -> Observation | StepError:
        """Execute an action and return the observation, or a StepError if the action failed."""
        result = super()._execute_action(action)
        if isinstance(result, StepError):
            return result
        result += self.page_obs()
        return result

    def _execute_bgym_step(self, action_str: str) -> str:
        """Execute a BrowserGym action string and return result message."""
        logger.info(f"Execute bgym step: {action_str}")
        result = "Success"

        try:
            code = self._action_set.to_python_code(action_str)
            execute_python_code(
                code=code,
                page=self.page,
                send_message_to_user=lambda message: logger.info(f"BrowserGym message: {message}"),
                report_infeasible_instructions=lambda message: logger.warning(f"Infeasible instruction: {message}"),
            )
            self._last_info = {"source": "action", "action": action_str, "action_error": ""}
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            self._last_info = {"source": "action", "action": action_str, "action_error": error_msg}
            result = f"Failed: {error_msg}"

        self._last_obs = self._extract_bgym_obs()
        self._last_reward = 0.0
        self._last_terminated = False
        return result

    # === BidBrowserActionSpace protocol implementation ===

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

    def _get_frame_for_bid(self, bid: str) -> Page | Frame:
        """Navigate to the correct frame for a BID using BrowserGym's naming convention.

        BIDs like 'a195' encode iframe hierarchy:
        - 'a' is the first iframe
        - 'aA' would be a nested iframe inside 'a'
        - '195' without prefix is in the main frame

        Returns the frame/page where the element with this BID lives.
        """
        current_frame: Page | Frame = self.page

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
        return self._execute_bgym_step("go_back()")

    def browser_forward(self) -> str:
        """Navigate forward in browser history."""
        return self._execute_bgym_step("go_forward()")

    def goto(self, url: str) -> str:
        """Navigate to the specified URL."""
        action_str = f'goto(url="{url}")'
        return self._execute_bgym_step(action_str)

    def noop(self) -> str:
        """No operation action."""
        return self._execute_bgym_step("noop()")

    def _extract_bgym_obs(self) -> dict[str, Any]:
        page = self.page
        if self.config.pre_observation_delay > 0:
            time.sleep(self.config.pre_observation_delay)
        self._wait_dom_loaded()

        for retries_left in reversed(range(EXTRACT_OBS_MAX_TRIES)):
            try:
                _pre_extract(page, tags_to_mark=self.config.tags_to_mark, lenient=(retries_left == 0))
                dom = extract_dom_snapshot(page)
                axtree = extract_merged_axtree(page)
                focused_element_bid = extract_focused_element_bid(page)
                scale_factor = getattr(page, "_bgym_scale_factor", 1.0)
                extra_properties = extract_dom_extra_properties(dom, scale_factor=scale_factor)
            except (Error, MarkingError):
                if retries_left > 0:
                    logger.warning(
                        f"Error extracting BrowserGym observation. Retrying ({retries_left}/{EXTRACT_OBS_MAX_TRIES})."
                    )
                    _post_extract(page)
                    time.sleep(0.5)
                    continue
                raise
            break

        _post_extract(page)
        obs: dict[str, Any] = {
            "dom_object": dom,
            "axtree_object": axtree,
            "extra_element_properties": extra_properties,
            "focused_element_bid": focused_element_bid,
        }
        if self.config.use_screenshot:
            obs["screenshot"] = extract_screenshot(page)
        return obs

    def _bgym_obs_to_cube_obs(self, bgym_obs: dict[str, Any]) -> Observation:
        """Convert BrowserGym observation dict to cube-harness Observation.

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
            obs.contents.append(Content.from_data(html_str, name="pruned_html"))

        # focused_element is placed before axtree so that axtree and screenshot
        # remain the last two items — preserving visibility in agents that use
        # a small render_last_n_steps window (the axtree already marks focused
        # elements inline, so placing focused_element first loses nothing).
        if "focused_element_bid" in bgym_obs:
            focused_bid = bgym_obs["focused_element_bid"]
            if focused_bid:
                obs.contents.append(Content.from_data(focused_bid, name="focused_element"))

        # Add accessibility tree if configured
        if self.config.use_axtree and "axtree_object" in bgym_obs:
            axtree_obj = bgym_obs["axtree_object"]
            if axtree_obj:
                axtree_str = flatten_axtree_to_str(axtree_obj)
                obs.contents.append(Content.from_data(axtree_str, name="axtree_txt"))

        # Add screenshot if configured (convert numpy array to PIL Image)
        if self.config.use_screenshot and "screenshot" in bgym_obs:
            screenshot = bgym_obs["screenshot"]
            if isinstance(screenshot, Image.Image):
                obs.contents.append(Content.from_data(screenshot, name="screenshot"))
            elif isinstance(screenshot, np.ndarray):
                screenshot_img = Image.fromarray(screenshot)
                obs.contents.append(Content.from_data(screenshot_img, name="screenshot"))

        # Add last action error if there was one (raw error message for agent to format)
        if "last_action_error" in bgym_obs:
            error = bgym_obs["last_action_error"]
            if error:
                obs.contents.append(Content.from_data(str(error), name="last_action_error"))

        return obs

    # === BrowserTool utility methods ===

    def evaluate_js(self, js: str) -> Any:
        """Evaluate JavaScript in the browser context and return the result."""
        return self.page.evaluate(js)

    def page_obs(self) -> Observation:
        self._last_obs = self._extract_bgym_obs()
        self._last_info = {"source": "page_obs"}
        self._last_reward = 0.0
        self._last_terminated = False
        return self._bgym_obs_to_cube_obs(self._last_obs)

    def close(self) -> None:
        self._close_runtime()
        self._last_obs = None
        self._last_info = None
        self._last_reward = 0.0
        self._last_terminated = False
