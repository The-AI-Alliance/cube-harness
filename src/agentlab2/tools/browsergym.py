import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from browsergym.core import _get_global_playwright
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
from PIL import Image
from playwright.sync_api import Browser, BrowserContext, Error, Frame, Page

from agentlab2.action_spaces.browser_action_space import BidBrowserActionSpace
from agentlab2.core import Action, Content, Observation
from agentlab2.tool import Tool, ToolConfig

logger = logging.getLogger(__name__)

EXTRACTION_RETRY_ERRORS = (
    "Frame was detached",
    "Frame with the given frameId is not found",
    "Execution context was destroyed",
    "Frame has been detached",
    "Cannot mark a child frame without a bid",
    "Cannot read properties of undefined",
)


class BrowsergymConfig(ToolConfig):
    """Configuration for BrowserGym-style Playwright tool."""

    # Browser configuration
    headless: bool = True
    viewport: dict | None = None
    slow_mo: int | None = None
    timeout: int | None = None
    locale: str | None = None
    timezone_id: str | None = None
    resizeable_window: bool = False

    # Playwright customization
    pw_chromium_kwargs: dict = {}
    pw_context_kwargs: dict = {}
    record_video_dir: str | None = None

    # Observation behavior
    tags_to_mark: str = "standard_html"  # "all" or "standard_html"
    pre_observation_delay: float = 0.0
    use_html: bool = True
    use_axtree: bool = True
    use_screenshot: bool = True
    prune_html: bool = True

    # Action behavior
    max_wait: int = 60

    def make(self) -> "BrowsergymTool":
        return BrowsergymTool(self)


class BrowsergymTool(Tool, BidBrowserActionSpace):
    """Playwright-based tool that reuses BrowserGym observation utilities."""

    action_space = BidBrowserActionSpace

    def __init__(self, config: BrowsergymConfig) -> None:
        super().__init__()
        self.config = config
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._last_obs: dict | None = None
        self._last_info: dict | None = None
        self._last_reward: float = 0.0
        self._last_terminated: bool = False

    def _ensure_page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser is not initialized. Call reset() first.")
        return self._page

    @property
    def page(self) -> Page:
        return self._ensure_page()

    @property
    def last_reward(self) -> float:
        return self._last_reward

    @property
    def last_terminated(self) -> bool:
        return self._last_terminated

    def _build_launch_args(self, viewport: dict[str, int]) -> list[str]:
        args = [
            (
                f"--window-size={viewport['width']},{viewport['height']}"
                if self.config.resizeable_window
                else None
            ),
            "--disable-features=OverlayScrollbars,ExtendedOverlayScrollbars",
        ]
        return [arg for arg in args if arg is not None]

    def _create_runtime(self) -> None:
        viewport = self.config.viewport or {"width": 1280, "height": 720}
        pw = _get_global_playwright()
        pw.selectors.set_test_id_attribute(BROWSERGYM_ID_ATTRIBUTE)

        self._browser = pw.chromium.launch(
            headless=self.config.headless,
            slow_mo=self.config.slow_mo,
            args=self._build_launch_args(viewport),
            ignore_default_args=["--hide-scrollbars"],
            **self.config.pw_chromium_kwargs,
        )
        self._context = self._browser.new_context(
            no_viewport=True if self.config.resizeable_window else None,
            viewport=viewport if not self.config.resizeable_window else None,
            record_video_dir=Path(self.config.record_video_dir) / "task_video" if self.config.record_video_dir else None,
            record_video_size=viewport,
            locale=self.config.locale,
            timezone_id=self.config.timezone_id,
            **self.config.pw_context_kwargs,
        )
        if self.config.timeout is not None:
            self._context.set_default_timeout(self.config.timeout)
        self._page = self._context.new_page()

    def _close_runtime(self) -> None:
        if self._context is not None:
            try:
                self._context.close()
            except Exception as e:
                logger.warning(f"Error closing BrowserGym context: {e}")
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception as e:
                logger.warning(f"Error closing BrowserGym browser: {e}")
        self._browser = None
        self._context = None
        self._page = None

    def _wait_dom_loaded(self) -> None:
        if self._context is None:
            return
        for page in self._context.pages:
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

    def execute_action(self, action: Action) -> Observation:
        action_obs = super().execute_action(action)
        action_obs += self.page_obs()
        return action_obs

    # === BidBrowserActionSpace protocol implementation ===

    def browser_press_key(self, key: str) -> None:
        self.page.keyboard.press(key)
        self._last_info = {"action": "browser_press_key"}

    def browser_type(self, bid: str, text: str) -> None:
        frame = self._get_frame_for_bid(bid)
        frame.get_by_test_id(bid).fill(text)
        self._last_info = {"action": "browser_type", "bid": bid}

    def browser_click(self, bid: str) -> None:
        state_before = self._get_checkbox_state(bid)
        frame = self._get_frame_for_bid(bid)
        frame.get_by_test_id(bid).click(timeout=3000)

        if state_before is not None:
            state_after = self._get_checkbox_state(bid)
            if state_after == state_before:
                self._toggle_checkbox_js(bid, not state_before)
                logger.info(f"Checkbox/radio {bid} clicked with JS fallback.")

        self._last_info = {"action": "browser_click", "bid": bid}

    def browser_drag(self, from_bid: str, to_bid: str) -> None:
        from_frame = self._get_frame_for_bid(from_bid)
        to_frame = self._get_frame_for_bid(to_bid)
        from_locator = from_frame.get_by_test_id(from_bid)
        to_locator = to_frame.get_by_test_id(to_bid)
        from_locator.drag_to(to_locator)
        self._last_info = {"action": "browser_drag", "from_bid": from_bid, "to_bid": to_bid}

    def browser_hover(self, bid: str) -> None:
        frame = self._get_frame_for_bid(bid)
        frame.get_by_test_id(bid).hover(timeout=3000)
        self._last_info = {"action": "browser_hover", "bid": bid}

    def browser_select_option(self, bid: str, value: str) -> None:
        frame = self._get_frame_for_bid(bid)
        frame.get_by_test_id(bid).select_option(value=value)
        self._last_info = {"action": "browser_select_option", "bid": bid}

    def browser_mouse_click_xy(self, x: int, y: int) -> None:
        self.page.mouse.click(x, y, delay=100)
        self._last_info = {"action": "browser_mouse_click_xy", "x": x, "y": y}

    def browser_wait(self, seconds: int) -> None:
        time.sleep(min(seconds, self.config.max_wait))
        self._last_info = {"action": "browser_wait", "seconds": seconds}

    def browser_back(self) -> None:
        self.page.go_back()
        self._last_info = {"action": "browser_back"}

    def browser_forward(self) -> None:
        self.page.go_forward()
        self._last_info = {"action": "browser_forward"}

    def browser_scroll(self, delta_x: float, delta_y: float) -> None:
        self.page.mouse.wheel(delta_x, delta_y)
        self._last_info = {"action": "browser_scroll", "delta_x": delta_x, "delta_y": delta_y}

    def browser_dbclick(self, bid: str) -> None:
        frame = self._get_frame_for_bid(bid)
        frame.get_by_test_id(bid).dblclick(timeout=3000)
        self._last_info = {"action": "browser_dbclick", "bid": bid}

    def browser_press(self, bid: str, comb: str) -> None:
        frame = self._get_frame_for_bid(bid)
        frame.get_by_test_id(bid).press(comb)
        self._last_info = {"action": "browser_press", "bid": bid, "comb": comb}

    def browser_clear(self, bid: str) -> None:
        frame = self._get_frame_for_bid(bid)
        frame.get_by_test_id(bid).clear()
        self._last_info = {"action": "browser_clear", "bid": bid}

    def browser_focus(self, bid: str) -> None:
        frame = self._get_frame_for_bid(bid)
        frame.get_by_test_id(bid).focus()
        self._last_info = {"action": "browser_focus", "bid": bid}
    
    def browser_goto(self, url: str) -> None:
        self.goto(url)
        self._last_info = {"action": "browser_goto", "url": url}

    def noop(self) -> None:
        self._last_info = {"action": "noop"}

    def _get_frame_for_bid(self, bid: str) -> Page | Frame:
        current_frame: Page | Frame = self.page
        i = 0
        while i < len(bid) and not bid[i:].isnumeric():
            i += 1
            while i < len(bid) and bid[i].isalpha() and bid[i].isupper():
                i += 1
            if i > 0:
                frame_bid = bid[:i]
                try:
                    frame_elem = current_frame.get_by_test_id(frame_bid)
                    if frame_elem.count() > 0:
                        candidate = frame_elem.frame_locator(":scope")
                        current_frame = candidate
                    else:
                        break
                except Exception:
                    break
        return current_frame

    def _get_checkbox_state(self, bid: str) -> bool | None:
        try:
            frame = self._get_frame_for_bid(bid)
            locator = frame.get_by_test_id(bid)
            if locator.count() == 0:
                return None
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
                return bool(result.get("checked"))
            return None
        except Exception:
            return None

    def _toggle_checkbox_js(self, bid: str, checked: bool) -> None:
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
            logger.debug("Could not toggle checkbox via JavaScript fallback.", exc_info=True)

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
            except (Error, MarkingError) as e:
                err_msg = str(e)
                if retries_left > 0 and any(retry_error in err_msg for retry_error in EXTRACTION_RETRY_ERRORS):
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

    def _bgym_obs_to_agentlab_obs(self, bgym_obs: dict[str, Any]) -> Observation:
        obs = Observation()
        if self.config.use_html and "dom_object" in bgym_obs:
            html_str = flatten_dom_to_str(
                bgym_obs["dom_object"],
                extra_properties=bgym_obs.get("extra_element_properties", {}),
                with_visible=False,
                filter_visible_only=False,
            )
            if self.config.prune_html:
                html_str = prune_html(html_str)
            obs.contents.append(Content(data=html_str, name="pruned_html"))

        if self.config.use_axtree and "axtree_object" in bgym_obs:
            axtree_obj = bgym_obs["axtree_object"]
            if axtree_obj:
                obs.contents.append(Content(data=flatten_axtree_to_str(axtree_obj), name="axtree_txt"))

        if self.config.use_screenshot and "screenshot" in bgym_obs:
            screenshot = bgym_obs["screenshot"]
            if isinstance(screenshot, Image.Image):
                obs.contents.append(Content(data=screenshot, name="screenshot"))
            elif isinstance(screenshot, np.ndarray):
                obs.contents.append(Content(data=Image.fromarray(screenshot), name="screenshot"))

        return obs

    # === BrowserTaskTool utility methods ===


    def evaluate_js(self, js: str) -> Any:
        return self.page.evaluate(js)

    def page_obs(self) -> Observation:
        self._last_obs = self._extract_bgym_obs()
        self._last_info = {"source": "page_obs"}
        self._last_reward = 0.0
        self._last_terminated = False
        return self._bgym_obs_to_agentlab_obs(self._last_obs)

    def close(self) -> None:
        self._close_runtime()
        self._last_obs = None
        self._last_info = None
        self._last_reward = 0.0
        self._last_terminated = False
