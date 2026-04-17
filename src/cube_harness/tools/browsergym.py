import logging
import time
from typing import Any

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
from cube.core import Action, ActionSchema, Content, Observation, StepError
from cube.tool import ToolConfig
from cube.tools.browser import BrowserTool
from cube_browser_playwright.playwright_session import PlaywrightSession, PlaywrightSessionConfig
from PIL import Image
from playwright.sync_api import Error, Frame, Page
from pydantic import Field

from cube_harness.tool import ToolWithTelemetry

try:
    from cube.tool import tool_action
except ImportError:  # pragma: no cover — older cube versions
    def tool_action(fn):  # type: ignore[misc]
        return fn

logger = logging.getLogger(__name__)


class BrowsergymConfig(ToolConfig):
    """Configuration for BrowserGym-style Playwright tool."""

    # Browser configuration (launch parameters)
    browser: PlaywrightSessionConfig = Field(default_factory=PlaywrightSessionConfig)

    # Action configuration
    action_subsets: list[str] = Field(default=["bid", "nav", "tab"])

    # Observation behavior
    tags_to_mark: str = "standard_html"  # "all" or "standard_html"
    pre_observation_delay: float = 0.5

    # Observation configuration
    use_html: bool = True
    use_axtree: bool = True
    use_screenshot: bool = True
    prune_html: bool = True

    # AXTree element attributes — requires extra_element_properties from the DOM snapshot
    axtree_with_visible: bool = False   # label visible elements (vis >= 0.5) as "visible"
    axtree_with_clickable: bool = False  # label clickable elements as "clickable"

    def make(self, container: Any = None) -> "BrowsergymTool":
        return BrowsergymTool(self)


class BrowsergymTool(ToolWithTelemetry, BrowserTool):
    """Browser tool using BrowserGym's action set on a Playwright Page.

    Exposes bgym's native actions (click, fill, scroll, ...) as tool actions.
    Pure browser — chat and infeasibility actions belong to ChatTool.
    """

    def __init__(self, config: BrowsergymConfig) -> None:
        super().__init__()
        self.config = config
        self._action_set = HighLevelActionSet(subsets=config.action_subsets, multiaction=False)
        self._action_schemas: list[ActionSchema] | None = None
        self._session: PlaywrightSession | None = None
        self._last_obs: dict | None = None
        self._last_info: dict | None = None
        self._last_reward: float = 0.0
        self._last_terminated: bool = False

    # === Action set: built from bgym's HighLevelActionSet + @tool_action methods ===

    @property
    def action_set(self) -> list[ActionSchema]:
        if self._action_schemas is None:
            bgym_schemas = _build_action_schemas(self._action_set)
            bgym_names = {s.name for s in bgym_schemas}
            # Also expose @tool_action methods defined on this class
            # (e.g. submit_form, keyboard_type_into) that aren't in bgym's action set.
            extra_schemas = []
            for attr_name in dir(self):
                if attr_name.startswith("_") or attr_name == "action_set" or attr_name in bgym_names:
                    continue
                if any(
                    isinstance(cls.__dict__.get(attr_name), property)
                    for cls in type(self).__mro__
                    if attr_name in cls.__dict__
                ):
                    continue
                is_action = any(
                    getattr(cls.__dict__.get(attr_name), "_is_action", False)
                    for cls in type(self).__mro__
                    if attr_name in cls.__dict__
                )
                if is_action:
                    extra_schemas.append(ActionSchema.from_function(getattr(self, attr_name)))
            self._action_schemas = bgym_schemas + extra_schemas
        return self._action_schemas

    # === Action execution: dispatch custom @tool_action methods or serialise to bgym string ===

    def _execute_action(self, action: Action) -> Observation | StepError:
        """Execute an action: custom @tool_action methods are dispatched directly;
        bgym-native actions are serialised to a bgym action string and executed."""
        method = getattr(self, action.name, None)
        is_custom = method is not None and any(
            getattr(cls.__dict__.get(action.name), "_is_action", False)
            for cls in type(self).__mro__
            if action.name in cls.__dict__
        )
        if is_custom:
            result = method(**action.arguments)
            obs = self.page_obs()
            action_obs = Observation(contents=[Content.from_data(str(result), tool_call_id=action.id)])
            return action_obs + obs
        action_str = _action_to_bgym_string(action)
        result = self._execute_bgym_step(action_str)
        obs = self.page_obs()
        action_obs = Observation(contents=[Content.from_data(result, tool_call_id=action.id)])
        return action_obs + obs

    # === BrowserTool interface ===

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
        return self._last_reward

    @property
    def last_terminated(self) -> bool:
        return self._last_terminated

    def goto(self, url: str) -> None:
        self._execute_bgym_step(f'goto(url="{url}")')

    def noop(self) -> None:
        self._execute_bgym_step("noop()")

    def evaluate_js(self, js: str) -> Any:
        return self.page.evaluate(js)

    def page_obs(self) -> Observation:
        self._last_obs = self._extract_bgym_obs()
        self._last_info = {"source": "page_obs"}
        self._last_reward = 0.0
        self._last_terminated = False
        return self._bgym_obs_to_cube_obs(self._last_obs)

    # === Lifecycle ===

    def reset(self) -> None:
        self._close_runtime()
        self._create_runtime()
        self._wait_dom_loaded()
        self._last_obs = self._extract_bgym_obs()
        self._last_info = {"source": "reset"}
        self._last_reward = 0.0
        self._last_terminated = False

    def close(self) -> None:
        self._close_runtime()
        self._last_obs = None
        self._last_info = None
        self._last_reward = 0.0
        self._last_terminated = False

    def _create_runtime(self) -> None:
        self._session = self.config.browser.make()
        self._session.playwright.selectors.set_test_id_attribute(BROWSERGYM_ID_ATTRIBUTE)

    def _close_runtime(self) -> None:
        if self._session is not None:
            self.session.stop()
            self._session = None

    def _wait_dom_loaded(self) -> None:
        if self._session is None:
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

    # === Core bgym step execution ===

    def _execute_bgym_step(self, action_str: str) -> str:
        """Execute a BrowserGym action string and return a result message.

        Captures three error channels:
        - Python exceptions (e.g. TimeoutError when element not found)
        - report_infeasible_instructions callback (bgym soft failures)
        - send_message_to_user callback (bgym task-completion signals)
        """
        logger.info(f"Execute bgym step: {action_str}")
        result = "Success"

        def send_message_to_user(_: str) -> None:
            assert False, "send_message_to_user should not be called"

        def report_infeasible_instructions(_: str) -> None:
            assert False, "report_infeasible_instructions should not be called"

        try:
            code = self._action_set.to_python_code(action_str)
            execute_python_code(
                code=code,
                page=self.page,
                send_message_to_user=send_message_to_user,
                report_infeasible_instructions=report_infeasible_instructions,
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

    # === Observation extraction ===

    # === Extra actions ===

    def _get_frame_for_bid(self, bid: str) -> Page | Frame:
        """Return the Page or Frame that contains the element with this BID.

        BrowserGym encodes iframe hierarchy into BID prefixes: the leading
        lowercase letters identify which iframe to descend into, and the
        trailing digits identify the element inside that frame.
        E.g. 'a182' is element 182 inside the first iframe ('a').
        """
        current: Page | Frame = self.page
        i = 0
        while i < len(bid) and not bid[i:].isnumeric():
            i += 1
            while i < len(bid) and bid[i].isalpha() and bid[i].isupper():
                i += 1
            if i > 0:
                frame_bid = bid[:i]
                try:
                    frame_elem = current.get_by_test_id(frame_bid)  # type: ignore[union-attr]
                    if frame_elem.count() > 0:
                        current = frame_elem.frame_locator(":scope")  # type: ignore[assignment]
                    else:
                        break
                except Exception:
                    break
        return current

    @tool_action
    def keyboard_type_into(self, bid: str, text: str) -> str:
        """Type text into an element character-by-character, firing keyboard events per character. Use this for fields that show autocomplete suggestions or dynamic dropdowns as you type — fill() sets the value directly and bypasses those events. After typing, call noop() to wait for suggestions to appear, then click the desired suggestion."""
        logger.info(f"keyboard_type_into: bid={bid!r} text={text!r}")
        result = "Success"
        try:
            frame = self._get_frame_for_bid(bid)
            locator = frame.get_by_test_id(bid)
            locator.press_sequentially(text, delay=50)
        except Exception as e:
            result = f"Failed: {type(e).__name__}: {e}"
        self._last_obs = self._extract_bgym_obs()
        self._last_info = {"source": "action", "action": f"keyboard_type_into({bid!r})", "action_error": "" if result == "Success" else result}
        self._last_reward = 0.0
        self._last_terminated = False
        return result

    @tool_action
    def submit_form(self) -> str:
        """Submit a ServiceNow create-record form by calling gsftSubmit() directly in gsft_main.

        The visible React Submit buttons bypass WorkArena's gsftSubmit hook (they use
        sysverb_insert_and_stay, which navigates to a new empty form without writing localStorage).
        Calling gsftSubmit() directly in the gsft_main iframe triggers the WorkArena validation
        patch which writes localStorage[session_sys_id_field] before submitting. Use this
        instead of clicking the visible Submit button on ServiceNow create-record tasks.
        """
        result = "Success"
        try:
            gsft_frame = next((f for f in self.page.frames if f.name == "gsft_main"), None)
            if gsft_frame is None:
                return "Failed: gsft_main frame not found — not on a ServiceNow form"
            # Find a real form element and set sys_action so old_gsftSubmit doesn't throw.
            # WorkArena's patch writes localStorage[session_sys_id_field] before calling
            # old_gsftSubmit, but old_gsftSubmit throws when form.sys_action is null.
            # Wrapping in try/catch (JS side) lets the localStorage write complete even if
            # old_gsftSubmit throws, and the returned dict lets us diagnose what happened.
            ls_info: dict = gsft_frame.evaluate(
                """() => {
                    const btn = document.querySelector('#sysverb_insert');
                    const form = document.querySelector('form');
                    if (form) { form.sys_action = 'sysverb_insert'; }
                    let gsftError = null;
                    try {
                        gsftSubmit(btn, form || null, 'sysverb_insert');
                    } catch(e) {
                        gsftError = e.message;
                    }
                    // Collect relevant localStorage keys for diagnosis
                    const lsKeys = Object.keys(localStorage).filter(k => k.includes('sys_id') || k.includes('session'));
                    const lsSnapshot = {};
                    lsKeys.forEach(k => { lsSnapshot[k] = localStorage.getItem(k); });
                    return {gsftError, lsSnapshot};
                }"""
            )
            ls_snapshot = ls_info.get("lsSnapshot", {})
            gsft_error = ls_info.get("gsftError")
            if gsft_error:
                logger.info(f"submit_form: gsftSubmit threw (expected): {gsft_error}")
            logger.info(f"submit_form: localStorage sys_id keys after call: {ls_snapshot}")
            if not ls_snapshot:
                result = "Failed: gsftSubmit completed but no sys_id written to localStorage"
        except Exception as e:
            result = f"Failed: {type(e).__name__}: {e}"
        self._last_obs = self._extract_bgym_obs()
        self._last_info = {"source": "action", "action": "submit_form()", "action_error": "" if result == "Success" else result}
        self._last_reward = 0.0
        self._last_terminated = False
        return result

    @tool_action
    def js_eval(self, code: str, frame: str = "main") -> str:
        """Evaluate JavaScript in the browser and return the JSON-serialized result.

        Useful for inspecting DOM state, reading localStorage, checking field values,
        or diagnosing why an action isn't working as expected.

        Args:
            code: JavaScript expression to evaluate. The result is JSON-serialized.
                  Example: "document.title"
                  Example: "JSON.stringify(localStorage)"
                  Example: "g_form.getUniqueValue()"
            frame: Frame to evaluate in. "main" = top-level page. Any other string
                   is matched against iframe names (e.g. "gsft_main" for ServiceNow).
        """
        import json

        try:
            if frame == "main":
                target = self.page
            else:
                target = next((f for f in self.page.frames if f.name == frame), None)
                if target is None:
                    return f"Failed: frame {frame!r} not found"
            raw = target.evaluate(f"() => {{ try {{ return {code}; }} catch(e) {{ return 'JS error: ' + e.message; }} }}")
            return json.dumps(raw, default=str)
        except Exception as e:
            return f"Failed: {type(e).__name__}: {e}"

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
                need_extra = self.config.axtree_with_visible or self.config.axtree_with_clickable
                extra_properties = extract_dom_extra_properties(dom, scale_factor=scale_factor) if need_extra else {}
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
            "last_action_error": self._last_info.get("action_error", "") if self._last_info else "",
        }
        if self.config.use_screenshot:
            obs["screenshot"] = extract_screenshot(page)
        return obs

    def _bgym_obs_to_cube_obs(self, bgym_obs: dict[str, Any]) -> Observation:
        """Convert BrowserGym observation dict to cube-harness Observation."""
        obs = Observation()

        extra_properties = bgym_obs.get("extra_element_properties", {})

        # HTML
        if self.config.use_html and "dom_object" in bgym_obs:
            dom_obj = bgym_obs["dom_object"]
            html_str = flatten_dom_to_str(dom_obj, extra_properties=extra_properties)
            if self.config.prune_html:
                html_str = prune_html(html_str)
            obs.contents.append(Content.from_data(html_str, name="pruned_html"))

        # Focused element (placed before axtree so axtree+screenshot remain last)
        if "focused_element_bid" in bgym_obs:
            focused_bid = bgym_obs["focused_element_bid"]
            if focused_bid:
                obs.contents.append(Content.from_data(focused_bid, name="focused_element"))

        # Accessibility tree
        if self.config.use_axtree and "axtree_object" in bgym_obs:
            axtree_obj = bgym_obs["axtree_object"]
            if axtree_obj:
                axtree_str = flatten_axtree_to_str(
                    axtree_obj,
                    extra_properties=extra_properties,
                    with_visible=self.config.axtree_with_visible,
                    with_clickable=self.config.axtree_with_clickable,
                )
                obs.contents.append(Content.from_data(axtree_str, name="axtree_txt"))

        # Screenshot
        if self.config.use_screenshot and "screenshot" in bgym_obs:
            screenshot = bgym_obs["screenshot"]
            if isinstance(screenshot, Image.Image):
                obs.contents.append(Content.from_data(screenshot, name="screenshot"))
            elif isinstance(screenshot, np.ndarray):
                screenshot_img = Image.fromarray(screenshot)
                obs.contents.append(Content.from_data(screenshot_img, name="screenshot"))

        # Last action error
        if "last_action_error" in bgym_obs:
            error = bgym_obs["last_action_error"]
            if error:
                obs.contents.append(Content.from_data(str(error), name="last_action_error"))

        # User messages from send_msg_to_user callback
        if self._last_info and self._last_info.get("user_messages"):
            for msg in self._last_info["user_messages"]:
                obs.contents.append(Content.from_data(msg, name="user_message"))

        return obs


# === Module-level helpers ===

# Descriptions that replace BrowserGym's upstream text.
# Use sparingly — only when the upstream description is misleading about when to use the action.
_ACTION_DESCRIPTION_OVERRIDES: dict[str, str] = {
    "fill": (
        "Fill a form field by setting its value directly. Works for <input>, <textarea> and "
        "[contenteditable] elements. Does NOT fire keyboard events — autocomplete suggestions "
        "and dynamic dropdowns will not appear. Use keyboard_type_into() for fields that show "
        "dropdown suggestions as you type."
    ),
}


def _build_action_schemas(action_set: HighLevelActionSet) -> list[ActionSchema]:
    """Convert bgym's HighLevelActionSet to a list of ActionSchema objects."""
    tool_descs = action_set.to_tool_description(api="openai")
    schemas = []
    for desc in tool_descs:
        # "type": "function" is at the top-level desc dict, not inside parameters.
        # parameters already has "type": "object" which Azure/OpenAI require — don't remove it.
        params = desc.get("parameters", {})
        name = desc["name"]
        description = _ACTION_DESCRIPTION_OVERRIDES.get(name, desc.get("description", name))
        schemas.append(ActionSchema(name=name, description=description, parameters=params))
    return schemas


def _action_to_bgym_string(action: Action) -> str:
    """Serialise a cube Action into a BrowserGym action string like 'click(bid="a51")'."""
    args_parts = []
    for key, value in action.arguments.items():
        args_parts.append(f"{key}={repr(value)}")
    return f"{action.name}({', '.join(args_parts)})"
