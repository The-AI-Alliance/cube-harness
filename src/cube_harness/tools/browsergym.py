import logging
import time
from collections.abc import Callable
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
from playwright.sync_api import Error, Page
from pydantic import Field

from cube_harness.tool import ToolWithTelemetry

logger = logging.getLogger(__name__)


class BrowsergymConfig(ToolConfig):
    """Configuration for BrowserGym-style Playwright tool."""

    # Browser configuration (launch parameters)
    browser: PlaywrightSessionConfig = Field(default_factory=PlaywrightSessionConfig)

    # Action configuration — pure browser actions only.
    # Chat ("send_msg_to_user") and infeasibility ("report_infeasible") are handled
    # by ChatTool when used in a Toolbox. Add "chat"/"infeas" here only if running
    # standalone without a ChatTool.
    action_subsets: list[str] = Field(default=["bid", "nav", "tab"])

    # Observation behavior
    tags_to_mark: str = "standard_html"  # "all" or "standard_html"
    pre_observation_delay: float = 0.5

    # Observation configuration
    use_html: bool = True
    use_axtree: bool = True
    use_screenshot: bool = True
    prune_html: bool = True

    def make(self, container: Any = None) -> "BrowsergymTool":
        return BrowsergymTool(self)


class BrowsergymTool(ToolWithTelemetry, BrowserTool):
    """Browser tool using BrowserGym's action set on a Playwright Page.

    Exposes bgym's native actions (click, fill, scroll, ...) as tool actions.
    Pure browser — chat and infeasibility actions belong to ChatTool.

    Optional callbacks ``on_send_message_to_user`` and ``on_report_infeasible``
    route bgym's callbacks to an external ChatSession when "chat"/"infeas"
    subsets are included for backward compatibility.
    """

    def __init__(
        self,
        config: BrowsergymConfig,
        on_send_message_to_user: Callable[[str], None] | None = None,
        on_report_infeasible: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._on_send_message_to_user = on_send_message_to_user
        self._on_report_infeasible = on_report_infeasible
        self._action_set = HighLevelActionSet(subsets=config.action_subsets, multiaction=False)
        self._action_schemas: list[ActionSchema] | None = None
        self._session: PlaywrightSession | None = None
        self._last_obs: dict | None = None
        self._last_info: dict | None = None
        self._last_reward: float = 0.0
        self._last_terminated: bool = False

    # === Action set: built from bgym's HighLevelActionSet ===

    @property
    def action_set(self) -> list[ActionSchema]:
        if self._action_schemas is None:
            self._action_schemas = _build_action_schemas(self._action_set)
        return self._action_schemas

    # === Action execution: serialise Action -> bgym string -> execute ===

    def _execute_action(self, action: Action) -> Observation | StepError:
        """Serialise an Action to a bgym action string, execute it, and return the observation."""
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
        infeasible_messages: list[str] = []
        user_messages: list[str] = []

        try:
            code = self._action_set.to_python_code(action_str)
            execute_python_code(
                code=code,
                page=self.page,
                send_message_to_user=lambda message: user_messages.append(message),
                report_infeasible_instructions=lambda message: infeasible_messages.append(message),
            )
            if infeasible_messages:
                if self._on_report_infeasible is not None:
                    for msg in infeasible_messages:
                        self._on_report_infeasible(msg)
                error_msg = "; ".join(infeasible_messages)
                self._last_info = {"source": "action", "action": action_str, "action_error": error_msg}
                result = f"Failed (infeasible): {error_msg}"
            else:
                if self._on_send_message_to_user is not None:
                    for msg in user_messages:
                        self._on_send_message_to_user(msg)
                self._last_info = {
                    "source": "action",
                    "action": action_str,
                    "action_error": "",
                    "user_messages": user_messages,
                }
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            self._last_info = {"source": "action", "action": action_str, "action_error": error_msg}
            result = f"Failed: {error_msg}"

        self._last_obs = self._extract_bgym_obs()
        self._last_reward = 0.0
        self._last_terminated = False
        return result

    # === Observation extraction ===

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
            "last_action_error": self._last_info.get("action_error", "") if self._last_info else "",
        }
        if self.config.use_screenshot:
            obs["screenshot"] = extract_screenshot(page)
        return obs

    def _bgym_obs_to_cube_obs(self, bgym_obs: dict[str, Any]) -> Observation:
        """Convert BrowserGym observation dict to cube-harness Observation."""
        obs = Observation()

        # HTML
        if self.config.use_html and "dom_object" in bgym_obs:
            dom_obj = bgym_obs["dom_object"]
            html_str = flatten_dom_to_str(
                dom_obj,
                extra_properties=bgym_obs.get("extra_element_properties", {}),
                with_visible=False,
                filter_visible_only=False,
            )
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
                axtree_str = flatten_axtree_to_str(axtree_obj)
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


def _build_action_schemas(action_set: HighLevelActionSet) -> list[ActionSchema]:
    """Convert bgym's HighLevelActionSet to a list of ActionSchema objects."""
    tool_descs = action_set.to_tool_description(api="openai")
    schemas = []
    for desc in tool_descs:
        # "type": "function" is at the top-level desc dict, not inside parameters.
        # parameters already has "type": "object" which Azure/OpenAI require — don't remove it.
        params = desc.get("parameters", {})
        name = desc["name"]
        description = desc.get("description", name)
        schemas.append(ActionSchema(name=name, description=description, parameters=params))
    return schemas


def _action_to_bgym_string(action: Action) -> str:
    """Serialise a cube Action into a BrowserGym action string like 'click(bid="a51")'."""
    args_parts = []
    for key, value in action.arguments.items():
        args_parts.append(f"{key}={repr(value)}")
    return f"{action.name}({', '.join(args_parts)})"
