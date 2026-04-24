"""Extra browser actions not covered by BrowserGym's built-in action set.

Provides ExtraWebActionsTool and ExtendedBrowserConfig. Use ExtendedBrowserConfig
in place of BrowsergymConfig when you need keyboard_type_into or js_eval:

    ToolboxConfig(tool_configs=[
        ExtendedBrowserConfig(browser=BrowsergymConfig(...)),
        ChatToolConfig(),
    ])
"""

import json
import logging
from typing import Any

from browsergym.core.action.utils import get_elem_by_bid
from cube.core import Action, Content, Observation, StepError
from cube.tool import Tool, Toolbox, ToolConfig, tool_action
from pydantic import Field

from cube_harness.tools.browsergym import BrowsergymConfig, BrowsergymTool

logger = logging.getLogger(__name__)


class ExtraWebActionsTool(Tool):
    """Extra browser actions that complement BrowserGym's built-in set.

    Holds a reference to a BrowsergymTool to share its page and observation
    extraction — both tools operate on the same browser session.
    """

    def __init__(self, browser: BrowsergymTool) -> None:
        self._browser = browser

    def execute_action(self, action: Action) -> Observation | StepError:
        method = self.get_action_method(action)
        try:
            result = str(method(**action.arguments) or "Success")
        except Exception as e:
            return StepError.from_exception(e)
        action_obs = Observation(contents=[Content.from_data(result, tool_call_id=action.id)])
        return action_obs + self._browser.page_obs()

    @tool_action
    def keyboard_type_into(self, bid: str, text: str) -> str:
        """Type text into an element character-by-character, firing keyboard events per character.

        Use this for fields that show autocomplete suggestions or dynamic dropdowns as you type
        — fill() sets the value directly and bypasses those events. After typing, call noop() to
        wait for suggestions to appear, then click the desired suggestion.
        """
        logger.info(f"keyboard_type_into: bid={bid!r} text={text!r}")
        try:
            get_elem_by_bid(self._browser.page, bid).press_sequentially(text, delay=50)
            return "Success"
        except Exception as e:
            return f"Failed: {type(e).__name__}: {e}"

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
        try:
            if frame == "main":
                target = self._browser.page
            else:
                target = next((f for f in self._browser.page.frames if f.name == frame), None)
                if target is None:
                    return f"Failed: frame {frame!r} not found"
            raw = target.evaluate(
                f"() => {{ try {{ return {code}; }} catch(e) {{ return 'JS error: ' + e.message; }} }}"
            )
            return json.dumps(raw, default=str)
        except Exception as e:
            return f"Failed: {type(e).__name__}: {e}"


class ExtendedBrowserConfig(ToolConfig):
    """BrowserGym tool bundled with ExtraWebActionsTool, returned as a flat Toolbox.

    Drop-in replacement for BrowsergymConfig when extra web actions are needed.
    When nested inside ToolboxConfig, the Toolbox is automatically flattened.
    """

    browser: BrowsergymConfig = Field(default_factory=BrowsergymConfig)

    def make(self, container: Any = None) -> Toolbox:
        bgym = self.browser.make(container)
        extra = ExtraWebActionsTool(bgym)
        return Toolbox([bgym, extra])
