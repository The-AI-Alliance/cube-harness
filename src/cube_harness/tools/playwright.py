import asyncio
import logging
from io import BytesIO

from cube.core import Action, Content, Observation, StepError
from cube_browser_tool import PlaywrightConfig
from PIL import Image
from playwright.async_api import Page as AsyncPage
from playwright.async_api import async_playwright

from cube_harness.action_spaces.browser_action_space import BrowserActionSpace
from cube_harness.tool import AsyncToolWithTelemetry
from cube_harness.utils import prune_html

logger = logging.getLogger(__name__)


class AsyncPlaywrightTool(AsyncToolWithTelemetry, BrowserActionSpace):
    """Fully asynchronous Playwright tool using playwright.async_api."""

    def __init__(self, config: PlaywrightConfig) -> None:
        super().__init__()
        self.config = config
        self._apw = None
        self._abrowser = None
        self._page: AsyncPage = None  # type: ignore

    async def initialize(self):
        self._apw = await async_playwright().start()
        self._abrowser = await self._apw.chromium.launch(chromium_sandbox=True, **self.config.pw_kwargs)
        self._page = await self._abrowser.new_page()

    async def _execute_action(self, action: Action) -> Observation | StepError:
        result = await super()._execute_action(action)
        if isinstance(result, StepError):
            return result
        try:
            result += await self.page_obs()
        except Exception as e:
            return StepError.from_exception(e)
        return result

    async def browser_press_key(self, key: str):
        """Press a key on the keyboard."""
        await self._page.keyboard.press(key)

    async def browser_type(self, selector: str, text: str):
        """Type text into the focused element."""
        await self._page.type(selector, text)

    async def browser_click(self, selector: str):
        """Click on a selector."""
        await self._page.click(selector, timeout=3000, strict=True)

    async def browser_drag(self, from_selector: str, to_selector: str):
        """Drag and drop from one selector to another."""
        from_elem = self._page.locator(from_selector)
        await from_elem.hover(timeout=500)
        await self._page.mouse.down()

        to_elem = self._page.locator(to_selector)
        await to_elem.hover(timeout=500)
        await self._page.mouse.up()

    async def browser_hover(self, selector: str):
        """Hover over a given element."""
        await self._page.hover(selector, timeout=3000, strict=True)

    async def browser_select_option(self, selector: str, value: str):
        """Select an option from a given element."""
        await self._page.select_option(selector, value)

    async def browser_mouse_click_xy(self, x: int, y: int):
        """Click at a given x, y coordinate using the mouse."""
        await self._page.mouse.click(x, y, delay=100)

    async def browser_wait(self, seconds: int):
        """Wait for a given number of seconds, up to max_wait."""
        await asyncio.sleep(min(seconds, self.config.max_wait))

    async def browser_back(self):
        """Navigate back in browser history."""
        await self._page.go_back()

    async def browser_forward(self):
        """Navigate forward in browser history."""
        await self._page.go_forward()

    async def noop(self):
        """No operation action."""
        pass

    async def evaluate_js(self, js: str):
        js_result = await self._page.evaluate(js)
        logger.info(f"JS result: {js_result}")
        return js_result

    async def goto(self, url: str):
        await self._page.goto(url)

    async def page_html(self) -> str:
        return await self._page.content()

    async def page_screenshot(self) -> Image.Image:
        scr_bytes = await self._page.screenshot()
        return Image.open(BytesIO(scr_bytes))

    async def page_axtree(self) -> str:
        axtree = await self._page.accessibility.snapshot()
        return flatten_axtree(axtree)

    async def page_obs(self) -> Observation:
        obs = Observation()
        if self.config.use_html:
            html = await self.page_html()
            if self.config.prune_html:
                obs.contents.append(Content.from_data(prune_html(html), name="pruned_html"))
            else:
                obs.contents.append(Content.from_data(html, name="html"))
        if self.config.use_axtree:
            obs.contents.append(Content.from_data(await self.page_axtree(), name="axtree_txt"))
        if self.config.use_screenshot:
            obs.contents.append(Content.from_data(await self.page_screenshot(), name="screenshot"))
        return obs

    async def close(self):
        await self._page.close()
        await self._abrowser.close()  # type: ignore
        await self._apw.stop()  # type: ignore


def flatten_axtree(axtree_dict: dict | None) -> str:
    """
    Traverses accessibility tree dictionary and returns its markdown view.

    Args:
        axtree_dict: Accessibility tree from playwright page.accessibility.snapshot()
                     Structure: dict with 'role', 'name', 'value', 'children' keys

    Returns:
        String representation of the accessibility tree in markdown format
    """
    if axtree_dict is None:
        return ""

    def traverse_node(node: dict, depth: int = 0) -> list[str]:
        """Recursively traverse the accessibility tree and build markdown lines."""
        lines = []
        indent = "  " * depth  # 2 spaces per indent level

        # Extract node information
        role = node.get("role", "")
        name = node.get("name", "")
        value = node.get("value", "")

        # Build the node representation
        parts = []
        if role:
            parts.append(f"{role}:")
        if name.strip():
            parts.append(f"{name}")
        if value:
            parts.append(f"[value: {value}]")

        # Only add line if there's meaningful content
        if parts:
            line = f"{indent}{' '.join(parts)}"
            lines.append(line)

        # Recursively process children
        children = node.get("children", [])
        for child in children:
            child_lines = traverse_node(child, depth + 1)
            lines.extend(child_lines)

        return lines

    # Start traversal from root
    all_lines = traverse_node(axtree_dict, depth=0)
    return "\n".join(all_lines)
