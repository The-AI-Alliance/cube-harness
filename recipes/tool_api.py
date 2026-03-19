"""Recipe: Expose the Playwright browser tool as an MCP server and verify tool calling.

Demonstrates the full MCP server roundtrip with a real browser tool:
1. Start a tiny HTTP server serving a test page
2. Create an AsyncPlaywrightTool (headless Chromium)
3. Wrap it with McpServer, which registers BrowserActionSpace as MCP tools
4. Navigate to the test page and verify page content comes back through MCP

Uses AsyncPlaywrightTool so tool execution is natively async — no greenlet
issues, no event-loop blocking, no nest_asyncio.

Prerequisites:
    playwright install chromium

Usage:
    uv run recipes/tool_api.py
"""

import asyncio
import logging
from typing import Any

from cube_browser_tool import PlaywrightConfig
from mcp.types import TextContent

from cube_harness.mcp.server import McpServer


def _get_content_blocks(call_tool_result: Any) -> list:
    """Extract content blocks from FastMCP.call_tool() result.

    call_tool() returns (content_blocks, metadata) tuple.
    """
    if isinstance(call_tool_result, tuple):
        return list(call_tool_result[0])
    return list(call_tool_result)


LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

TEST_PAGE = "<html><body>hi there!</body></html>"
TEST_PORT = 8791


async def start_test_server() -> asyncio.Server:
    """Start a minimal HTTP server that serves a single page."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read(4096)
        response = f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n{TEST_PAGE}"
        writer.write(response.encode())
        await writer.drain()
        writer.close()

    return await asyncio.start_server(handle, "127.0.0.1", TEST_PORT)


EXPECTED_BROWSER_TOOLS = {
    "noop",
    "goto",
    "browser_wait",
    "browser_click",
    "browser_type",
    "browser_press_key",
    "browser_hover",
    "browser_back",
    "browser_forward",
    "browser_drag",
    "browser_select_option",
    "browser_mouse_click_xy",
}


async def main() -> None:
    # -- 0. Start test HTTP server --
    http_server = await start_test_server()
    logger.info("Test HTTP server started on port %d.", TEST_PORT)

    # -- 1. Create AsyncPlaywrightTool --
    tool = PlaywrightConfig(headless=True, use_screenshot=False).make_async()
    await tool.initialize()
    logger.info("AsyncPlaywrightTool initialized (headless Chromium).")

    try:
        # -- 2. Wrap with McpServer --
        server = McpServer(tool=tool)
        mcp = server.raw
        logger.info("McpServer created, tools registered on FastMCP.")

        # -- 3. List registered MCP tools --
        tools = await mcp.list_tools()
        logger.info("Discovered %d MCP tools:", len(tools))
        for t in tools:
            logger.info("  - %s: %s", t.name, t.description[:80] if t.description else "")

        tool_names = {t.name for t in tools}
        missing = EXPECTED_BROWSER_TOOLS - tool_names
        assert not missing, f"Missing expected browser tools: {missing}"
        logger.info("All %d BrowserActionSpace tools registered.", len(EXPECTED_BROWSER_TOOLS))

        # -- 4. Test noop: simplest roundtrip --
        logger.info("Calling noop...")
        blocks = _get_content_blocks(await mcp.call_tool("noop", {}))
        text_blocks = [b for b in blocks if isinstance(b, TextContent)]
        logger.info("  noop returned %d block(s), text: %s", len(blocks), [b.text[:60] for b in text_blocks])
        assert any("Success" in b.text for b in text_blocks), "Expected 'Success' in noop result"

        # -- 5. Test goto: navigate to test page and verify HTML in observation --
        logger.info("Calling goto(url='http://127.0.0.1:%d')...", TEST_PORT)
        blocks = _get_content_blocks(await mcp.call_tool("goto", {"url": f"http://127.0.0.1:{TEST_PORT}"}))
        text_blocks = [b for b in blocks if isinstance(b, TextContent)]
        page_html = " ".join(b.text for b in text_blocks)
        logger.info("  page HTML contains: %s", page_html[:120])
        assert "hi there!" in page_html, f"Expected 'hi there!' in page HTML, got: {page_html[:200]}"

        logger.info("All good! Playwright MCP server roundtrip verified.")

    finally:
        await tool.close()
        http_server.close()
        await http_server.wait_closed()
        logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
