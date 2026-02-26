"""Tests for MCP server integration."""

import pytest
from mcp.types import TextContent

from agentlab2.mcp.server import McpServer, McpServerConfig
from tests.conftest import MockTool


class TestMcpServer:
    @pytest.fixture
    def server(self) -> McpServer:
        tool = MockTool()
        return McpServer(tool=tool, config=McpServerConfig(server_name="test"))

    def test_registers_all_tool_actions(self, server: McpServer) -> None:
        """All MockTool actions should be registered as MCP tools."""
        tools = server.raw._tool_manager._tools
        assert "click" in tools
        assert "type_text" in tools
        assert len(tools) == 2

    def test_tool_schemas_have_correct_parameters(self, server: McpServer) -> None:
        """Registered tools should have parameter schemas matching the original methods."""
        click_tool = server.raw._tool_manager._tools["click"]
        params = click_tool.parameters

        assert "properties" in params
        assert "element_id" in params["properties"]

        type_text_tool = server.raw._tool_manager._tools["type_text"]
        params = type_text_tool.parameters

        assert "element_id" in params["properties"]
        assert "text" in params["properties"]

    def test_tool_descriptions(self, server: McpServer) -> None:
        """Tools should have descriptions from the original method docstrings."""
        click_tool = server.raw._tool_manager._tools["click"]
        assert "Click" in click_tool.description or "click" in click_tool.description.lower()

    @pytest.mark.asyncio
    async def test_call_click_tool(self, server: McpServer) -> None:
        """Calling the click tool should dispatch to MockTool.click."""
        content_list, _raw = await server.raw.call_tool("click", {"element_id": "btn1"})

        assert len(content_list) >= 1
        assert isinstance(content_list[0], TextContent)
        assert "Clicked on btn1" in content_list[0].text

    @pytest.mark.asyncio
    async def test_call_type_text_tool(self, server: McpServer) -> None:
        """Calling type_text should dispatch to MockTool.type_text."""
        content_list, _raw = await server.raw.call_tool("type_text", {"element_id": "input1", "text": "hello"})

        assert len(content_list) >= 1
        assert isinstance(content_list[0], TextContent)
        assert "hello" in content_list[0].text
        assert "input1" in content_list[0].text

    @pytest.mark.asyncio
    async def test_tool_execution_updates_tool_state(self, server: McpServer) -> None:
        """Tool calls should mutate the underlying tool's state."""
        tool: MockTool = server._tool  # type: ignore[assignment]

        await server.raw.call_tool("click", {"element_id": "btn1"})
        await server.raw.call_tool("click", {"element_id": "btn2"})

        assert tool.click_count == 2

    def test_raw_escape_hatch(self, server: McpServer) -> None:
        """raw property should expose the underlying FastMCP instance."""
        from mcp.server.fastmcp import FastMCP

        assert isinstance(server.raw, FastMCP)

    def test_default_config(self) -> None:
        """McpServer should work with default config."""
        tool = MockTool()
        server = McpServer(tool=tool)

        assert server._config.server_name == "agentlab2"
        assert server._config.transport == "stdio"
