"""Tests for agentlab2.tools.mcp module."""

from unittest.mock import AsyncMock, MagicMock, patch

import mcp.types as mcp_types
import pytest
from mcp.client.stdio import StdioServerParameters

from agentlab2.core import Action, ActionSchema
from agentlab2.tools.mcp import (
    MCPServerConfig,
    MCPToolConfig,
    _AsyncBridge,
    _convert_mcp_result,
    _extract_text,
    _mcp_tool_to_action_schema,
)


class TestMCPServerConfig:
    """Tests for MCPServerConfig."""

    def test_stdio_transport_detection(self) -> None:
        config = MCPServerConfig(command="python", args=["-m", "server"])
        assert config._detect_transport() == "stdio"

    def test_url_transport_detection(self) -> None:
        config = MCPServerConfig(url="http://localhost:3001/sse")
        assert config._detect_transport() == "streamable_http"

    def test_explicit_transport(self) -> None:
        config = MCPServerConfig(url="http://localhost:3001/sse", transport="sse")
        assert config._detect_transport() == "sse"

    def test_no_transport_raises(self) -> None:
        config = MCPServerConfig()
        with pytest.raises(ValueError, match="Cannot detect transport"):
            config._detect_transport()

    def test_to_server_params_stdio(self) -> None:
        config = MCPServerConfig(command="python", args=["-m", "server"], env={"KEY": "val"})
        params = config.to_server_params()
        assert isinstance(params, StdioServerParameters)
        assert params.command == "python"
        assert params.args == ["-m", "server"]
        assert params.env == {"KEY": "val"}

    def test_to_server_params_rejects_url(self) -> None:
        config = MCPServerConfig(url="http://localhost:3001")
        with pytest.raises(ValueError, match="only for stdio"):
            config.to_server_params()


class TestMCPToolConfig:
    """Tests for MCPToolConfig."""

    def test_serialization_round_trip(self) -> None:
        config = MCPToolConfig(
            servers={"test": MCPServerConfig(command="echo", args=["hello"])},
            name_prefix=True,
            timeout_seconds=15.0,
        )
        data = config.model_dump()
        restored = MCPToolConfig.model_validate(data)
        assert restored.name_prefix is True
        assert restored.timeout_seconds == 15.0
        assert "test" in restored.servers
        assert restored.servers["test"].command == "echo"


class TestMCPToolToActionSchema:
    """Tests for _mcp_tool_to_action_schema conversion."""

    def test_basic_conversion(self) -> None:
        mcp_tool = mcp_types.Tool(
            name="read_file",
            description="Read a file from disk",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
        )
        schema = _mcp_tool_to_action_schema("read_file", mcp_tool)
        assert isinstance(schema, ActionSchema)
        assert schema.name == "read_file"
        assert schema.description == "Read a file from disk"
        assert "properties" in schema.parameters
        assert "path" in schema.parameters["properties"]

    def test_prefixed_name(self) -> None:
        mcp_tool = mcp_types.Tool(name="read_file", description="Read file", inputSchema={"type": "object"})
        schema = _mcp_tool_to_action_schema("fs__read_file", mcp_tool)
        assert schema.name == "fs__read_file"

    def test_no_description_fallback(self) -> None:
        mcp_tool = mcp_types.Tool(name="my_tool", inputSchema={"type": "object"})
        schema = _mcp_tool_to_action_schema("my_tool", mcp_tool)
        assert "MCP tool: my_tool" in schema.description

    def test_empty_parameters(self) -> None:
        mcp_tool = mcp_types.Tool(name="noop", description="Do nothing", inputSchema={"type": "object"})
        schema = _mcp_tool_to_action_schema("noop", mcp_tool)
        assert schema.parameters == {"type": "object"}

    def test_as_dict_format(self) -> None:
        """Verify the schema produces LLM-compatible dict format."""
        mcp_tool = mcp_types.Tool(
            name="search",
            description="Search for items",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        schema = _mcp_tool_to_action_schema("search", mcp_tool)
        d = schema.as_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "search"
        assert d["function"]["description"] == "Search for items"
        assert d["function"]["parameters"]["properties"]["query"]["type"] == "string"


class TestConvertMCPResult:
    """Tests for _convert_mcp_result."""

    def test_text_only(self) -> None:
        result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="Hello world")],
        )
        assert _convert_mcp_result(result) == "Hello world"

    def test_multiple_text(self) -> None:
        result = mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(type="text", text="Line 1"),
                mcp_types.TextContent(type="text", text="Line 2"),
            ],
        )
        assert _convert_mcp_result(result) == "Line 1\nLine 2"

    def test_empty_content(self) -> None:
        result = mcp_types.CallToolResult(content=[])
        assert _convert_mcp_result(result) == "Success"

    def test_image_content(self) -> None:
        # Create a minimal 1x1 red PNG as base64
        import base64

        from PIL import Image

        img = Image.new("RGB", (1, 1), color="red")
        import io

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        result = mcp_types.CallToolResult(
            content=[mcp_types.ImageContent(type="image", data=img_b64, mimeType="image/png")],
        )
        converted = _convert_mcp_result(result)
        assert isinstance(converted, list)
        assert len(converted) == 1
        assert isinstance(converted[0]["data"], Image.Image)
        assert converted[0]["name"] == "mcp_image"

    def test_mixed_content(self) -> None:
        import base64
        import io

        from PIL import Image

        img = Image.new("RGB", (1, 1), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        result = mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(type="text", text="Description"),
                mcp_types.ImageContent(type="image", data=img_b64, mimeType="image/png"),
            ],
        )
        converted = _convert_mcp_result(result)
        assert isinstance(converted, list)
        assert len(converted) == 2
        assert converted[0]["data"] == "Description"
        assert isinstance(converted[1]["data"], Image.Image)


class TestExtractText:
    """Tests for _extract_text."""

    def test_text_blocks(self) -> None:
        content = [
            mcp_types.TextContent(type="text", text="Error: file not found"),
            mcp_types.TextContent(type="text", text="Check path."),
        ]
        assert _extract_text(content) == "Error: file not found\nCheck path."

    def test_empty_content(self) -> None:
        assert _extract_text([]) == "Unknown error"


class TestAsyncBridge:
    """Tests for _AsyncBridge."""

    def test_start_and_stop(self) -> None:
        bridge = _AsyncBridge()
        bridge.start()
        assert bridge._loop is not None
        assert bridge._loop.is_running()
        bridge.stop()

    def test_run_coroutine(self) -> None:
        bridge = _AsyncBridge()
        bridge.start()

        async def add(a: int, b: int) -> int:
            return a + b

        result = bridge.run(add(2, 3))
        assert result == 5
        bridge.stop()

    def test_run_after_stop_raises(self) -> None:
        bridge = _AsyncBridge()
        bridge.start()
        bridge.stop()

        async def noop() -> None:
            pass

        coro = noop()
        with pytest.raises(RuntimeError, match="not running"):
            bridge.run(coro)
        coro.close()  # prevent "coroutine was never awaited" warning

    def test_double_stop_safe(self) -> None:
        bridge = _AsyncBridge()
        bridge.start()
        bridge.stop()
        bridge.stop()  # should not raise


class TestMCPToolWithMockCore:
    """Tests for MCPTool using mocked _MCPToolCore."""

    def test_make_from_config(self) -> None:
        config = MCPToolConfig(servers={"test": MCPServerConfig(command="echo")})
        tool = config.make()
        assert tool.config is config
        assert not tool._connected

    @patch("agentlab2.tools.mcp._MCPToolCore")
    def test_reset_connects(self, mock_core_cls: MagicMock) -> None:
        mock_core = MagicMock()
        mock_core.connect = AsyncMock()
        mock_core.discover_tools = AsyncMock(return_value=[ActionSchema(name="test_tool", description="A test tool")])
        mock_core.disconnect = AsyncMock()
        mock_core_cls.return_value = mock_core

        config = MCPToolConfig(servers={"s": MCPServerConfig(command="echo")})
        tool = config.make()
        tool.reset()

        assert tool._connected
        assert len(tool.action_set) == 1
        assert tool.action_set[0].name == "test_tool"
        tool.close()

    @patch("agentlab2.tools.mcp._MCPToolCore")
    def test_execute_action_text_result(self, mock_core_cls: MagicMock) -> None:
        mock_core = MagicMock()
        mock_core.connect = AsyncMock()
        mock_core.discover_tools = AsyncMock(return_value=[ActionSchema(name="greet", description="Greet")])
        mock_core.call_tool = AsyncMock(return_value="Hello!")
        mock_core.disconnect = AsyncMock()
        mock_core_cls.return_value = mock_core

        config = MCPToolConfig(servers={"s": MCPServerConfig(command="echo")})
        tool = config.make()
        tool.reset()

        action = Action(id="call_1", name="greet", arguments={"name": "world"})
        obs = tool.execute_action(action)

        assert len(obs.contents) == 1
        assert obs.contents[0].data == "Hello!"
        assert obs.contents[0].tool_call_id == "call_1"
        tool.close()

    @patch("agentlab2.tools.mcp._MCPToolCore")
    def test_execute_action_list_result(self, mock_core_cls: MagicMock) -> None:
        mock_core = MagicMock()
        mock_core.connect = AsyncMock()
        mock_core.discover_tools = AsyncMock(
            return_value=[ActionSchema(name="screenshot", description="Take screenshot")]
        )
        mock_core.call_tool = AsyncMock(return_value=[{"data": "text result"}, {"data": "more", "name": "extra"}])
        mock_core.disconnect = AsyncMock()
        mock_core_cls.return_value = mock_core

        config = MCPToolConfig(servers={"s": MCPServerConfig(command="echo")})
        tool = config.make()
        tool.reset()

        action = Action(id="call_2", name="screenshot", arguments={})
        obs = tool.execute_action(action)

        assert len(obs.contents) == 2
        assert obs.contents[0].data == "text result"
        assert obs.contents[0].tool_call_id == "call_2"
        assert obs.contents[1].name == "extra"
        tool.close()

    @patch("agentlab2.tools.mcp._MCPToolCore")
    def test_execute_action_error_handling(self, mock_core_cls: MagicMock) -> None:
        mock_core = MagicMock()
        mock_core.connect = AsyncMock()
        mock_core.discover_tools = AsyncMock(return_value=[ActionSchema(name="fail", description="Fail")])
        mock_core.call_tool = AsyncMock(side_effect=RuntimeError("Server crashed"))
        mock_core.disconnect = AsyncMock()
        mock_core_cls.return_value = mock_core

        config = MCPToolConfig(servers={"s": MCPServerConfig(command="echo")})
        tool = config.make()
        tool.reset()

        action = Action(id="call_3", name="fail", arguments={})
        obs = tool.execute_action(action)

        assert "Error executing MCP action" in obs.contents[0].data
        assert "Server crashed" in obs.contents[0].data
        tool.close()

    @patch("agentlab2.tools.mcp._MCPToolCore")
    def test_close_resets_state(self, mock_core_cls: MagicMock) -> None:
        mock_core = MagicMock()
        mock_core.connect = AsyncMock()
        mock_core.discover_tools = AsyncMock(return_value=[ActionSchema(name="t", description="t")])
        mock_core.disconnect = AsyncMock()
        mock_core_cls.return_value = mock_core

        config = MCPToolConfig(servers={"s": MCPServerConfig(command="echo")})
        tool = config.make()
        tool.reset()
        assert tool._connected

        tool.close()
        assert not tool._connected
        assert tool._action_schemas == []
        assert tool._core is None

    @patch("agentlab2.tools.mcp._MCPToolCore")
    def test_action_set_triggers_connect(self, mock_core_cls: MagicMock) -> None:
        mock_core = MagicMock()
        mock_core.connect = AsyncMock()
        mock_core.discover_tools = AsyncMock(
            return_value=[ActionSchema(name="tool_a", description="A"), ActionSchema(name="tool_b", description="B")]
        )
        mock_core.disconnect = AsyncMock()
        mock_core_cls.return_value = mock_core

        config = MCPToolConfig(servers={"s": MCPServerConfig(command="echo")})
        tool = config.make()

        # Accessing action_set before reset should auto-connect
        actions = tool.action_set
        assert len(actions) == 2
        assert tool._connected
        tool.close()
