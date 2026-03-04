"""Tests for MCP conversion utilities."""

from cube.core import ActionSchema, Content, Observation
from mcp.types import ImageContent, TextContent
from PIL import Image
from pydantic import BaseModel

from agentlab2.mcp.convert import action_schema_to_mcp_tool, observation_to_mcp_content


class TestActionSchemaToMcpTool:
    def test_basic_conversion(self, sample_action_schema: ActionSchema) -> None:
        tool = action_schema_to_mcp_tool(sample_action_schema)

        assert tool.name == "click"
        assert tool.description == "Click on an element"
        assert tool.inputSchema == sample_action_schema.parameters

    def test_empty_parameters(self) -> None:
        schema = ActionSchema(name="noop", description="Do nothing", parameters={})
        tool = action_schema_to_mcp_tool(schema)

        assert tool.name == "noop"
        assert tool.description == "Do nothing"
        assert tool.inputSchema == {}

    def test_complex_parameters(self) -> None:
        schema = ActionSchema(
            name="browser_type",
            description="Type text into an element",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector"},
                    "text": {"type": "string", "description": "Text to type"},
                },
                "required": ["selector", "text"],
            },
        )
        tool = action_schema_to_mcp_tool(schema)

        assert tool.name == "browser_type"
        assert tool.inputSchema["properties"]["selector"]["type"] == "string"
        assert tool.inputSchema["required"] == ["selector", "text"]


class TestObservationToMcpContent:
    def test_text_content(self) -> None:
        obs = Observation(contents=[Content.from_data("Hello, world!")])
        result = observation_to_mcp_content(obs)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text == "Hello, world!"

    def test_dict_content(self) -> None:
        obs = Observation(contents=[Content.from_data({"key": "value", "num": 42})])
        result = observation_to_mcp_content(obs)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert '"key": "value"' in result[0].text
        assert '"num": 42' in result[0].text

    def test_list_content(self) -> None:
        obs = Observation(contents=[Content.from_data([1, 2, 3])])
        result = observation_to_mcp_content(obs)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text == "[1, 2, 3]"

    def test_image_content(self) -> None:
        img = Image.new("RGB", (10, 10), color="red")
        obs = Observation(contents=[Content.from_data(img)])
        result = observation_to_mcp_content(obs)

        assert len(result) == 1
        assert isinstance(result[0], ImageContent)
        assert result[0].mimeType == "image/png"
        assert len(result[0].data) > 0  # base64 string

    def test_basemodel_content(self) -> None:
        class MyModel(BaseModel):
            name: str
            value: int

        obs = Observation(contents=[Content.from_data(MyModel(name="test", value=42))])
        result = observation_to_mcp_content(obs)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert '"name":"test"' in result[0].text or '"name": "test"' in result[0].text

    def test_multiple_contents(self) -> None:
        img = Image.new("RGB", (5, 5), color="blue")
        obs = Observation(
            contents=[
                Content.from_data("Action result: Clicked button"),
                Content.from_data(img),
                Content.from_data({"html": "<div>page</div>"}),
            ]
        )
        result = observation_to_mcp_content(obs)

        assert len(result) == 3
        assert isinstance(result[0], TextContent)
        assert isinstance(result[1], ImageContent)
        assert isinstance(result[2], TextContent)

    def test_empty_observation(self) -> None:
        obs = Observation(contents=[])
        result = observation_to_mcp_content(obs)

        assert result == []
