"""Conversion utilities between cube-harness types and MCP types."""

import base64
import io
import json

from cube.core import ActionSchema, Content, Observation
from mcp.types import ImageContent, TextContent, Tool
from PIL import Image
from pydantic import BaseModel


def action_schema_to_mcp_tool(schema: ActionSchema) -> Tool:
    """Convert an cube-harness ActionSchema to an MCP Tool definition.

    Maps:
        schema.name → tool.name
        schema.description → tool.description
        schema.parameters → tool.inputSchema
    """
    return Tool(
        name=schema.name,
        description=schema.description,
        inputSchema=schema.parameters,
    )


def observation_to_mcp_content(obs: Observation) -> list[TextContent | ImageContent]:
    """Convert an cube-harness Observation to a list of MCP content items.

    Maps:
        str Content.data → TextContent
        dict/list/BaseModel Content.data → TextContent (json-serialized)
        PIL Image Content.data → ImageContent (base64-encoded PNG)
    """
    result: list[TextContent | ImageContent] = []
    for content in obs.contents:
        result.append(_content_to_mcp(content))
    return result


def _content_to_mcp(content: Content) -> TextContent | ImageContent:
    """Convert a single cube-harness Content item to an MCP content item."""
    data = content.data

    if isinstance(data, Image.Image):
        buf = io.BytesIO()
        data.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return ImageContent(type="image", data=b64, mimeType="image/png")

    if isinstance(data, BaseModel):
        text = data.model_dump_json(serialize_as_any=True)
    elif isinstance(data, (dict, list)):
        text = json.dumps(data)
    else:
        text = str(data)

    return TextContent(type="text", text=text)
