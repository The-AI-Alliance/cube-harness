import asyncio
import base64
import io
import logging
import threading
from contextlib import AsyncExitStack
from typing import Any

import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from PIL import Image
from pydantic import Field

from agentlab2.base import TypedBaseModel
from agentlab2.core import Action, ActionSchema, Content, Observation
from agentlab2.metrics.tracer import GEN_AI_TOOL_CALL_RESULT, tool_span
from agentlab2.tool import AbstractTool, ToolConfig

logger = logging.getLogger(__name__)


class MCPServerConfig(TypedBaseModel):
    """Configuration for a single MCP server.

    For stdio servers, provide command/args/env.
    For HTTP/SSE servers, provide url.
    Transport is auto-detected from provided fields unless explicitly set.
    """

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    url: str | None = None
    transport: str | None = None  # "stdio" | "sse" | "streamable_http"

    def _detect_transport(self) -> str:
        if self.transport:
            return self.transport
        if self.command:
            return "stdio"
        if self.url:
            return "streamable_http"
        raise ValueError("Cannot detect transport: provide 'command' (stdio) or 'url' (sse/streamable_http)")

    def to_server_params(self) -> StdioServerParameters:
        """Convert to MCP SDK server parameters (only for stdio transport)."""
        if self._detect_transport() != "stdio":
            raise ValueError("to_server_params() is only for stdio transport")
        if not self.command:
            raise ValueError("stdio server requires 'command'")
        return StdioServerParameters(command=self.command, args=self.args, env=self.env)


class MCPToolConfig(ToolConfig):
    """Configuration for MCPTool.

    Args:
        servers: Dict mapping server names to their configs.
        name_prefix: If True, prefix tool names with server name (e.g. "server__tool").
            Useful to avoid collisions when multiple servers expose tools with the same name.
        timeout_seconds: Timeout for individual tool calls.
    """

    servers: dict[str, MCPServerConfig]
    name_prefix: bool = False
    timeout_seconds: float = 30.0

    def make(self) -> "MCPTool":
        return MCPTool(config=self)


class _AsyncBridge:
    """Runs an asyncio event loop in a background thread for sync-async bridging."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Any, timeout: float = 120) -> Any:
        """Submit a coroutine to the background loop and block until it completes."""
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("Async bridge is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        if self._loop and not self._loop.is_closed():
            self._loop.close()
        self._loop = None
        self._thread = None


class _MCPToolCore:
    """Async core managing MCP server connections, tool discovery, and execution.

    Runs a single long-lived asyncio Task that owns the AsyncExitStack so that
    connect and disconnect happen in the same task context (required by anyio
    cancel-scope tracking used inside the MCP SDK).
    """

    def __init__(self, config: MCPToolConfig) -> None:
        self._config = config
        self._sessions: dict[str, ClientSession] = {}
        self._action_schemas: list[ActionSchema] = []
        self._tool_to_session: dict[str, ClientSession] = {}
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._lifecycle_task: asyncio.Task[None] | None = None

    async def start(self) -> list[ActionSchema]:
        """Launch the lifecycle task, wait for connections and tool discovery."""
        ready: asyncio.Event = asyncio.Event()
        error_holder: list[BaseException] = []
        self._shutdown_event = asyncio.Event()
        self._lifecycle_task = asyncio.get_running_loop().create_task(
            self._lifecycle(ready, error_holder),
        )
        await ready.wait()
        if error_holder:
            raise error_holder[0]
        return self._action_schemas

    async def _lifecycle(self, ready: asyncio.Event, error_holder: list[BaseException]) -> None:
        """Single task that owns the AsyncExitStack for the entire connection lifetime."""
        try:
            async with AsyncExitStack() as stack:
                for server_name, server_config in self._config.servers.items():
                    try:
                        session = await self._connect_server(server_name, server_config, stack)
                        self._sessions[server_name] = session
                    except Exception as e:
                        logger.error(f"Failed to connect to MCP server '{server_name}': {e}")
                        error_holder.append(e)
                        ready.set()
                        return

                await self._discover_tools()
                ready.set()

                # Keep the stack alive until told to shut down
                await self._shutdown_event.wait()
            # AsyncExitStack.__aexit__ runs here, in the same task that entered it
        finally:
            self._sessions.clear()
            self._tool_to_session.clear()
            self._action_schemas.clear()

    async def _connect_server(self, name: str, config: MCPServerConfig, stack: AsyncExitStack) -> ClientSession:
        """Connect to a single MCP server and return its session."""
        transport = config._detect_transport()

        if transport == "stdio":
            params = config.to_server_params()
            streams = await stack.enter_async_context(stdio_client(params))
            read_stream, write_stream = streams
        elif transport == "sse":
            assert config.url is not None, "SSE transport requires 'url'"
            streams = await stack.enter_async_context(sse_client(config.url))
            read_stream, write_stream = streams
        elif transport == "streamable_http":
            assert config.url is not None, "Streamable HTTP transport requires 'url'"
            streams = await stack.enter_async_context(streamable_http_client(config.url))
            read_stream, write_stream = streams[0], streams[1]
        else:
            raise ValueError(f"Unknown transport '{transport}' for server '{name}'")

        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        logger.info(f"Connected to MCP server '{name}' via {transport}")
        return session

    async def _discover_tools(self) -> None:
        """Discover tools from all connected sessions and build routing map."""
        self._action_schemas = []
        self._tool_to_session = {}

        for server_name, session in self._sessions.items():
            result = await session.list_tools()
            for mcp_tool in result.tools:
                action_name = f"{server_name}__{mcp_tool.name}" if self._config.name_prefix else mcp_tool.name

                if action_name in self._tool_to_session:
                    existing_server = next(
                        sn for sn, sess in self._sessions.items() if sess is self._tool_to_session[action_name]
                    )
                    raise ValueError(
                        f"Tool name collision: '{action_name}' exists in both '{existing_server}' and "
                        f"'{server_name}'. Set name_prefix=True in MCPToolConfig to disambiguate."
                    )

                schema = _mcp_tool_to_action_schema(action_name, mcp_tool)
                self._action_schemas.append(schema)
                self._tool_to_session[action_name] = session

        logger.info(f"Discovered {len(self._action_schemas)} MCP tools: {[s.name for s in self._action_schemas]}")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str | list[dict[str, Any]]:
        """Call an MCP tool by name, routing to the correct server session."""
        session = self._tool_to_session.get(name)
        if session is None:
            raise ValueError(f"Unknown MCP tool: '{name}'")

        # Strip server prefix to get the original MCP tool name
        mcp_name = name
        if self._config.name_prefix:
            for server_name in self._sessions:
                prefix = f"{server_name}__"
                if name.startswith(prefix):
                    mcp_name = name[len(prefix) :]
                    break

        result = await session.call_tool(mcp_name, arguments or {})

        if result.isError:
            error_text = _extract_text(result.content)
            raise RuntimeError(f"MCP tool '{mcp_name}' returned error: {error_text}")

        return _convert_mcp_result(result)

    async def stop(self) -> None:
        """Signal the lifecycle task to shut down and wait for cleanup."""
        self._shutdown_event.set()
        if self._lifecycle_task:
            await self._lifecycle_task
            self._lifecycle_task = None


class MCPTool(AbstractTool):
    """Tool that connects to MCP servers and exposes their tools as AgentLab2 actions.

    Manages connections to one or more MCP servers, discovers their tools
    via list_tools(), and routes execute_action() calls to the correct server.

    Uses a background event loop thread to bridge the async MCP SDK with the
    synchronous AgentLab2 tool interface.

    Example:
        config = MCPToolConfig(servers={
            "fs": MCPServerConfig(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
        })
        tool = config.make()
        tool.reset()       # connects to servers
        print(tool.action_set)
        tool.close()       # disconnects
    """

    def __init__(self, config: MCPToolConfig) -> None:
        self.config = config
        self._bridge = _AsyncBridge()
        self._core: _MCPToolCore | None = None
        self._action_schemas: list[ActionSchema] = []
        self._connected = False

    def _connect(self) -> None:
        """Connect to all configured MCP servers."""
        if self._connected:
            return
        self._bridge.start()
        self._core = _MCPToolCore(self.config)
        self._action_schemas = self._bridge.run(self._core.start())
        self._connected = True

    def reset(self) -> None:
        """Connect to MCP servers if not already connected."""
        if not self._connected:
            self._connect()

    def execute_action(self, action: Action) -> Observation:
        """Execute an action by routing to the appropriate MCP server."""
        assert self._core is not None, "MCPTool core is not initialized"
        if not self._connected:
            self._connect()

        with tool_span(action) as span:
            try:
                result = self._bridge.run(
                    self._core.call_tool(action.name, action.arguments),
                    timeout=self.config.timeout_seconds,
                )
            except Exception as e:
                result = f"Error executing MCP action '{action.name}': {e}"
                logger.exception(result)

            span.set_attribute(GEN_AI_TOOL_CALL_RESULT, str(result))

        if isinstance(result, list):
            contents = [Content(data=item["data"], tool_call_id=action.id, name=item.get("name")) for item in result]
            return Observation(contents=contents)
        return Observation(contents=[Content(data=result, tool_call_id=action.id)])

    @property
    def action_set(self) -> list[ActionSchema]:
        """Return discovered MCP tools as ActionSchemas."""
        if not self._connected:
            self._connect()
        return self._action_schemas

    def close(self) -> None:
        """Disconnect from all MCP servers and stop the background event loop."""
        if self._core and self._connected:
            try:
                self._bridge.run(self._core.stop())
            except Exception as e:
                logger.warning(f"Error disconnecting MCP servers: {e}")
        self._bridge.stop()
        self._core = None
        self._action_schemas = []
        self._connected = False


def _mcp_tool_to_action_schema(name: str, mcp_tool: mcp_types.Tool) -> ActionSchema:
    """Convert an MCP Tool definition to an AgentLab2 ActionSchema."""
    return ActionSchema(
        name=name,
        description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
        parameters=mcp_tool.inputSchema,
    )


def _convert_mcp_result(result: mcp_types.CallToolResult) -> str | list[dict[str, Any]]:
    """Convert MCP CallToolResult to a string or list of content dicts.

    Text-only results are joined into a single string.
    Mixed results (text + images) return a list of dicts with 'data' and optional 'name' keys.
    """
    if not result.content:
        return "Success"

    texts: list[str] = []
    mixed: list[dict[str, Any]] = []

    for block in result.content:
        if isinstance(block, mcp_types.TextContent):
            texts.append(block.text)
            mixed.append({"data": block.text})
        elif isinstance(block, mcp_types.ImageContent):
            img_bytes = base64.b64decode(block.data)
            img = Image.open(io.BytesIO(img_bytes))
            img.load()
            mixed.append({"data": img, "name": "mcp_image"})
        else:
            mixed.append({"data": str(block)})

    if len(mixed) == len(texts):
        return "\n".join(texts) if texts else "Success"

    return mixed


def _extract_text(content: list[Any]) -> str:
    """Extract text from MCP content blocks for error messages."""
    parts = []
    for block in content:
        if isinstance(block, mcp_types.TextContent):
            parts.append(block.text)
        else:
            parts.append(str(block))
    return "\n".join(parts) if parts else "Unknown error"
