# MCP Server Integration

**Module:** `cube_harness.mcp`

## Purpose

Expose any `AbstractTool` / `AbstractAsyncTool` as an MCP (Model Context Protocol)
server. Agents built on MCP-native frameworks (Claude Desktop, MCP clients, etc.)
can drive CUBE tools directly without going through the harness's episode loop.

## Public API

### `McpServerConfig`
```python
class McpServerConfig(TypedBaseModel):
    server_name: str = "cube_harness"
    transport: Literal["stdio", "sse"] = "stdio"
    host: str = "localhost"
    port: int = 8000                   # used only for sse transport
```

### `McpServer`
```python
class McpServer:
    def __init__(self, tool: AbstractTool | AbstractAsyncTool, config: McpServerConfig | None = None)

    @property
    def raw(self) -> FastMCP           # escape hatch — constitution SR-003

    def run(self) -> None              # blocks; runs the FastMCP event loop
```

On init, iterates `tool.action_set` and registers each action as an MCP tool on
`FastMCP`, preserving name and description.

### `observation_to_mcp_content` (`cube_harness.mcp.convert`)
Converts a CUBE `Observation` into a list of MCP content items (`TextContent`,
`ImageContent`). Handles all `Content` subclasses defined by cube-standard.

## Contracts

- MCP tool handlers use `FastMCP`'s async interface. Sync `AbstractTool` actions are
  wrapped in an async adapter (`_make_async_handler`) that runs them in a thread.
- MCP clients expect each tool to return content; CUBE `Observation.to_llm_messages`
  is not directly compatible with MCP — conversion must go through
  `observation_to_mcp_content`.
- `stdio` transport is the default (for Claude Desktop and similar). `sse` transport
  is for HTTP/SSE-based MCP clients.

## Invariants

1. Tool registration happens once at construction. Adding/removing actions after
   `__init__` requires a new `McpServer`.
2. Errors in action execution are surfaced to MCP clients as tool errors, not server
   crashes.

## Gotchas

- `FastMCP.run()` blocks. Embed in a subprocess or background thread if the harness
  process needs to keep running.
- `stdio` transport swallows stdout — do not `print` inside tool actions when using
  stdio. Use logging (goes to stderr) instead.
- `sse` transport exposes the server on the network — bind to `localhost` unless
  intentional.
