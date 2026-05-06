# Tool (Telemetry Wrapper)

**Module:** `cube_harness.tool`

## Purpose

Thin telemetry wrapper over cube-standard's `Tool` and `AsyncTool`. Wraps every
`execute_action` call in an OpenTelemetry span so tool invocations appear in the
episode trace without modifying the core protocol.

## Public API

### `ToolWithTelemetry(Tool)`
```python
class ToolWithTelemetry(Tool):
    def execute_action(self, action: Action) -> Observation | StepError
    # Wraps _execute_action in a tool_span. DO NOT override execute_action.

    def _execute_action(self, action: Action) -> Observation | StepError
    # OVERRIDE THIS. Default delegates to Tool.execute_action (the @tool_action dispatch).
```

### `AsyncToolWithTelemetry(AsyncTool)`
Same pattern, but both methods are coroutines.

## Contract

- **Subclasses must override `_execute_action`**, not `execute_action`. The span
  must wrap the **complete** execution including any subclass post-processing
  (e.g., appending page observations).
- The span records `GEN_AI_TOOL_CALL_RESULT` with a string form of the result.
  `StepError` is recorded as `"Error executing action <name>: <exception_str>"`.

## Relation to cube-standard

- `cube.tool.Tool` provides `@tool_action` dispatch.
- `ToolWithTelemetry` adds an OpenTelemetry span around that dispatch.
- Any tool used inside the harness should subclass `ToolWithTelemetry` / `AsyncToolWithTelemetry`
  rather than the base `cube.tool.Tool` directly, so traces include tool spans.

## Gotchas

- Overriding `execute_action` on a `ToolWithTelemetry` subclass silently strips
  telemetry. Always override `_execute_action`.
- `result.contents[0].data` is coerced to `str` for the span attribute — large
  structured results (e.g., HTML dumps) become very long attribute strings.
  Acceptable today; a truncation policy may land later.
