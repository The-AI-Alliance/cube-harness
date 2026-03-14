import logging

from cube.core import Action, Observation, StepError
from cube.tool import AsyncTool, Tool

from cube_harness.metrics.tracer import GEN_AI_TOOL_CALL_RESULT, tool_span

logger = logging.getLogger(__name__)


class ToolWithTelemetry(Tool):
    """AL2 Tool subclass that wraps execute_action with OpenTelemetry tracing.

    Subclasses must override _execute_action instead of execute_action so that
    the telemetry span always wraps the complete execution, including any
    subclass-specific post-processing (e.g. appending page observations).
    """

    def execute_action(self, action: Action) -> Observation | StepError:
        with tool_span(action) as span:
            result = self._execute_action(action)
            if isinstance(result, StepError):
                result_str = f"Error executing action {action.name}: {result.exception_str}"
            else:
                result_str = str(result.contents[0].data)
            span.set_attribute(GEN_AI_TOOL_CALL_RESULT, result_str)
        return result

    def _execute_action(self, action: Action) -> Observation | StepError:
        """Override this in subclasses instead of execute_action."""
        return super().execute_action(action)


class AsyncToolWithTelemetry(AsyncTool):
    """AL2 AsyncTool subclass that wraps execute_action with OpenTelemetry tracing.

    Subclasses must override _execute_action instead of execute_action so that
    the telemetry span always wraps the complete execution, including any
    subclass-specific post-processing (e.g. appending page observations).
    """

    async def execute_action(self, action: Action) -> Observation | StepError:
        with tool_span(action) as span:
            result = await self._execute_action(action)
            if isinstance(result, StepError):
                result_str = f"Error executing action {action.name}: {result.exception_str}"
            else:
                result_str = str(result.contents[0].data)
            span.set_attribute(GEN_AI_TOOL_CALL_RESULT, result_str)
        return result

    async def _execute_action(self, action: Action) -> Observation | StepError:
        """Override this in subclasses instead of execute_action."""
        return await super().execute_action(action)
