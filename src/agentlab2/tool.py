import logging

from agentlab2.metrics.tracer import GEN_AI_TOOL_CALL_RESULT, tool_span
from cube.core import Action, Observation, StepError
from cube.tool import Tool

logger = logging.getLogger(__name__)


class ToolWithTelemetry(Tool):
    """AL2 Tool subclass that wraps execute_action with OpenTelemetry tracing."""

    def execute_action(self, action: Action) -> Observation | StepError:
        with tool_span(action) as span:
            result = super().execute_action(action)
            if isinstance(result, StepError):
                result_str = f"Error executing action {action.name}: {result.exception_str}"
            else:
                result_str = str(result.contents[0].data)
            span.set_attribute(GEN_AI_TOOL_CALL_RESULT, result_str)
        return result
