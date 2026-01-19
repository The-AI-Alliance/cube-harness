import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, List, get_protocol_members

from opentelemetry import trace
from opentelemetry.trace import SpanKind

from agentlab2.core import Action, ActionSchema, Content, Observation, TypedBaseModel

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)


class AbstractTool(ABC):
    """
    Abstract interface for objects that can react on a list of actions.
    List defined by the ActionSpace that tool inherits.
    """

    def reset(self) -> None:
        """Optional reset the environment to its initial state."""
        pass

    @abstractmethod
    def execute_action(self, action: Action) -> Any:
        """Execute a single action and return the result."""
        pass

    @property
    @abstractmethod
    def action_set(self) -> List[ActionSchema]:
        """Returns list of actions supported by that tool."""
        pass

    def close(self) -> None:
        """Optional clean up environment resources."""
        pass


class ToolConfig(TypedBaseModel, ABC):
    """Base class for tool configurations."""

    @abstractmethod
    def make(self) -> AbstractTool:
        pass


class Tool(AbstractTool):
    """
    Base class for tool that implements an action space protocol.

    :var action_space: Protocol defining the actions this tool supports
    :var trace_tool_io: If True, include arguments and results in trace spans (may contain sensitive data)
    """

    action_space: Any
    trace_tool_io: bool = False

    def get_action_method(self, action) -> Callable:
        if not getattr(self.action_space, action.name, None):
            raise ValueError(f"Action {action.name} is not a part of {self.action_space}.")
        if not (fn := getattr(self, action.name, None)):
            raise ValueError(f"Action {action.name} is not implemented in {self.__class__.__name__}.")
        return fn

    def execute_action(self, action: Action) -> Observation:
        fn = self.get_action_method(action)
        span_name = f"execute_tool {action.name}"

        with _tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
            span.set_attribute("gen_ai.tool.name", action.name)
            span.set_attribute("gen_ai.tool.call.id", action.id)
            if self.trace_tool_io:
                span.set_attribute("gen_ai.tool.call.arguments", json.dumps(action.arguments))

            try:
                action_result = fn(**action.arguments) or "Success"
            except Exception as e:
                action_result = f"Error executing action {action.name}: {e}"
                logger.exception(action_result)

            if self.trace_tool_io:
                span.set_attribute("gen_ai.tool.call.result", str(action_result))

        return Observation(contents=[Content(data=action_result, tool_call_id=action.id)])

    @property
    def action_set(self) -> List[ActionSchema]:
        """Returns list of actions supported by that environment."""
        action_names = get_protocol_members(self.action_space)
        return [ActionSchema.from_function(getattr(self, name)) for name in action_names]
