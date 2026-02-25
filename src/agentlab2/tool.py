import logging
from typing import Any, Callable, List

from typing_extensions import get_protocol_members

from agentlab2.metrics.tracer import GEN_AI_TOOL_CALL_RESULT, tool_span
from cube.core import Action, ActionSchema, Content, Observation
from cube.tool import AbstractTool

logger = logging.getLogger(__name__)


class Tool(AbstractTool):
    """
    Base class for tool that implements an action space protocol.

    :var action_space: Protocol defining the actions this tool supports
    """

    action_space: Any

    def get_action_method(self, action) -> Callable:
        if not getattr(self.action_space, action.name, None):
            raise ValueError(f"Action {action.name} is not a part of {self.action_space}.")
        if not (fn := getattr(self, action.name, None)):
            raise ValueError(f"Action {action.name} is not implemented in {self.__class__.__name__}.")
        return fn

    def execute_action(self, action: Action) -> Observation:
        fn = self.get_action_method(action)

        with tool_span(action) as span:
            try:
                action_result = fn(**action.arguments) or "Success"
            except Exception as e:
                action_result = f"Error executing action {action.name}: {e}"
                logger.exception(action_result)

            span.set_attribute(GEN_AI_TOOL_CALL_RESULT, str(action_result))

        return Observation(contents=[Content.from_data(action_result, tool_call_id=action.id)])

    @property
    def action_set(self) -> List[ActionSchema]:
        """Returns list of actions supported by that environment."""
        action_names = get_protocol_members(self.action_space)
        return [ActionSchema.from_function(getattr(self, name)) for name in action_names]
