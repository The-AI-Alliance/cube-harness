import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, List

from termcolor import colored
from typing_extensions import get_protocol_members

from agentlab2.core import Action, ActionSchema, Content, Observation, TypedBaseModel
from agentlab2.metrics.tracer import GEN_AI_TOOL_CALL_RESULT, tool_span

logger = logging.getLogger(__name__)


class AbstractTool(ABC):
    """
    Abstract interface for objects that can react on a list of actions.
    List defined by the Protocol that tool inherits.
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
    """

    action_space: Any

    def get_action_method(self, action) -> Callable:
        if not getattr(self.action_space, action.name, None):
            raise ValueError(f"Action {action.name} is not a part of {self.action_space}.")
        if not (fn := getattr(self, action.name, None)):
            raise ValueError(f"Action {action.name} is not implemented in {self.__class__.__name__}.")
        return fn

    def execute_action(self, action: Action) -> Observation:
        try:
            fn = self.get_action_method(action)
        except ValueError as e:
            logger.warning(colored(str(e), "red"))
            return Observation(
                contents=[Content(data=f"Uknown tool name: {action.name}", tool_call_id=action.id)],
                has_error=True,
            )

        with tool_span(action) as span:
            try:
                action_result = fn(**action.arguments) or "Success"
            except Exception as e:
                action_result = f"Error executing action {action.name}: {e}"
                logger.exception(action_result)

            span.set_attribute(GEN_AI_TOOL_CALL_RESULT, str(action_result))

        return Observation(contents=[Content(data=action_result, tool_call_id=action.id)])

    @property
    def action_set(self) -> List[ActionSchema]:
        """Returns list of actions supported by that environment."""
        action_names = get_protocol_members(self.action_space)
        return [ActionSchema.from_function(getattr(self, name)) for name in action_names]
