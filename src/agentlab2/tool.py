import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, List, get_protocol_members

from agentlab2.core import Action, ActionSchema

logger = logging.getLogger(__name__)


class AbstractTool(ABC):
    """Base class for objects that can react on some actions"""

    def reset(self) -> None:
        """Optional reset the environment to its initial state."""
        pass

    @abstractmethod
    def execute_action(self, action: Action) -> Any:
        """Execute a single action and return the result."""
        pass

    @abstractmethod
    def action_set(self) -> List[ActionSchema]:
        """Returns list of actions supported by that tool."""
        pass

    def close(self) -> None:
        """Optional clean up environment resources."""
        pass


class Tool(AbstractTool):
    """
    Base class for tool that implements an action space protocol.

    :var Returns: Description
    """

    action_space: Any

    def get_action_method(self, action) -> Callable:
        if not getattr(self.action_space, action.name, None):
            raise ValueError(f"Action {action.name} is not a part of {self.action_space}.")
        if not (fn := getattr(self, action.name, None)):
            raise ValueError(f"Action {action.name} is not implemented in {self.__class__.__name__}.")
        return fn

    def execute_action(self, action: Action) -> Any:
        fn = self.get_action_method(action)
        try:
            action_result = fn(**action.arguments) or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)
        return action_result

    def action_set(self) -> List[ActionSchema]:
        """Returns list of actions supported by that environment."""
        action_names = get_protocol_members(self.action_space)
        return [ActionSchema.from_function(getattr(self, name)) for name in action_names]
