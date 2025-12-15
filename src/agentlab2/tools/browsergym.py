import logging
from typing import Any

from agentlab2.action_spaces.browser_action_space import BrowserActionSpace
from agentlab2.core import Action, ActionSchema
from agentlab2.environment import Tool

logger = logging.getLogger(__name__)


class BrowsergymTool(Tool, BrowserActionSpace):
    """
    TODO: Browsergym tool implementation.
    Implements BrowserActionSpace protocol.
    """

    def __init__(
        self,
        max_wait: int = 60,
        use_html: bool = True,
        use_axtree: bool = False,
        use_screenshot: bool = True,
        prune_html: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.max_wait = max_wait
        self._actions = {}

    def execute_action(self, action: Action) -> Any:
        fn = self._actions[action.name]
        try:
            action_result = fn(**action.arguments) or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)
        return action_result

    @property
    def actions(self) -> list[ActionSchema]:
        return [ActionSchema.from_function(fn) for fn in self._actions.values()]

    def reset(self):
        pass
