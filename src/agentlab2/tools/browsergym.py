import logging
from typing import Any

from agentlab2.action_spaces.browser_action_space import BrowserActionSpace
from agentlab2.core import Action
from agentlab2.environment import Tool

logger = logging.getLogger(__name__)


class BrowsergymTool(Tool, BrowserActionSpace):
    """
    TODO: Browsergym tool implementation.
    Implements BrowserActionSpace protocol.
    """

    action_space = BrowserActionSpace

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

    def execute_action(self, action: Action) -> Any:
        if not getattr(BrowserActionSpace, action.name, None):
            raise ValueError(f"Action {action.name} is not a part of BrowserActionSpace.")
        if not (fn := getattr(self, action.name, None)):
            raise ValueError(f"Action {action.name} is not implemented in {self.__class__.__name__}.")
        try:
            action_result = fn(**action.arguments) or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)
        return action_result

    def reset(self):
        pass
