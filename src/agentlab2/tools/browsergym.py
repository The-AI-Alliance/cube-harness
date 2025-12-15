import logging

from agentlab2.action_spaces.browser_action_space import BrowserActionSpace
from agentlab2.tool import Tool

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
