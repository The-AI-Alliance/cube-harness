"""WorkArena task wrapper for AgentLab2."""

import logging
from typing import Any, Callable

from termcolor import colored

from agentlab2.action_spaces.browser_action_space import BrowserActionSpace
from agentlab2.core import ActionSchema, ActionSpace, Observation, Task
from agentlab2.tools.base import BrowserTaskTool

logger = logging.getLogger(__name__)


class WorkArenaTask(Task):
    """AgentLab2 Task wrapper for WorkArena tasks.

    This task wraps a WorkArena task class (e.g., CreateUserTask, AllMenuTask)
    and calls the WorkArena task API directly via the Playwright page, without
    routing through BrowserGym's BrowserEnv.

    WorkArena tasks expose a direct Playwright API:
        - task.setup(page) -> (goal, info)
        - task.validate(page, chat_messages) -> (reward, done, msg, info)
        - task.teardown()
    """

    validate_per_step: bool = True
    supported_actions = ActionSpace(
        BrowserActionSpace.browser_press_key,
        BrowserActionSpace.browser_type,
        BrowserActionSpace.browser_click,
        BrowserActionSpace.browser_drag,
        BrowserActionSpace.browser_hover,
        BrowserActionSpace.browser_select_option,
        BrowserActionSpace.browser_mouse_click_xy,
        BrowserActionSpace.browser_wait,
        BrowserActionSpace.browser_back,
        BrowserActionSpace.browser_forward,
        BrowserActionSpace.noop,
    )
    _tool: BrowserTaskTool
    _wa_task: Any

    def __init__(
        self,
        id: str,
        workarena_task_class: Callable[..., Any],
        seed: int,
        level: str = "l1",
    ) -> None:
        """Initialize a WorkArena task wrapper.

        Args:
            id: Unique task identifier (e.g., "workarena.servicenow.create-user").
            workarena_task_class: The WorkArena task class to instantiate.
            seed: Random seed for task generation.
            level: Task level ("l1", "l2", or "l3").
        """
        self.id = id
        self.workarena_task_class = workarena_task_class
        self.seed = seed
        self.level = level
        self._wa_task = None

    def setup(self, tool: BrowserTaskTool) -> tuple[Observation, dict]:
        """Set up the WorkArena task by calling the task's setup() directly.

        Instantiates the WorkArena task and calls its setup() with the
        underlying Playwright page. No BrowserGym BrowserEnv is involved.

        Args:
            tool: A BrowserTaskTool instance (e.g., SyncPlaywrightTool).

        Returns:
            Tuple of (initial observation, task info dict).
        """
        self._tool = tool
        logger.info(f"Setting up WorkArena task {self.id} with seed {self.seed}")

        self._wa_task = self.workarena_task_class(seed=self.seed)

        try:
            goal, task_info = self._wa_task.setup(page=tool.page)
        except Exception as e:
            logger.error(f"Error during WorkArena task setup: {e}", exc_info=True)
            raise

        logger.info(f"WorkArena page URL after setup: {tool.page.url}")
        logger.info(colored(f"WorkArena task goal: {goal}", "green"))

        obs = Observation.from_text(goal) + tool.page_obs()

        info = {
            "task_id": self.id,
            "task_class": self.workarena_task_class.__name__,
            "seed": self.seed,
            "level": self.level,
            **task_info,
        }

        return obs, info

    def validate_task(self, obs: Observation) -> tuple[float, dict]:
        """Validate the task by calling the WorkArena task's validate() directly.

        Args:
            obs: Current observation (unused, validation uses the page directly).

        Returns:
            Tuple of (reward, info dict with done status).
        """
        reward, done, _msg, task_info = self._wa_task.validate(
            page=self._tool.page, chat_messages=[]
        )
        return reward, {"done": done, **task_info}

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        """Filter actions to those supported by WorkArena tasks.

        Args:
            actions: List of all available actions.

        Returns:
            Filtered list of actions supported for WorkArena.
        """
        filtered = [a for a in actions if a.name in self.supported_actions.names]
        logger.debug(f"Filtered {len(filtered)} out of {len(actions)} actions for WorkArena task.")
        return filtered

    def teardown(self) -> None:
        """Clean up the WorkArena task.

        Calls the WorkArena task's teardown() to clean up resources
        (user deletion, data cleanup, etc.).
        """
        if self._wa_task is not None:
            self._wa_task.teardown()
