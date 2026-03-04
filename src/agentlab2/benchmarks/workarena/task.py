"""WorkArena task wrapper for AgentLab2."""

import logging
import time

from browsergym.workarena.tasks.base import AbstractServiceNowTask
from cube.core import ActionSchema, Observation
from termcolor import colored

from agentlab2.action_spaces.browser_action_space import BidBrowserActionSpace
from agentlab2.core import ActionSpace
from agentlab2.legacy import Task
from agentlab2.tools.browsergym import BrowsergymTool

logger = logging.getLogger(__name__)


class WorkArenaTask(Task):
    """AgentLab2 Task wrapper for WorkArena BrowserGym tasks."""

    validate_per_step: bool = True
    supported_actions: ActionSpace = ActionSpace(
        BidBrowserActionSpace.browser_press_key,
        BidBrowserActionSpace.browser_type,
        BidBrowserActionSpace.browser_click,
        BidBrowserActionSpace.browser_drag,
        BidBrowserActionSpace.browser_hover,
        BidBrowserActionSpace.browser_select_option,
        BidBrowserActionSpace.browser_mouse_click_xy,
        BidBrowserActionSpace.browser_wait,
        BidBrowserActionSpace.browser_back,
        BidBrowserActionSpace.browser_forward,
        BidBrowserActionSpace.noop,
    )
    _tool: BrowsergymTool

    def __init__(
        self,
        id: str,
        workarena_task_class: type[AbstractServiceNowTask],
        seed: int,
        level: str = "l1",
        wait_first_page_time: float = 10.0,
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
        self._workarena_task: AbstractServiceNowTask | None = None
        self.wait_first_page_time = wait_first_page_time

    def setup(self, tool: BrowsergymTool) -> tuple[Observation, dict]:
        """Set up the WorkArena task with direct task lifecycle management.
        Args:
            tool: The BrowsergymTool instance to configure.

        Returns:
            Tuple of (initial observation, task info dict).
        """
        self._tool = tool
        logger.info(f"Setting up WorkArena task {self.id} with seed {self.seed}")
        self._workarena_task = self.workarena_task_class(seed=self.seed)
        _apply_task_runtime_preferences(self._tool, self._workarena_task)
        self._tool.reset()
        goal, task_info = self._workarena_task.setup(self._tool.page)

        logger.info(f"WorkArena page URL after setup: {self._tool.page.url}")
        logger.info(f"WorkArena page title: {self._tool.page.title()}")
        logger.info(f"WorkArena task class: {self._workarena_task.__class__.__name__}")

        obs = Observation.from_text(goal)
        obs += self._tool.page_obs()

        # Get the goal from the BrowserGym task
        tool.noop()  # perform a noop to ensure env is ready
        time.sleep(self.wait_first_page_time)  # wait for page to load
        logger.info(colored(f"WorkArena task goal: {goal}", "green"))

        # Build initial observation with goal and page state
        obs = Observation.from_text(goal)
        obs += tool.page_obs()

        info = {
            "task_id": self.id,
            "task_class": self.workarena_task_class.__name__,
            "seed": self.seed,
            "level": self.level,
            "goal": goal,
            **task_info,
        }
        return obs, info

    def validate_task(self, obs: Observation) -> tuple[float, dict]:
        """Validate the task and return the reward and done status.

        Args:
            obs: Current observation (unused, validation uses tool state).

        Returns:
            Tuple of (reward, info dict with done status).
        """
        if self._workarena_task is None:
            raise RuntimeError("WorkArena task is not initialized. Call setup() first.")

        reward, done, _user_message, task_info = self._workarena_task.validate(self._tool.page, [])
        return reward, {"done": done, **task_info}

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        """Filter actions to those supported by WorkArena tasks.

        Args:
            actions: List of all available actions.

        Returns:
            Filtered list of actions supported for WorkArena.
        """
        supported_action_names = {action.__name__ for action in self.supported_actions}
        filtered = [a for a in actions if a.name in supported_action_names]
        logger.debug(f"Filtered {len(filtered)} out of {len(actions)} actions for WorkArena task.")
        return filtered

    def finished(self) -> bool:
        """Check task completion via WorkArena validate() and cache the result."""
        _reward, done, _user_message, _task_info = self._workarena_task.validate(self._tool.page, [])
        return done or self._tool._last_terminated

    def teardown(self) -> None:
        """Clean up WorkArena task resources."""
        if self._workarena_task is not None:
            try:
                self._workarena_task.teardown()
            except Exception as e:
                logger.warning(f"Error during WorkArena task teardown: {e}")
            finally:
                self._workarena_task = None


def _apply_task_runtime_preferences(tool: BrowsergymTool, workarena_task: AbstractServiceNowTask) -> None:
    """Apply task runtime properties to the tool config when not explicitly set."""
    updated_config = tool.config.model_copy(
        update={
            "viewport": tool.config.viewport
            if tool.config.viewport is not None
            else getattr(workarena_task, "viewport", None),
            "slow_mo": tool.config.slow_mo
            if tool.config.slow_mo is not None
            else getattr(workarena_task, "slow_mo", None),
            "timeout": tool.config.timeout
            if tool.config.timeout is not None
            else getattr(workarena_task, "timeout", None),
            "locale": tool.config.locale if tool.config.locale is not None else getattr(workarena_task, "locale", None),
            "timezone_id": tool.config.timezone_id
            if tool.config.timezone_id is not None
            else getattr(workarena_task, "timezone_id", None),
        }
    )
    tool.config = updated_config
