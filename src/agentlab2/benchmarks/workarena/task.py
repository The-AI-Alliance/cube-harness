"""WorkArena task wrapper for AgentLab2."""

import logging
from typing import Any, Callable

from termcolor import colored

from agentlab2.action_spaces.browser_action_space import BidBrowserActionSpace
from agentlab2.core import ActionSchema, ActionSubset, Observation, Task
from agentlab2.tools.browsergym import BrowsergymTool

logger = logging.getLogger(__name__)


class WorkArenaTask(Task):
    """AgentLab2 Task wrapper for WorkArena BrowserGym tasks.

    This task wraps a WorkArena task class (e.g., CreateUserTask, AllMenuTask)
    and delegates setup and validation to the BrowserGym environment.

    The task calls `tool.set_gym_task()` during setup to initialize the
    BrowserGym environment with the correct WorkArena task.
    """

    validate_per_step: bool = True
    supported_actions: ActionSubset = (
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

    def setup(self, tool: BrowsergymTool) -> tuple[Observation, dict]:
        """Set up the WorkArena task by initializing the BrowserGym environment.

        This method calls `tool.set_gym_task()` to create a new BrowserGym
        environment with the WorkArena task configured. The WorkArena task
        handles its own setup (user creation, navigation, etc.) during
        the BrowserGym env.reset() call.

        Args:
            tool: The BrowsergymTool instance to configure.

        Returns:
            Tuple of (initial observation, task info dict).
        """
        self._tool = tool
        logger.info(f"Setting up WorkArena task {self.id} with seed {self.seed}")

        # Configure BrowserGym with the WorkArena task
        # Note: seed is passed separately, not in task_kwargs, because
        # BrowserGym's env.reset() passes seed to the task constructor
        try:
            tool.set_gym_task(
                task_entrypoint=self.workarena_task_class,
                seed=self.seed,
            )
        except Exception as e:
            logger.error(f"Error during set_gym_task: {e}", exc_info=True)
            raise

        # Debug: log page URL and state after setup
        if tool.page:
            logger.info(f"WorkArena page URL after setup: {tool.page.url}")
            logger.info(f"WorkArena page title: {tool.page.title()}")

        # Log the BrowserGym task state
        if tool._env and tool._env.task:
            logger.info(f"BrowserGym task class: {tool._env.task.__class__.__name__}")
            if hasattr(tool._env.task, "start_url"):
                logger.info(f"BrowserGym task start_url: {tool._env.task.start_url}")

        # Get the goal from the BrowserGym task
        goal = self._get_goal_from_env()
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
        }

        return obs, info

    def _get_goal_from_env(self) -> str:
        """Extract the goal text from the BrowserGym environment."""
        if self._tool._last_obs and "goal" in self._tool._last_obs:
            goal = self._tool._last_obs["goal"]
            if isinstance(goal, str):
                return goal
            elif isinstance(goal, list):
                # Goal might be a list of message objects
                goal_parts = []
                for item in goal:
                    if isinstance(item, dict) and "text" in item:
                        goal_parts.append(item["text"])
                    elif isinstance(item, str):
                        goal_parts.append(item)
                return "\n".join(goal_parts)
        return f"Complete the WorkArena task: {self.id}"

    def validate_task(self, obs: Observation) -> tuple[float, dict]:
        """Validate the task using BrowserGym's validation.

        The reward and termination status are obtained from the last
        BrowserGym step, which internally calls the WorkArena task's
        validate() method.

        Args:
            obs: Current observation (unused, validation uses tool state).

        Returns:
            Tuple of (reward, info dict with done status).
        """
        reward = self._tool.last_reward
        done = self._tool.last_terminated

        # Get additional task info from BrowserGym if available
        task_info = {}
        if self._tool._last_info and "task_info" in self._tool._last_info:
            task_info = self._tool._last_info["task_info"]

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
        """Check if the task is finished based on BrowserGym termination."""
        return self._tool.last_terminated

    def teardown(self) -> None:
        """Clean up after the task.

        WorkArena tasks handle their own teardown (user deletion, etc.)
        through the BrowserGym environment's close() method.
        """
        pass
