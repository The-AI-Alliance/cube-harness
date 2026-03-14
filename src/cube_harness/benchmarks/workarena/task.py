"""WorkArena task wrapper for cube-harness."""

import logging
import time

from browsergym.workarena.tasks.base import AbstractServiceNowTask
from cube.core import ActionSchema, Observation
from cube.tool import BrowserTool

from cube_harness.action_spaces.browser_action_space import BidBrowserActionSpace
from cube_harness.core import ActionSpace
from cube_harness.legacy import Task

logger = logging.getLogger(__name__)


class WorkArenaTask(Task):
    """cube-harness Task wrapper for WorkArena BrowserGym tasks.

    This task wraps a WorkArena task class (e.g., CreateUserTask, AllMenuTask)
    and delegates setup and validation to the WorkArena task directly via the
    Playwright page exposed by the tool.

    The task calls the WorkArena task's setup(page) during setup to initialize
    the ServiceNow environment (user creation, navigation, etc.).
    """

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
    _tool: BrowserTool

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
            wait_first_page_time: Seconds to wait after setup for the page to fully load.
        """
        self.id = id
        self.workarena_task_class = workarena_task_class
        self.seed = seed
        self.level = level
        self._workarena_task: AbstractServiceNowTask | None = None
        self.wait_first_page_time = wait_first_page_time

    def setup(self, tool: BrowserTool) -> tuple[Observation, dict]:
        """Set up the WorkArena task.

        Applies task-specific browser preferences, resets the browser, then
        calls the WorkArena task's own setup() with the live Playwright page.

        Args:
            tool: Browser tool exposing reset(), noop(), page_obs(), and page.

        Returns:
            Tuple of (initial observation, task info dict).
        """
        self._tool = tool
        logger.info(f"Setting up WorkArena task {self.id} with seed {self.seed}")
        self._workarena_task = self.workarena_task_class(seed=self.seed)
        _apply_task_runtime_preferences(self._tool, self._workarena_task)
        self._tool.reset()

        page, _ = self._tool.session.get_playwright_session()
        goal, task_info = self._workarena_task.setup(page)

        logger.info(f"WorkArena page URL after setup: {page.url}")
        logger.info(f"WorkArena page title: {page.title()}")
        logger.info(f"WorkArena task class: {self._workarena_task.__class__.__name__}")

        self._tool.noop()  # sync env state before first observation
        time.sleep(self.wait_first_page_time)
        logger.info(f"WorkArena task goal: {goal}")

        obs = Observation.from_text(goal)
        obs += self._tool.page_obs()

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
        page, _ = self._tool.session.get_playwright_session()
        reward, done, _user_message, task_info = self._workarena_task.validate(page, [])
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
        """Check task completion via WorkArena validate()."""
        if self._workarena_task is None:
            raise RuntimeError("WorkArena task is not initialized. Call setup() first.")
        page, _ = self._tool.session.get_playwright_session()
        _, done, _, _ = self._workarena_task.validate(page, [])
        return done

    def teardown(self) -> None:
        """Clean up WorkArena task resources."""
        if self._workarena_task is not None:
            try:
                self._workarena_task.teardown()
            except Exception as e:
                logger.warning(f"Error during WorkArena task teardown: {e}")
            finally:
                self._workarena_task = None


def _apply_task_runtime_preferences(tool: BrowserTool, workarena_task: AbstractServiceNowTask) -> None:
    """Apply WorkArena task browser preferences (locale, slow_mo, timeout) onto tool.config.browser_config.

    WorkArena tasks declare preferred viewport, locale, slow_mo, and timeout as class
    attributes. These are applied to the tool's BrowserSessionConfig before reset() is
    called so the browser launches with the right settings.
    """
    bc = tool.config.browser_config  # type: ignore[attr-defined]
    updated_bc = bc.model_copy(
        update={
            "viewport": bc.viewport if bc.viewport is not None else getattr(workarena_task, "viewport", None),
            "slow_mo": bc.slow_mo if bc.slow_mo is not None else getattr(workarena_task, "slow_mo", None),
            "timeout": bc.timeout if bc.timeout is not None else getattr(workarena_task, "timeout", None),
            "locale": bc.locale if bc.locale is not None else getattr(workarena_task, "locale", None),
            "timezone_id": bc.timezone_id
            if bc.timezone_id is not None
            else getattr(workarena_task, "timezone_id", None),
        }
    )
    tool.config = tool.config.model_copy(update={"browser_config": updated_bc})  # type: ignore[attr-defined]
