"""WorkArena task wrapper for AgentLab2."""

import logging
from typing import Any, Callable

from agentlab2.action_spaces.chat_action_space import ChatActionSpace
from agentlab2.action_spaces.browser_action_space import BidBrowserActionSpace
from agentlab2.core import ActionSchema, ActionSpace, Observation, Task
from agentlab2.tools.browsergym import BrowsergymTool
from agentlab2.tools.chat import ChatTool
from agentlab2.tools.toolbox import Toolbox

logger = logging.getLogger(__name__)


class WorkArenaTask(Task):
    """AgentLab2 Task wrapper for WorkArena BrowserGym tasks."""

    validate_per_step: bool = True
    supported_actions: ActionSpace = ActionSpace(
        ChatActionSpace.send_message,
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
        BidBrowserActionSpace.browser_scroll,
        BidBrowserActionSpace.browser_dbclick,
        BidBrowserActionSpace.browser_press,
        BidBrowserActionSpace.browser_clear,
        BidBrowserActionSpace.browser_goto,
        BidBrowserActionSpace.browser_focus,
        BidBrowserActionSpace.noop,
    )
    _browser_tool: BrowsergymTool
    _chat_tool: ChatTool

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
        self._workarena_task: Any | None = None
        self._cached_reward: float = 0.0
        self._cached_done: bool = False
        self._cached_task_info: dict[str, Any] = {}
        self._validation_cached: bool = False


    def setup(self, tool: Toolbox) -> tuple[Observation, dict]:
        """Set up the WorkArena task with direct task lifecycle management."""
        self._browser_tool = tool.find_tool(BrowsergymTool)
        self._chat_tool = tool.find_tool(ChatTool)

        logger.info(f"Setting up WorkArena task {self.id} with seed {self.seed}")
        self._workarena_task = self.workarena_task_class(seed=self.seed)
        self._apply_task_runtime_preferences()

        tool.reset()
        goal, task_info = self._workarena_task.setup(self._browser_tool.page)

        logger.info(f"WorkArena page URL after setup: {self._browser_tool.page.url}")
        logger.info(f"WorkArena page title: {self._browser_tool.page.title()}")
        logger.info(f"WorkArena task class: {self._workarena_task.__class__.__name__}")

        goal_text = self._goal_to_text(goal)
        obs = Observation.from_text(goal_text)
        obs += self._browser_tool.page_obs()

        self._validation_cached = False
        self._cached_reward = 0.0
        self._cached_done = False
        self._cached_task_info = {}

        info = {
            "task_id": self.id,
            "task_class": self.workarena_task_class.__name__,
            "seed": self.seed,
            "level": self.level,
            **(task_info if isinstance(task_info, dict) else {}),
        }
        return obs, info

    def _apply_task_runtime_preferences(self) -> None:
        """Apply task runtime properties to the tool config when not explicitly set."""
        if self._workarena_task is None:
            return

        updated_config = self._browser_tool.config.model_copy(
            update={
                "viewport": self._browser_tool.config.viewport
                if self._browser_tool.config.viewport is not None
                else getattr(self._workarena_task, "viewport", None),
                "slow_mo": self._browser_tool.config.slow_mo
                if self._browser_tool.config.slow_mo is not None
                else getattr(self._workarena_task, "slow_mo", None),
                "timeout": self._browser_tool.config.timeout
                if self._browser_tool.config.timeout is not None
                else getattr(self._workarena_task, "timeout", None),
                "locale": self._browser_tool.config.locale
                if self._browser_tool.config.locale is not None
                else getattr(self._workarena_task, "locale", None),
                "timezone_id": self._browser_tool.config.timezone_id
                if self._browser_tool.config.timezone_id is not None
                else getattr(self._workarena_task, "timezone_id", None),
            }
        )
        self._browser_tool.config = updated_config

    def _goal_to_text(self, goal: Any) -> str:
        if goal is None:
            return f"Complete the WorkArena task: {self.id}"
        if isinstance(goal, str):
            return goal
        if isinstance(goal, list):
            parts: list[str] = []
            for item in goal:
                if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                return "\n".join(parts)
        return str(goal)

    def _validate_now(self) -> None:
        if self._workarena_task is None:
            raise RuntimeError("WorkArena task is not initialized. Call setup() first.")

        reward, done, _user_message, task_info = self._workarena_task.validate(self._browser_tool.page, self._chat_tool.messages)
        self._cached_reward = float(reward)
        self._cached_done = bool(done)
        self._cached_task_info = task_info if isinstance(task_info, dict) else {}
        self._validation_cached = True

    def validate_task(self, obs: Observation) -> tuple[float, dict]:
        """Validate the task and reuse cached step validation when available."""
        if not self._validation_cached:
            self._validate_now()
        self._validation_cached = False
        return self._cached_reward, {"done": self._cached_done, **self._cached_task_info}

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
        self._validate_now()
        return self._cached_done

    def teardown(self) -> None:
        """Clean up WorkArena task resources."""
        if self._workarena_task is not None:
            try:
                self._workarena_task.teardown()
            except Exception as e:
                logger.warning(f"Error during WorkArena task teardown: {e}")
            finally:
                self._workarena_task = None
