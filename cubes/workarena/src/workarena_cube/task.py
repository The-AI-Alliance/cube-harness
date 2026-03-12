"""WorkArena task implementation for the CUBE framework."""

import importlib
import logging
import time
from typing import Any

from browsergym.workarena.tasks.base import AbstractServiceNowTask
from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import ActionSchema, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import tool_action
from cube_browser_tool import PlaywrightConfig, SyncPlaywrightTool
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)

_SUPPORTED_ACTION_NAMES = frozenset(
    {
        "browser_press_key",
        "browser_type",
        "browser_click",
        "browser_drag",
        "browser_hover",
        "browser_select_option",
        "browser_mouse_click_xy",
        "browser_wait",
        "browser_back",
        "browser_forward",
        "noop",
        "workarena_cheat",
    }
)


class WorkArenaCheatTool(SyncPlaywrightTool):
    """SyncPlaywrightTool with an additional workarena_cheat action — for debug use only."""

    def __init__(self, config: PlaywrightConfig) -> None:
        super().__init__(config)
        self._workarena_task: AbstractServiceNowTask | None = None

    @tool_action
    def workarena_cheat(self) -> str:
        """Execute the WorkArena built-in cheat to solve the task automatically."""
        if self._workarena_task is None:
            return "No WorkArena task initialized — cheat unavailable."
        self._workarena_task.cheat(self._page, [])
        return "WorkArena cheat executed."


class WorkArenaCheatToolConfig(PlaywrightConfig):
    """PlaywrightConfig variant that creates a WorkArenaCheatTool."""

    def make(self, container: Any = None) -> WorkArenaCheatTool:
        return WorkArenaCheatTool(self)


class WorkArenaTask(Task):
    """CUBE Task wrapper for WorkArena ServiceNow tasks."""

    seed: int
    wait_first_page_time: float = 10.0
    validate_per_step: bool = True

    _workarena_task: AbstractServiceNowTask | None = PrivateAttr(default=None)

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        """Instantiate and set up the WorkArena task, returning the initial observation."""
        task_class = _load_task_class(self.metadata.extra_info["task_class_path"])
        self._workarena_task = task_class(seed=self.seed)
        _apply_task_runtime_preferences(self.tool, self._workarena_task)
        if isinstance(self.tool, WorkArenaCheatTool):
            self.tool._workarena_task = self._workarena_task
        self.tool.reset()
        page = self.tool.page
        goal, task_info = self._workarena_task.setup(page)

        logger.info(f"WorkArena page URL after setup: {page.url}")
        logger.info(f"WorkArena page title: {page.title()}")
        logger.info(f"WorkArena task class: {self._workarena_task.__class__.__name__}")

        self.tool.noop()
        time.sleep(self.wait_first_page_time)
        logger.info(f"WorkArena task goal: {goal}")

        obs = Observation.from_text(goal) + self.tool.page_obs()
        info = {
            "task_id": self.id,
            "task_class": task_class.__name__,
            "seed": self.seed,
            "goal": goal,
            **task_info,
        }
        return obs, info

    def evaluate(self, obs: Observation) -> tuple[float, dict[str, Any]]:
        """Score the current task state via WorkArena's validate()."""
        if self._workarena_task is None:
            raise RuntimeError("WorkArena task is not initialized. Call reset() first.")
        page = self.tool.page
        reward, done, _user_message, task_info = self._workarena_task.validate(page, [])
        return reward, {"done": done, **task_info}

    def finished(self, obs: Observation) -> bool:
        """Check if the task is done via WorkArena's validate()."""
        if self._workarena_task is None:
            return False
        page = self.tool.page
        _reward, done, _user_message, _task_info = self._workarena_task.validate(page, [])
        return done

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        """Filter to BID browser actions supported by WorkArena."""
        filtered = [a for a in actions if a.name in _SUPPORTED_ACTION_NAMES]
        logger.debug(f"Filtered {len(filtered)} out of {len(actions)} actions for WorkArena task.")
        return filtered

    def close(self) -> None:
        """Teardown the WorkArena task and close the tool."""
        if self._workarena_task is not None:
            try:
                self._workarena_task.teardown()
            except Exception as e:
                logger.warning(f"Error during WorkArena task teardown: {e}")
            finally:
                self._workarena_task = None
        super().close()


class WorkArenaTaskConfig(TaskConfig):
    """Serializable configuration for a single WorkArena task."""

    task_metadata: TaskMetadata

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> WorkArenaTask:
        _ = runtime_context, container_backend
        return WorkArenaTask(
            metadata=self.task_metadata,
            tool_config=self.tool_config,
            seed=self.seed if self.seed is not None else 42,
        )


def _load_task_class(class_path: str) -> type:
    """Reconstruct a task class from its dotted module-qualified name."""
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _apply_task_runtime_preferences(tool: Any, workarena_task: AbstractServiceNowTask) -> None:
    """Apply WorkArena task runtime defaults to the tool config when not explicitly set."""
    browser_config = tool.config.pw_kwargs
    updated_browser_config = {
        "viewport": browser_config.viewport
        if "viewport" in browser_config
        else getattr(workarena_task, "viewport", None),
        "slow_mo": browser_config.slow_mo if "slow_mo" in browser_config else getattr(workarena_task, "slow_mo", None),
        "timeout": browser_config.timeout if "timeout" in browser_config else getattr(workarena_task, "timeout", None),
        "locale": browser_config.locale if "locale" in browser_config else getattr(workarena_task, "locale", None),
        "timezone_id": browser_config.timezone_id
        if "timezone_id" in browser_config
        else getattr(workarena_task, "timezone_id", None),
    }
    tool.config.pw_kwargs.update(updated_browser_config)
