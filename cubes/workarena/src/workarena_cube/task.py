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
from pydantic import PrivateAttr
from termcolor import colored

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
    }
)


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
        self.tool.reset()
        page = getattr(self.tool, "page")
        goal, task_info = self._workarena_task.setup(page)

        logger.info(f"WorkArena page URL after setup: {page.url}")
        logger.info(f"WorkArena page title: {page.title()}")
        logger.info(f"WorkArena task class: {self._workarena_task.__class__.__name__}")
        logger.info(colored(f"WorkArena task goal: {goal}", "green"))

        self.tool.noop()
        time.sleep(self.wait_first_page_time)

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
        page = getattr(self.tool, "page")
        reward, done, _user_message, task_info = self._workarena_task.validate(page, [])
        return reward, {"done": done, **task_info}

    def finished(self, obs: Observation) -> bool:
        """Check if the task is done via WorkArena's validate()."""
        if self._workarena_task is None:
            return False
        page = getattr(self.tool, "page")
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
    config = getattr(tool, "config", None)
    if config is None:
        return
    updated_config = config.model_copy(
        update={
            "viewport": config.viewport if config.viewport is not None else getattr(workarena_task, "viewport", None),
            "slow_mo": config.slow_mo if config.slow_mo is not None else getattr(workarena_task, "slow_mo", None),
            "timeout": config.timeout if config.timeout is not None else getattr(workarena_task, "timeout", None),
            "locale": config.locale if config.locale is not None else getattr(workarena_task, "locale", None),
            "timezone_id": config.timezone_id
            if config.timezone_id is not None
            else getattr(workarena_task, "timezone_id", None),
        }
    )
    tool.config = updated_config
