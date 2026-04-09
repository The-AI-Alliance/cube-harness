"""WorkArena task implementation for the CUBE framework."""

import importlib
import logging
import time
from typing import Any, Protocol, runtime_checkable

from browsergym.workarena.tasks.base import AbstractServiceNowTask
from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import ActionSchema, Observation
from cube.task import Task, TaskConfig
from cube.tool import Toolbox, tool_action
from cube.tools.browser import BrowserTool
from cube_browser_playwright import PlaywrightSession, PlaywrightSessionConfig, Viewport
from cube_browser_tool import PlaywrightConfig, SyncPlaywrightTool
from cube_chat_tool import ChatTool
from playwright.sync_api import Page
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
        "send_message",
    }
)


@runtime_checkable
class WorkarenaBrowserToolConfig(Protocol):
    """
    Protocol for browser tool configs used by WorkArenaTask — requires a `browser` attribute and a `make()` method.
    Both BrowsergymConfig and PlaywrightConfig satisfy this protocol, so WorkArenaTask can work with either.
    """

    browser: PlaywrightSessionConfig

    def make(self, container: Any = None) -> "WorkArenaBrowserTool": ...


@runtime_checkable
class WorkArenaBrowserTool(Protocol):
    """
    Protocol for browser tools used by WorkArena tasks — requires a Playwright `page` attribute.
    Both BrowsergymTool and SyncPlaywrightTool satisfy this protocol, so WorkArenaTask can work with either.
    """

    config: WorkarenaBrowserToolConfig

    @property
    def page(self) -> Page: ...

    def noop(self) -> Any: ...

    def page_obs(self) -> Observation: ...


class WorkArenaCheatTool(SyncPlaywrightTool):
    """SyncPlaywrightTool with an additional workarena_cheat action — for debug use only."""

    def __init__(self, config: PlaywrightConfig, session: PlaywrightSession) -> None:
        super().__init__(config, session)
        self._workarena_task: AbstractServiceNowTask | None = None

    @tool_action
    def workarena_cheat(self) -> str:
        """Execute the WorkArena built-in cheat to solve the task automatically."""
        if self._workarena_task is None:
            return "No WorkArena task initialized — cheat unavailable."
        self._workarena_task.cheat(self.page, [])
        return "WorkArena cheat executed."


class WorkArenaCheatToolConfig(PlaywrightConfig):
    """PlaywrightConfig variant that creates a WorkArenaCheatTool."""

    def make(self, container: Any = None) -> WorkArenaCheatTool:
        session = self.browser.make()
        return WorkArenaCheatTool(self, session)


class WorkArenaTask(Task):
    """CUBE Task wrapper for WorkArena ServiceNow tasks."""

    seed: int
    wait_first_page_time: float = 10.0
    validate_per_step: bool = True

    _workarena_task: AbstractServiceNowTask | None = PrivateAttr(default=None)
    _validate_cache: tuple[Any, ...] | None = PrivateAttr(default=None)
    _validate_cache_key: tuple[Any, ...] | None = PrivateAttr(default=None)

    @property
    def _browser_tool(self) -> WorkArenaBrowserTool:
        """Resolve the playwright tool whether the tool is direct or inside a Toolbox."""
        if isinstance(self.tool, Toolbox):
            tool = self.tool.find_tool(BrowserTool)
            if tool is None:
                raise RuntimeError("No BrowserTool found in Toolbox")
        else:
            tool = self.tool
        if not isinstance(tool, WorkArenaBrowserTool):
            raise RuntimeError(
                f"The browser tool must satisfy the WorkArenaBrowserTool protocol (e.g., BrowsergymTool or SyncPlaywrightTool), got {type(tool).__name__}"
            )
        return tool

    @property
    def _chat_tool(self) -> ChatTool | None:
        """Return the ChatTool if present in a Toolbox, else None."""
        if isinstance(self.tool, Toolbox):
            return self.tool.find_tool(ChatTool)
        return None

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        """Instantiate and set up the WorkArena task, returning the initial observation."""
        task_class = _load_task_class(self.metadata.extra_info["task_class_path"])
        self._workarena_task = task_class(seed=self.seed)
        _apply_task_runtime_preferences(self._browser_tool, self._workarena_task)
        if isinstance(self._browser_tool, WorkArenaCheatTool):
            self._browser_tool._workarena_task = self._workarena_task
        self.tool.reset()
        page = self._browser_tool.page
        goal, task_info = self._workarena_task.setup(page)

        logger.info(f"WorkArena page URL after setup: {page.url}")
        logger.info(f"WorkArena page title: {page.title()}")
        logger.info(f"WorkArena task class: {self._workarena_task.__class__.__name__}")

        self._validate_cache = None
        self._validate_cache_key = None
        self._browser_tool.noop()
        time.sleep(self.wait_first_page_time)
        logger.info(f"WorkArena task goal: {goal}")

        page_obs = self._browser_tool.page_obs()
        if self._chat_tool is not None:
            self._chat_tool.add_message("user", goal)
            obs = Observation.from_text(self._chat_tool.chat_obs()) + page_obs
        else:
            obs = Observation.from_text(goal) + page_obs
        info = {
            "task_id": self.id,
            "task_class": task_class.__name__,
            "seed": self.seed,
            "goal": goal,
            **task_info,
        }
        return obs, info

    @property
    def _chat_messages(self) -> list[dict]:
        """Return the current chat message history, or empty list if no chat tool."""
        chat = self._chat_tool
        return chat.messages if chat is not None else []

    def _validate(self) -> tuple[float, bool, str, dict]:
        """Call WorkArena's validate() with per-step caching to avoid duplicate REST calls.

        Both evaluate() and finished() are called on every step when validate_per_step=True,
        so this caches the result keyed on (chat_messages, page_url) and reuses it within
        the same step.
        """
        if self._workarena_task is None:
            raise RuntimeError("WorkArena task is not initialized. Call reset() first.")
        page = self._browser_tool.page
        chat_messages = self._chat_messages
        cache_key = (
            tuple((m.get("role"), m.get("message")) for m in chat_messages),
            page.url,
        )
        if self._validate_cache is None or self._validate_cache_key != cache_key:
            self._validate_cache = self._workarena_task.validate(page, chat_messages)
            self._validate_cache_key = cache_key
        return self._validate_cache  # type: ignore[return-value]

    def evaluate(self, obs: Observation) -> tuple[float, dict[str, Any]]:
        """Score the current task state via WorkArena's validate()."""
        reward, done, _user_message, task_info = self._validate()
        return reward, {"done": done, **task_info}

    def finished(self, obs: Observation) -> bool:
        """Check if the task is done via WorkArena's validate()."""
        if self._workarena_task is None:
            return False
        _reward, done, _user_message, _task_info = self._validate()
        return done

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        """Filter to BID browser actions supported by WorkArena."""
        supported_actions = _SUPPORTED_ACTION_NAMES
        if self._chat_tool is None:
            supported_actions = supported_actions - {"send_message"}
        filtered = [a for a in actions if a.name in supported_actions]
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

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> WorkArenaTask:
        from workarena_cube.benchmark import WorkArenaBenchmark

        _ = runtime_context, container_backend
        meta = WorkArenaBenchmark.task_metadata[self.task_id]
        return WorkArenaTask(
            metadata=meta,
            tool_config=self.tool_config,
            seed=self.seed if self.seed is not None else 42,
        )


def _load_task_class(class_path: str) -> type:
    """Reconstruct a task class from its dotted module-qualified name."""
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _apply_task_runtime_preferences(tool: WorkArenaBrowserTool, workarena_task: AbstractServiceNowTask) -> None:
    """Apply WorkArena task runtime defaults to the tool config when not explicitly set."""
    browser_config = tool.config.browser
    explicitly_set = browser_config.model_fields_set
    updates: dict[str, Any] = {}
    for field in ("slow_mo", "timeout", "locale", "timezone_id"):
        if field not in explicitly_set and getattr(workarena_task, field, None) is not None:
            updates[field] = getattr(workarena_task, field)
    if "viewport" not in explicitly_set:
        raw_vp = getattr(workarena_task, "viewport", None)
        if isinstance(raw_vp, dict):
            updates["viewport"] = Viewport(**raw_vp)
        elif isinstance(raw_vp, Viewport):
            updates["viewport"] = raw_vp
    if updates:
        tool.config.browser = browser_config.model_copy(update=updates)
