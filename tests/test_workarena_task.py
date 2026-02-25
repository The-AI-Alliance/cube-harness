"""Unit tests for WorkArena task integration with direct BrowsergymTool lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock

from agentlab2.benchmarks.workarena.task import WorkArenaTask
from agentlab2.core import Observation
from agentlab2.tools.browsergym import BrowsergymConfig
from agentlab2.tools.chat import ChatTool


class DummyWorkArenaTask:
    """Simple in-memory WorkArena-like task for lifecycle tests."""

    viewport = {"width": 1111, "height": 777}
    slow_mo = 123
    timeout = 4567
    locale = "en-US"
    timezone_id = "UTC"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.validate_calls = 0
        self.teardown_calls = 0
        self.last_chat_messages: list = []

    @classmethod
    def get_task_id(cls) -> str:
        return "dummy-workarena-task"

    def setup(self, page: MagicMock) -> tuple[str, dict]:
        _ = page
        return "Complete the dummy task", {"setup_info": "ok"}

    def validate(self, page: MagicMock, chat_messages: list) -> tuple[float, bool, str, dict]:
        _ = page
        self.last_chat_messages = list(chat_messages)
        self.validate_calls += 1
        return 1.0, True, "", {"task_info_key": "value"}

    def teardown(self) -> None:
        self.teardown_calls += 1


def _make_tool(config: BrowsergymConfig | None = None) -> MagicMock:
    tool = MagicMock()
    tool.config = config or BrowsergymConfig()
    tool.page = MagicMock()
    tool.page.url = "https://example.org"
    tool.page.title.return_value = "Example"
    tool.page_obs.return_value = Observation.from_text("page observation")
    return tool


def test_setup_initializes_task_and_observation() -> None:
    workarena_task = WorkArenaTask(
        id="dummy-task",
        workarena_task_class=DummyWorkArenaTask,
        seed=42,
        level="l1",
    )
    tool = _make_tool()

    obs, info = workarena_task.setup(tool)

    assert workarena_task._workarena_task is not None
    assert isinstance(obs, Observation)
    assert len(obs.contents) == 2
    assert info["task_id"] == "dummy-task"
    assert info["setup_info"] == "ok"
    tool.reset.assert_called_once()
    assert tool.config.viewport == DummyWorkArenaTask.viewport
    assert tool.config.timeout == DummyWorkArenaTask.timeout


def test_setup_respects_explicit_tool_overrides() -> None:
    workarena_task = WorkArenaTask(
        id="dummy-task",
        workarena_task_class=DummyWorkArenaTask,
        seed=42,
        level="l1",
    )
    tool = _make_tool(
        BrowsergymConfig(
            viewport={"width": 2000, "height": 1000},
            timeout=9999,
            locale="fr-FR",
        )
    )

    workarena_task.setup(tool)

    assert tool.config.viewport == {"width": 2000, "height": 1000}
    assert tool.config.timeout == 9999
    assert tool.config.locale == "fr-FR"


def test_finished_caches_validation_for_validate_task() -> None:
    workarena_task = WorkArenaTask(
        id="dummy-task",
        workarena_task_class=DummyWorkArenaTask,
        seed=42,
        level="l1",
    )
    tool = _make_tool()
    workarena_task.setup(tool)

    done = workarena_task.finished()
    reward, info = workarena_task.validate_task(Observation())

    assert done is True
    assert reward == 1.0
    assert info["done"] is True
    assert info["task_info_key"] == "value"
    assert workarena_task._workarena_task is not None
    assert workarena_task._workarena_task.validate_calls == 1


def test_validate_task_without_finished_calls_validation() -> None:
    workarena_task = WorkArenaTask(
        id="dummy-task",
        workarena_task_class=DummyWorkArenaTask,
        seed=42,
        level="l1",
    )
    tool = _make_tool()
    workarena_task.setup(tool)

    reward, info = workarena_task.validate_task(Observation())

    assert reward == 1.0
    assert info["done"] is True
    assert workarena_task._workarena_task is not None
    assert workarena_task._workarena_task.validate_calls == 1


def test_validate_task_passes_chat_messages_to_workarena_validate() -> None:
    workarena_task = WorkArenaTask(
        id="dummy-task",
        workarena_task_class=DummyWorkArenaTask,
        seed=42,
        level="l1",
    )
    tool = _make_tool()
    workarena_task.setup(tool)
    workarena_task._chat_tool = ChatTool()
    workarena_task._chat_tool.send_message("message one")
    workarena_task._chat_tool.send_message("message two")

    reward, info = workarena_task.validate_task(Observation())

    assert reward == 1.0
    assert info["done"] is True
    assert workarena_task._workarena_task is not None
    assert workarena_task._workarena_task.last_chat_messages == ["message one", "message two"]


def test_teardown_calls_underlying_task_teardown() -> None:
    workarena_task = WorkArenaTask(
        id="dummy-task",
        workarena_task_class=DummyWorkArenaTask,
        seed=42,
        level="l1",
    )
    tool = _make_tool()
    workarena_task.setup(tool)
    underlying = workarena_task._workarena_task

    workarena_task.teardown()

    assert underlying is not None
    assert underlying.teardown_calls == 1
    assert workarena_task._workarena_task is None
