"""Common fixtures for agentlab2 tests."""

import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pytest
from PIL import Image

from agentlab2.agent import Agent, AgentConfig
from agentlab2.benchmark import Benchmark
from agentlab2.core import (
    Action,
    ActionSchema,
    AgentOutput,
    Content,
    EnvironmentOutput,
    Observation,
    Task,
    Trajectory,
    TrajectoryStep,
)
from agentlab2.environment import EnvConfig, Environment
from agentlab2.episode import Episode
from agentlab2.llm import LLMConfig, Prompt
from agentlab2.tool import Tool, ToolConfig

# --- Core fixtures ---


@pytest.fixture
def tmp_dir():
    """Temporary directory fixture for tests that need file I/O."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_action_schema() -> ActionSchema:
    """Sample action schema for testing."""
    return ActionSchema(
        name="click",
        description="Click on an element",
        parameters={
            "type": "object",
            "properties": {"element_id": {"type": "string", "description": "The element to click"}},
            "required": ["element_id"],
        },
    )


@pytest.fixture
def sample_action() -> Action:
    """Sample action for testing."""
    return Action(id="action_1", name="click", arguments={"element_id": "button_1"})


@pytest.fixture
def sample_content() -> Content:
    """Sample text content."""
    return Content(data="Hello, world!", name="greeting")


@pytest.fixture
def sample_image_content() -> Content:
    """Sample image content."""
    img = Image.new("RGB", (100, 100), color="red")
    return Content(data=img, name="screenshot")


@pytest.fixture
def sample_observation() -> Observation:
    """Sample observation with text content."""
    return Observation(contents=[Content(data="Task: Click the button")])


@pytest.fixture
def sample_env_output(sample_observation) -> EnvironmentOutput:
    """Sample environment output."""
    return EnvironmentOutput(obs=sample_observation, reward=0.5, done=False, info={"step": 1})


@pytest.fixture
def sample_agent_output(sample_action) -> AgentOutput:
    """Sample agent output."""
    return AgentOutput(actions=[sample_action])


@pytest.fixture
def sample_trajectory(sample_env_output, sample_agent_output) -> Trajectory:
    """Sample trajectory with steps."""
    traj = Trajectory(metadata={"task_id": "test_task"})
    traj.steps.append(TrajectoryStep(output=sample_env_output, start_time=0.0, end_time=0.1))
    traj.steps.append(TrajectoryStep(output=sample_agent_output, start_time=0.1, end_time=0.2))
    return traj


# --- LLM fixtures ---


@pytest.fixture
def sample_llm_config() -> LLMConfig:
    """Sample LLM configuration."""
    return LLMConfig(model_name="gpt-5-nano", temperature=0.7, max_tokens=4096)


@pytest.fixture
def sample_prompt() -> Prompt:
    """Sample prompt for LLM."""
    return Prompt(
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}],
        tools=[],
    )


# --- Tool fixtures ---


@runtime_checkable
class MockActionSpace(Protocol):
    """Mock action space protocol for testing."""

    def click(self, element_id: str) -> str:
        """Click on an element."""
        ...

    def type_text(self, element_id: str, text: str) -> str:
        """Type text into an element."""
        ...


class MockTool(Tool):
    """Mock tool implementation for testing."""

    action_space = MockActionSpace

    def __init__(self):
        self.click_count = 0
        self.typed_texts = []

    def click(self, element_id: str) -> str:
        """Click on an element.

        Args:
            element_id: The element to click.

        Returns:
            Click confirmation message.
        """
        self.click_count += 1
        return f"Clicked on {element_id}"

    def type_text(self, element_id: str, text: str) -> str:
        """Type text into an element.

        Args:
            element_id: The element to type into.
            text: The text to type.

        Returns:
            Type confirmation message.
        """
        self.typed_texts.append((element_id, text))
        return f"Typed '{text}' into {element_id}"

    def reset(self):
        self.click_count = 0
        self.typed_texts = []


@pytest.fixture
def mock_tool() -> MockTool:
    """Mock tool for testing."""
    return MockTool()


class MockToolConfig(ToolConfig):
    """Mock tool configuration for testing."""

    def make(self) -> MockTool:
        return MockTool()


@pytest.fixture
def mock_tool_config() -> MockToolConfig:
    """Mock tool config for testing."""
    return MockToolConfig()


# --- Task fixtures ---


class MockTask(Task):
    """Mock task implementation for testing."""

    id = "mock_task_1"

    def __init__(self, goal: str = "Complete the test task"):
        self.goal = goal
        self.setup_called = False
        self.teardown_called = False
        self.validate_called = False

    def setup(self, tool) -> tuple[Observation, dict]:
        self.setup_called = True
        return Observation.from_text(self.goal), {"task_type": "mock"}

    def teardown(self) -> None:
        self.teardown_called = True

    def validate_task(self, *args) -> tuple[float, dict]:
        self.validate_called = True
        return 1.0, {"success": True}

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        return actions


@pytest.fixture
def mock_task() -> MockTask:
    """Mock task for testing."""
    return MockTask()


# --- Benchmark fixtures ---


class SerializableBenchmark(Benchmark):
    """Simple benchmark without custom __init__ for JSON serialization tests."""

    def setup(self):
        pass

    def close(self):
        pass

    def load_tasks(self) -> list[Task]:
        return []


@pytest.fixture
def mock_env_config(mock_tool_config, mock_task) -> EnvConfig:
    """Mock environment config with mock tool."""
    return EnvConfig(task=mock_task, tool_config=mock_tool_config)


@pytest.fixture
def mock_tool_env(mock_task, mock_tool) -> Environment:
    """Mock ToolEnv for testing."""
    return Environment(task=mock_task, tool=mock_tool)


# --- Agent fixtures ---


class MockAgentConfig(AgentConfig):
    """Mock agent configuration."""

    name: str = "mock_agent"

    def make(self, *args) -> "MockAgent":
        return MockAgent(config=self)


class MockAgent(Agent):
    """Mock agent implementation for testing."""

    name = "MockAgent"
    description = "A mock agent for testing"
    input_content_types = ["text"]
    output_content_types = ["action"]

    def __init__(self, config: MockAgentConfig):
        super().__init__(config)
        self.step_count = 0
        self.actions_to_return: list[Action] = []

    def step(self, obs: Observation) -> AgentOutput:
        self.step_count += 1
        if self.actions_to_return:
            actions = self.actions_to_return
        else:
            actions = [Action(name="final_step", arguments={})]
        return AgentOutput(actions=actions)


@pytest.fixture
def mock_agent_config() -> MockAgentConfig:
    """Mock agent config for testing."""
    return MockAgentConfig()


@pytest.fixture
def mock_agent(mock_agent_config) -> MockAgent:
    """Mock agent for testing."""
    return MockAgent(config=mock_agent_config)


# --- Benchmark fixtures ---


class MockBenchmark(Benchmark):
    """Mock benchmark for testing."""

    setup_called: bool = False
    close_called: bool = False
    install_called: bool = False
    uninstall_called: bool = False

    def __init__(self, tasks_list: list[Any], tool_config: ToolConfig, metadata: dict | None = None):
        super().__init__(tool_config=tool_config, metadata=metadata or {})
        self._tasks = tasks_list

    def setup(self):
        self.setup_called = True

    def close(self):
        self.close_called = True

    def install(self):
        self.install_called = True

    def uninstall(self):
        self.uninstall_called = True

    def load_tasks(self) -> list[Task]:
        return self._tasks


@pytest.fixture
def mock_benchmark(mock_task, mock_tool_config) -> MockBenchmark:
    """Mock benchmark with one task."""
    return MockBenchmark(tasks_list=[mock_task], tool_config=mock_tool_config)


# --- Episode fixtures ---


@pytest.fixture
def mock_episode(tmp_dir, mock_agent_config, mock_env_config) -> Episode:
    """Sample episode for testing."""
    return Episode(
        id=0,
        output_dir=tmp_dir,
        agent_config=mock_agent_config,
        env_config=mock_env_config,
    )
