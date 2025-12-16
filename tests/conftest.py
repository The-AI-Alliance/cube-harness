"""Common fixtures for agentlab2 tests."""

import tempfile
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
    Trajectory,
)
from agentlab2.environment import Environment, EnvironmentConfig, Task, ToolboxEnv
from agentlab2.episode import Episode
from agentlab2.llm import LLMConfig, Prompt
from agentlab2.tool import Tool

# --- Core fixtures ---


@pytest.fixture
def tmp_dir():
    """Temporary directory fixture for tests that need file I/O."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


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
    traj.append(sample_env_output)
    traj.append(sample_agent_output)
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


# --- Task fixtures ---


class MockTask(Task):
    """Mock task implementation for testing."""

    id = "mock_task_1"

    def __init__(self, goal: str = "Complete the test task"):
        self.goal = goal
        self.setup_called = False
        self.teardown_called = False
        self.validate_called = False

    def setup(self, env) -> tuple[str, dict]:
        self.setup_called = True
        return self.goal, {"task_type": "mock"}

    def teardown(self, env) -> None:
        self.teardown_called = True

    def validate_task(self, env, obs: Observation) -> tuple[float, dict]:
        self.validate_called = True
        return 1.0, {"success": True}

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        return actions


@pytest.fixture
def mock_task() -> MockTask:
    """Mock task for testing."""
    return MockTask()


# --- Environment fixtures ---


class MockEnvironmentConfig(EnvironmentConfig):
    """Mock environment configuration."""

    tools: list = []

    def make(self) -> Environment:
        assert self._task is not None, "MockEnvironmentConfig requires a Task to be assigned"
        return ToolboxEnv(task=self._task, tools=self.tools)


class SerializableEnvConfig(EnvironmentConfig):
    """Environment config without tools for JSON serialization tests."""

    def make(self) -> Environment:
        assert self._task is not None, "Task must be set in EnvironmentConfig before making the environment."
        return ToolboxEnv(task=self._task, tools=[])


class SerializableBenchmark(Benchmark):
    """Simple benchmark without custom __init__ for JSON serialization tests."""

    def setup(self):
        pass

    def close(self):
        pass

    def env_configs(self) -> list[EnvironmentConfig]:
        return []


@pytest.fixture
def mock_env_config(mock_tool, mock_task) -> MockEnvironmentConfig:
    """Mock environment config with mock tool."""
    config = MockEnvironmentConfig(tools=[mock_tool])
    config._task = mock_task
    return config


# --- Agent fixtures ---


class MockAgentConfig(AgentConfig):
    """Mock agent configuration."""

    name: str = "mock_agent"

    def make(self) -> "MockAgent":
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

    def __init__(
        self, tasks_list: list[Any], env_config: EnvironmentConfig | None = None, metadata: dict | None = None
    ):
        super().__init__(env_config=env_config or MockEnvironmentConfig(), metadata=metadata or {})
        self._tasks = tasks_list

    def setup(self):
        self.setup_called = True

    def close(self):
        self.close_called = True

    def install(self):
        self.install_called = True

    def uninstall(self):
        self.uninstall_called = True

    def env_configs(self) -> list[EnvironmentConfig]:
        assert self.env_config is not None, "MockBenchmark requires an EnvironmentConfig to be set"
        return [self.env_config.model_copy(update=dict(_task=task)) for task in self._tasks]


@pytest.fixture
def mock_benchmark(mock_task, mock_env_config) -> MockBenchmark:
    """Mock benchmark with one task."""
    return MockBenchmark(tasks_list=[mock_task], env_config=mock_env_config)


# --- Episode fixtures ---


@pytest.fixture
def sample_episode(tmp_dir, mock_agent_config, mock_env_config) -> Episode:
    """Sample episode for testing."""
    return Episode(
        id=0,
        exp_name="test_exp",
        output_dir=tmp_dir,
        agent_config=mock_agent_config,
        env_config=mock_env_config,
    )
