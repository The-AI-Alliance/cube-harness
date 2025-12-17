"""Tests for agentlab2.core module."""

import json

import pytest
from PIL import Image

from agentlab2.core import (
    Action,
    ActionSchema,
    AgentOutput,
    Content,
    EnvironmentOutput,
    Observation,
    Trajectory,
)


class TestActionSchema:
    """Tests for ActionSchema class."""

    def test_action_schema_creation(self, sample_action_schema):
        """Test basic ActionSchema creation."""
        assert sample_action_schema.name == "click"
        assert sample_action_schema.description == "Click on an element"
        assert "properties" in sample_action_schema.parameters

    def test_action_schema_default_parameters(self):
        """Test ActionSchema with default empty parameters."""
        schema = ActionSchema(name="noop", description="Do nothing")
        assert schema.parameters == {}

    def test_action_schema_from_function(self):
        """Test creating ActionSchema from a Python function."""

        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet a person.

            Args:
                name: The name of the person to greet.
                greeting: The greeting to use.

            Returns:
                The greeting message.
            """
            return f"{greeting}, {name}!"

        schema = ActionSchema.from_function(greet)
        assert schema.name == "greet"
        assert "Greet a person" in schema.description
        assert "name" in schema.parameters.get("properties", {})

    def test_action_schema_as_dict(self, sample_action_schema):
        """Test conversion to LLM API dict format."""
        result = sample_action_schema.as_dict()
        assert result["type"] == "function"
        assert result["function"]["name"] == "click"
        assert result["function"]["description"] == "Click on an element"
        assert "parameters" in result["function"]


class TestAction:
    """Tests for Action class."""

    def test_action_creation(self, sample_action):
        """Test basic Action creation."""
        assert sample_action.id == "action_1"
        assert sample_action.name == "click"
        assert sample_action.arguments == {"element_id": "button_1"}

    def test_action_without_id(self):
        """Test Action creation without id."""
        action = Action(name="scroll", arguments={"direction": "down"})
        assert action.id is None
        assert action.name == "scroll"

    def test_action_default_arguments(self):
        """Test Action with default empty arguments."""
        action = Action(name="refresh")
        assert action.arguments == {}

    def test_action_serialization(self, sample_action):
        """Test Action JSON serialization."""
        json_str = sample_action.model_dump_json()
        data = json.loads(json_str)
        assert data["name"] == "click"
        assert data["arguments"]["element_id"] == "button_1"


class TestContent:
    """Tests for Content class."""

    def test_text_content_creation(self, sample_content):
        """Test text Content creation."""
        assert sample_content.data == "Hello, world!"
        assert sample_content.name == "greeting"
        assert sample_content.tool_call_id is None

    def test_content_with_tool_call_id(self):
        """Test Content with tool_call_id."""
        content = Content(data="Result", tool_call_id="tool_1")
        assert content.tool_call_id == "tool_1"

    def test_image_content_creation(self, sample_image_content):
        """Test image Content creation."""
        assert isinstance(sample_image_content.data, Image.Image)
        assert sample_image_content.name == "screenshot"

    def test_image_serialization_roundtrip(self, sample_image_content):
        """Test image content serialization and deserialization."""
        # Serialize
        json_str = sample_image_content.model_dump_json()
        data = json.loads(json_str)

        # Verify serialized format
        assert data["data"].startswith("data:image/png;base64,")

        # Deserialize
        restored = Content.model_validate(data)
        assert isinstance(restored.data, Image.Image)
        assert restored.data.size == (100, 100)

    def test_text_content_to_message(self, sample_content):
        """Test text content conversion to LLM message."""
        msg = sample_content.to_message()
        assert msg["role"] == "user"
        assert "##greeting" in msg["content"]
        assert "Hello, world!" in msg["content"]

    def test_content_without_name_to_message(self):
        """Test content without name conversion to message."""
        content = Content(data="Simple text")
        msg = content.to_message()
        assert msg["role"] == "user"
        assert msg["content"] == "Simple text"

    def test_tool_result_content_to_message(self):
        """Test tool result content conversion to message."""
        content = Content(data="Tool output", tool_call_id="call_123")
        msg = content.to_message()
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_123"

    def test_image_content_to_message(self, sample_image_content):
        """Test image content conversion to LLM message."""
        msg = sample_image_content.to_message()
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        # First item should be text with name
        assert msg["content"][0]["type"] == "text"
        # Second should be image
        assert msg["content"][1]["type"] == "image_url"

    def test_dict_content_to_message(self):
        """Test dict content conversion to message."""
        content = Content(data={"key": "value"}, name="json_data")
        msg = content.to_message()
        assert msg["role"] == "user"
        assert "##json_data" in msg["content"]
        assert '"key"' in msg["content"]

    def test_list_content_to_message(self):
        """Test list content conversion to message."""
        content = Content(data=[1, 2, 3])
        msg = content.to_message()
        assert "[1, 2, 3]" in msg["content"]


class TestObservation:
    """Tests for Observation class."""

    def test_observation_creation(self, sample_observation):
        """Test basic Observation creation."""
        assert len(sample_observation.contents) == 1
        assert sample_observation.contents[0].data == "Task: Click the button"

    def test_observation_empty(self):
        """Test empty Observation."""
        obs = Observation()
        assert obs.contents == []

    def test_observation_from_text(self):
        """Test creating Observation from text."""
        obs = Observation.from_text("Hello, agent!")
        assert len(obs.contents) == 1
        assert obs.contents[0].data == "Hello, agent!"

    def test_observation_to_llm_messages(self):
        """Test conversion to LLM messages."""
        obs = Observation(
            contents=[Content(data="First message"), Content(data="Second message", tool_call_id="tool_1")]
        )
        messages = obs.to_llm_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "tool"

    def test_observation_multiple_contents(self):
        """Test Observation with multiple contents."""
        img = Image.new("RGB", (50, 50), color="blue")
        obs = Observation(contents=[Content(data="Instruction text"), Content(data=img, name="image")])
        messages = obs.to_llm_messages()
        assert len(messages) == 2


class TestEnvironmentOutput:
    """Tests for EnvironmentOutput class."""

    def test_env_output_creation(self, sample_env_output):
        """Test basic EnvironmentOutput creation."""
        assert sample_env_output.reward == 0.5
        assert sample_env_output.done is False
        assert sample_env_output.info == {"step": 1}

    def test_env_output_defaults(self):
        """Test EnvironmentOutput with default values."""
        obs = Observation.from_text("test")
        env_out = EnvironmentOutput(obs=obs)
        assert env_out.reward == 0.0
        assert env_out.done is False
        assert env_out.info == {}

    def test_env_output_done_state(self):
        """Test EnvironmentOutput with done=True."""
        obs = Observation.from_text("Task completed")
        env_out = EnvironmentOutput(obs=obs, reward=1.0, done=True)
        assert env_out.done is True
        assert env_out.reward == 1.0

    def test_env_output_serialization(self, sample_env_output):
        """Test EnvironmentOutput JSON serialization."""
        json_str = sample_env_output.model_dump_json()
        data = json.loads(json_str)
        assert data["reward"] == 0.5
        assert data["done"] is False


class TestAgentOutput:
    """Tests for AgentOutput class."""

    def test_agent_output_creation(self, sample_agent_output):
        """Test basic AgentOutput creation."""
        assert len(sample_agent_output.actions) == 1
        assert sample_agent_output.actions[0].name == "click"

    def test_agent_output_defaults(self):
        """Test AgentOutput with default values."""
        output = AgentOutput()
        assert output.actions == []
        assert output.llm_calls == []

    def test_agent_output_multiple_actions(self):
        """Test AgentOutput with multiple actions."""
        actions = [
            Action(name="click", arguments={"element_id": "btn1"}),
            Action(name="type_text", arguments={"text": "hello"}),
        ]
        output = AgentOutput(actions=actions)
        assert len(output.actions) == 2

    def test_agent_output_serialization(self, sample_agent_output):
        """Test AgentOutput JSON serialization."""
        json_str = sample_agent_output.model_dump_json()
        data = json.loads(json_str)
        assert len(data["actions"]) == 1
        assert data["actions"][0]["name"] == "click"


class TestTrajectory:
    """Tests for Trajectory class."""

    def test_trajectory_creation(self, sample_trajectory):
        """Test basic Trajectory creation."""
        assert sample_trajectory.metadata == {"task_id": "test_task"}
        assert len(sample_trajectory.steps) == 2

    def test_trajectory_empty(self):
        """Test empty Trajectory."""
        traj = Trajectory()
        assert traj.steps == []
        assert traj.metadata == {}

    def test_trajectory_append(self):
        """Test appending steps to Trajectory."""
        traj = Trajectory()
        obs = Observation.from_text("test")
        env_out = EnvironmentOutput(obs=obs, reward=0.5)
        traj.append(env_out)
        assert len(traj.steps) == 1
        assert traj.steps[0] == env_out

    def test_trajectory_last_env_step(self, sample_trajectory, sample_env_output):
        """Test getting last environment step."""
        last_env = sample_trajectory.last_env_step()
        assert isinstance(last_env, EnvironmentOutput)
        assert last_env.reward == sample_env_output.reward

    def test_trajectory_last_env_step_multiple(self):
        """Test last_env_step with multiple env outputs."""
        traj = Trajectory()
        obs1 = Observation.from_text("first")
        obs2 = Observation.from_text("second")
        traj.append(EnvironmentOutput(obs=obs1, reward=0.1))
        traj.append(AgentOutput(actions=[]))
        traj.append(EnvironmentOutput(obs=obs2, reward=0.9))
        last_env = traj.last_env_step()
        assert last_env.reward == 0.9

    def test_trajectory_last_env_step_no_env_output(self):
        """Test last_env_step raises error when no EnvironmentOutput exists."""
        traj = Trajectory()
        traj.append(AgentOutput(actions=[]))
        with pytest.raises(ValueError, match="No EnvironmentOutput found"):
            traj.last_env_step()

    def test_trajectory_final_reward(self):
        """Test getting final reward from trajectory."""
        traj = Trajectory()
        obs = Observation.from_text("done")
        traj.append(EnvironmentOutput(obs=obs, reward=0.5))
        traj.append(AgentOutput(actions=[]))
        traj.append(EnvironmentOutput(obs=obs, reward=1.0, done=True))
        traj.append(AgentOutput(actions=[]))
        assert traj.last_env_step().reward == 1.0

    def test_trajectory_serialization(self, sample_trajectory):
        """Test Trajectory JSON serialization."""
        json_str = sample_trajectory.model_dump_json(serialize_as_any=True)
        data = json.loads(json_str)
        assert data["metadata"]["task_id"] == "test_task"
        assert len(data["steps"]) == 2
