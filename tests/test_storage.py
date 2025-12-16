"""Tests for agentlab2.storage module."""

import json
from pathlib import Path

import pytest
from PIL import Image

from agentlab2.core import (
    Action,
    AgentOutput,
    Content,
    EnvironmentOutput,
    Observation,
    Trajectory,
    TrajectoryStep,
)
from agentlab2.llm import LLMCall, LLMConfig, Message, Prompt
from agentlab2.storage import FileStorage, LLMCallRef


class TestFileStorageBasic:
    """Basic tests for FileStorage class."""

    def test_init_creates_path(self, tmp_dir):
        """Test FileStorage initialization."""
        storage = FileStorage(tmp_dir)
        assert storage.output_dir == Path(tmp_dir)
        assert storage._current_traj_paths == {}

    def test_init_with_string_path(self, tmp_dir):
        """Test FileStorage accepts string path."""
        storage = FileStorage(str(tmp_dir))
        assert storage.output_dir == Path(tmp_dir)

    def test_save_trajectory_creates_directories(self, tmp_dir):
        """Test save_trajectory creates necessary directories."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="test_traj_1", metadata={"task_id": "task_1"})

        storage.save_trajectory(traj)

        traj_dir = Path(tmp_dir) / "trajectories"
        assert traj_dir.exists()

    def test_save_trajectory_creates_metadata_file(self, tmp_dir):
        """Test save_trajectory creates metadata JSON file."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(
            id="test_traj_1",
            metadata={"task_id": "task_1", "agent": "test_agent"},
            start_time=0.0,
            end_time=1.0,
        )

        storage.save_trajectory(traj)

        metadata_path = Path(tmp_dir) / "trajectories" / "test_traj_1.metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            data = json.load(f)
        assert data == {
            "id": "test_traj_1",
            "metadata": {"task_id": "task_1", "agent": "test_agent"},
            "start_time": 0.0,
            "end_time": 1.0,
            "reward_info": {},
        }

    def test_save_trajectory_creates_jsonl_file(self, tmp_dir):
        """Test save_trajectory creates JSONL file."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="test_traj_1")

        storage.save_trajectory(traj)

        jsonl_path = Path(tmp_dir) / "trajectories" / "test_traj_1.jsonl"
        assert jsonl_path.exists()


class TestFileStorageWithSteps:
    """Tests for FileStorage with trajectory steps."""

    def test_save_trajectory_with_env_step(self, tmp_dir, sample_env_output):
        """Test saving trajectory with environment output step."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="test_traj")
        traj.steps.append(TrajectoryStep(output=sample_env_output))

        storage.save_trajectory(traj)

        jsonl_path = Path(tmp_dir) / "trajectories" / "test_traj.jsonl"
        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 1

        step_data = json.loads(lines[0])
        assert "output" in step_data
        assert "obs" in step_data["output"]

    def test_save_trajectory_with_agent_step(self, tmp_dir, sample_agent_output):
        """Test saving trajectory with agent output step."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="test_traj")
        traj.steps.append(TrajectoryStep(output=sample_agent_output))

        storage.save_trajectory(traj)

        jsonl_path = Path(tmp_dir) / "trajectories" / "test_traj.jsonl"
        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 1

        step_data = json.loads(lines[0])
        assert "actions" in step_data["output"]

    def test_save_trajectory_with_multiple_steps(self, tmp_dir, sample_env_output, sample_agent_output):
        """Test saving trajectory with multiple steps."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="test_traj")
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        traj.steps.append(TrajectoryStep(output=sample_agent_output))
        traj.steps.append(TrajectoryStep(output=sample_env_output))

        storage.save_trajectory(traj)

        jsonl_path = Path(tmp_dir) / "trajectories" / "test_traj.jsonl"
        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_save_step_appends_to_trajectory(self, tmp_dir, sample_env_output, sample_agent_output):
        """Test save_step appends to existing trajectory."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="test_traj")
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage.save_trajectory(traj)

        # Append additional step
        storage.save_step(TrajectoryStep(output=sample_agent_output), "test_traj", 1)

        jsonl_path = Path(tmp_dir) / "trajectories" / "test_traj.jsonl"
        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_save_step_without_trajectory_raises_error(self, tmp_dir, sample_env_output):
        """Test save_step raises error if trajectory not initialized."""
        storage = FileStorage(tmp_dir)

        with pytest.raises(ValueError, match="Trajectory path not set"):
            storage.save_step(TrajectoryStep(output=sample_env_output), "unknown_traj", 0)


class TestFileStorageWithLLMCalls:
    """Tests for FileStorage LLM call extraction."""

    @pytest.fixture
    def sample_llm_call(self):
        """Create a sample LLM call for testing."""
        return LLMCall(
            id="llm_call_1",
            llm_config=LLMConfig(model_name="test-model"),
            prompt=Prompt(messages=[{"role": "user", "content": "Hello"}]),
            output=Message(role="assistant", content="Hi there!"),
        )

    def test_save_extracts_llm_calls_to_separate_files(self, tmp_dir, sample_llm_call):
        """Test that LLM calls are extracted to separate files."""
        storage = FileStorage(tmp_dir)

        agent_output = AgentOutput(
            actions=[Action(name="click", arguments={"element": "btn"})],
            llm_calls=[sample_llm_call],
        )
        traj = Trajectory(id="test_traj")
        traj.steps.append(TrajectoryStep(output=agent_output))

        storage.save_trajectory(traj)

        # Check LLM call file exists
        llm_calls_dir = Path(tmp_dir) / "llm_calls"
        assert llm_calls_dir.exists()

        llm_call_files = list(llm_calls_dir.glob("*.json"))
        assert len(llm_call_files) == 1
        assert "test_traj_step000_llm_call_1" in llm_call_files[0].name

    def test_save_stores_llm_call_reference_in_jsonl(self, tmp_dir, sample_llm_call):
        """Test that JSONL contains LLM call reference instead of full data."""
        storage = FileStorage(tmp_dir)

        agent_output = AgentOutput(
            actions=[Action(name="click", arguments={})],
            llm_calls=[sample_llm_call],
        )
        traj = Trajectory(id="test_traj")
        traj.steps.append(TrajectoryStep(output=agent_output))

        storage.save_trajectory(traj)

        jsonl_path = Path(tmp_dir) / "trajectories" / "test_traj.jsonl"
        with open(jsonl_path) as f:
            step_data = json.loads(f.readline())

        # Should only have 'id' key in llm_calls reference
        llm_calls = step_data["output"]["llm_calls"]
        assert len(llm_calls) == 1
        assert set(llm_calls[0].keys()) == {"id"}
        assert llm_calls[0]["id"] == "llm_call_1"

    def test_save_multiple_llm_calls(self, tmp_dir):
        """Test saving step with multiple LLM calls."""
        storage = FileStorage(tmp_dir)

        llm_calls = [
            LLMCall(
                id=f"call_{i}",
                llm_config=LLMConfig(model_name="test-model"),
                prompt=Prompt(messages=[{"role": "user", "content": f"Message {i}"}]),
                output=Message(role="assistant", content=f"Response {i}"),
            )
            for i in range(3)
        ]

        agent_output = AgentOutput(actions=[], llm_calls=llm_calls)
        traj = Trajectory(id="test_traj")
        traj.steps.append(TrajectoryStep(output=agent_output))

        storage.save_trajectory(traj)

        llm_calls_dir = Path(tmp_dir) / "llm_calls"
        llm_call_files = list(llm_calls_dir.glob("*.json"))
        assert len(llm_call_files) == 3


class TestFileStorageLoad:
    """Tests for FileStorage loading functionality."""

    def test_load_trajectory_basic(self, tmp_dir):
        """Test basic trajectory loading."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="test_traj", metadata={"task_id": "task_1"})
        obs = Observation.from_text("Test observation")
        traj.steps.append(TrajectoryStep(output=EnvironmentOutput(obs=obs, reward=0.5)))

        storage.save_trajectory(traj)

        # Load using new storage instance
        storage2 = FileStorage(tmp_dir)
        loaded = storage2.load_trajectory("test_traj")

        assert loaded.id == "test_traj"
        assert loaded.metadata == {"task_id": "task_1"}
        assert len(loaded.steps) == 1

    def test_load_trajectory_preserves_step_data(self, tmp_dir, sample_env_output):
        """Test that loaded trajectory preserves step data."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="test_traj")
        traj.steps.append(TrajectoryStep(output=sample_env_output, start_time=1.0, end_time=2.0))

        storage.save_trajectory(traj)

        loaded = storage.load_trajectory("test_traj")
        loaded_step = loaded.steps[0]

        assert loaded_step.start_time == 1.0
        assert loaded_step.end_time == 2.0
        assert isinstance(loaded_step.output, EnvironmentOutput)
        assert loaded_step.output.reward == sample_env_output.reward

    def test_load_trajectory_not_found(self, tmp_dir):
        """Test loading non-existent trajectory raises error."""
        storage = FileStorage(tmp_dir)

        with pytest.raises(FileNotFoundError, match="Trajectory metadata not found"):
            storage.load_trajectory("nonexistent")

    def test_load_trajectory_resolves_llm_calls(self, tmp_dir):
        """Test that loading resolves LLM call references."""
        storage = FileStorage(tmp_dir)

        llm_call = LLMCall(
            id="test_call",
            llm_config=LLMConfig(model_name="test-model"),
            prompt=Prompt(messages=[{"role": "user", "content": "Hello"}]),
            output=Message(role="assistant", content="Hi!"),
        )
        agent_output = AgentOutput(
            actions=[Action(name="test", arguments={})],
            llm_calls=[llm_call],
        )
        traj = Trajectory(id="test_traj")
        traj.steps.append(TrajectoryStep(output=agent_output))

        storage.save_trajectory(traj)

        # Load and verify LLM calls are resolved
        loaded = storage.load_trajectory("test_traj")
        loaded_output = loaded.steps[0].output

        assert isinstance(loaded_output, AgentOutput)
        assert len(loaded_output.llm_calls) == 1

        loaded_llm_call = loaded_output.llm_calls[0]
        assert loaded_llm_call.id == "test_call"
        assert loaded_llm_call.output.content == "Hi!"


class TestFileStorageLoadAll:
    """Tests for FileStorage load_all_trajectories."""

    def test_load_all_empty_directory(self, tmp_dir):
        """Test loading from empty directory returns empty list."""
        storage = FileStorage(tmp_dir)

        result = storage.load_all_trajectories()

        assert result == []

    def test_load_all_no_trajectories_dir(self, tmp_dir):
        """Test loading when trajectories dir doesn't exist."""
        storage = FileStorage(tmp_dir)

        result = storage.load_all_trajectories()

        assert result == []

    def test_load_all_single_trajectory(self, tmp_dir, sample_env_output):
        """Test loading single trajectory."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="traj_1", metadata={"task_id": "task_1"})
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage.save_trajectory(traj)

        result = storage.load_all_trajectories()

        assert len(result) == 1
        assert result[0].id == "traj_1"

    def test_load_all_multiple_trajectories(self, tmp_dir, sample_env_output):
        """Test loading multiple trajectories."""
        storage = FileStorage(tmp_dir)

        for i in range(3):
            traj = Trajectory(id=f"traj_{i}", metadata={"task_id": f"task_{i}"})
            traj.steps.append(TrajectoryStep(output=sample_env_output))
            storage.save_trajectory(traj)

        result = storage.load_all_trajectories()

        assert len(result) == 3
        ids = {t.id for t in result}
        assert ids == {"traj_0", "traj_1", "traj_2"}

    def test_load_all_with_exp_dir_parameter(self, tmp_dir, sample_env_output):
        """Test load_all_trajectories with explicit exp_dir parameter."""
        # Save to one directory
        storage1 = FileStorage(tmp_dir)
        traj = Trajectory(id="traj_1")
        traj.steps.append(TrajectoryStep(output=sample_env_output))
        storage1.save_trajectory(traj)

        # Load using different storage instance with exp_dir parameter
        storage2 = FileStorage("/some/other/path")
        result = storage2.load_all_trajectories(exp_dir=tmp_dir)

        assert len(result) == 1
        assert result[0].id == "traj_1"


class TestFileStorageWithImages:
    """Tests for FileStorage with image content."""

    def test_save_and_load_trajectory_with_image(self, tmp_dir):
        """Test saving and loading trajectory with image content."""
        storage = FileStorage(tmp_dir)

        # Create image content
        img = Image.new("RGB", (100, 100), color="blue")
        obs = Observation(contents=[Content(data=img, name="screenshot")])
        env_output = EnvironmentOutput(obs=obs, reward=0.0)

        traj = Trajectory(id="test_traj")
        traj.steps.append(TrajectoryStep(output=env_output))

        storage.save_trajectory(traj)

        # Load and verify
        loaded = storage.load_trajectory("test_traj")
        assert len(loaded.steps) == 1
        assert isinstance(loaded.steps[0].output, EnvironmentOutput)
        loaded_content = loaded.steps[0].output.obs.contents[0]

        assert isinstance(loaded_content.data, Image.Image)
        assert loaded_content.data.size == (100, 100)
        assert loaded_content.name == "screenshot"


class TestFileStorageRoundtrip:
    """End-to-end roundtrip tests for FileStorage."""

    def test_full_trajectory_roundtrip(self, tmp_dir):
        """Test complete save/load roundtrip with all features."""
        storage = FileStorage(tmp_dir)

        # Create LLM call
        llm_call = LLMCall(
            id="call_1",
            llm_config=LLMConfig(model_name="gpt-4"),
            prompt=Prompt(messages=[{"role": "user", "content": "Click the button"}]),
            output=Message(role="assistant", content="I'll click the button."),
        )

        # Create trajectory with multiple step types
        traj = Trajectory(
            id="full_test",
            metadata={"task_id": "test_task", "agent": "test_agent"},
            start_time=100.0,
            end_time=200.0,
        )

        # Env step
        obs1 = Observation.from_text("Initial state")
        traj.steps.append(
            TrajectoryStep(output=EnvironmentOutput(obs=obs1, reward=0.0), start_time=100.0, end_time=101.0)
        )

        # Agent step with LLM call
        agent_output = AgentOutput(
            actions=[Action(id="act_1", name="click", arguments={"element": "btn"})],
            llm_calls=[llm_call],
        )
        traj.steps.append(TrajectoryStep(output=agent_output, start_time=101.0, end_time=102.0))

        # Final env step
        obs2 = Observation.from_text("Task completed")
        traj.steps.append(
            TrajectoryStep(output=EnvironmentOutput(obs=obs2, reward=1.0, done=True), start_time=102.0, end_time=103.0)
        )

        # Save
        storage.save_trajectory(traj)

        # Load with fresh storage instance
        storage2 = FileStorage(tmp_dir)
        loaded = storage2.load_trajectory("full_test")

        # Verify metadata
        assert loaded.id == "full_test"
        assert loaded.metadata["task_id"] == "test_task"
        assert loaded.metadata["agent"] == "test_agent"

        # Verify steps
        assert len(loaded.steps) == 3

        # Verify env step
        step0 = loaded.steps[0]
        assert isinstance(step0.output, EnvironmentOutput)
        assert step0.start_time == 100.0

        # Verify agent step with LLM call
        step1 = loaded.steps[1]
        assert isinstance(step1.output, AgentOutput)
        assert len(step1.output.actions) == 1
        assert step1.output.actions[0].name == "click"
        assert len(step1.output.llm_calls) == 1
        assert step1.output.llm_calls[0].output.content == "I'll click the button."

        # Verify final step
        step2 = loaded.steps[2]
        assert isinstance(step2.output, EnvironmentOutput)
        assert step2.output.reward == 1.0
        assert step2.output.done is True


class TestLLMCallRef:
    """Tests for LLMCallRef model."""

    def test_llm_call_ref_creation(self):
        """Test LLMCallRef creation."""
        ref = LLMCallRef(id="test_id")
        assert ref.id == "test_id"

    def test_llm_call_ref_serialization(self):
        """Test LLMCallRef JSON serialization."""
        ref = LLMCallRef(id="test_id")
        data = json.loads(ref.model_dump_json())
        assert data == {"id": "test_id"}
