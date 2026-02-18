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
from agentlab2.storage import FileStorage


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
            "_type": "agentlab2.core.Trajectory",
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


class TestFileStorageLogs:
    """Tests for per-episode log helpers in FileStorage."""

    def test_get_log_path(self, tmp_dir: Path) -> None:
        """Test that log path uses logs/{trajectory_id}.log convention."""
        storage = FileStorage(tmp_dir)

        log_path = storage.get_log_path("task_a_ep3")

        assert log_path == Path(tmp_dir) / "logs" / "task_a_ep3.log"

    def test_load_logs_returns_full_file_contents(self, tmp_dir: Path) -> None:
        """Test that load_logs returns complete log file content."""
        storage = FileStorage(tmp_dir)
        log_path = storage.get_log_path("task_b_ep1")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("line 1\nline 2\nline 3\n")

        loaded = storage.load_logs("task_b_ep1")

        assert loaded == "line 1\nline 2\nline 3\n"
        assert storage.has_logs("task_b_ep1") is True

    def test_load_logs_missing_file(self, tmp_dir: Path) -> None:
        """Test missing log file handling."""
        storage = FileStorage(tmp_dir)

        loaded = storage.load_logs("missing_ep0")

        assert loaded == ""
        assert storage.has_logs("missing_ep0") is False


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
        assert "llm_call_id" in llm_calls[0]
        assert llm_calls[0]["llm_call_id"] == "llm_call_1"

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


class TestFileStorageEpisodeConfig:
    """Tests for FileStorage episode config save/load functionality."""

    def test_save_episode_config_creates_directory(self, tmp_dir, mock_agent_config, mock_tool_config):
        """Test save_episode_config creates episode_configs directory."""
        from agentlab2.episode import EpisodeConfig

        storage = FileStorage(tmp_dir)
        episode_config = EpisodeConfig(
            id=0,
            task_id="test_task",
            agent_config=mock_agent_config,
            tool_config=mock_tool_config,
            exp_name="test_exp",
            output_dir=tmp_dir,
            max_steps=100,
        )

        storage.save_episode_config(episode_config)

        config_dir = Path(tmp_dir) / "episode_configs"
        assert config_dir.exists()

    def test_save_episode_config_creates_file(self, tmp_dir, mock_agent_config, mock_tool_config):
        """Test save_episode_config creates correct config file."""
        from agentlab2.episode import EpisodeConfig

        storage = FileStorage(tmp_dir)
        episode_config = EpisodeConfig(
            id=5,
            task_id="my_task_123",
            agent_config=mock_agent_config,
            tool_config=mock_tool_config,
            exp_name="test_exp",
            output_dir=tmp_dir,
            max_steps=200,
        )

        storage.save_episode_config(episode_config)

        config_path = Path(tmp_dir) / "episode_configs" / "episode_5_task_my_task_123.json"
        assert config_path.exists()

    def test_load_episode_config_roundtrip(self, tmp_dir, mock_agent_config, mock_tool_config):
        """Test episode config save/load round-trip."""
        from agentlab2.episode import EpisodeConfig

        storage = FileStorage(tmp_dir)

        # Create and save config
        original_config = EpisodeConfig(
            id=42,
            task_id="roundtrip_task",
            agent_config=mock_agent_config,
            tool_config=mock_tool_config,
            exp_name="roundtrip_exp",
            output_dir=tmp_dir,
            max_steps=500,
        )

        storage.save_episode_config(original_config)

        # Load config
        config_path = Path(tmp_dir) / "episode_configs" / "episode_42_task_roundtrip_task.json"
        loaded_config = storage.load_episode_config(config_path)

        # Verify all fields match
        assert loaded_config.id == original_config.id
        assert loaded_config.task_id == original_config.task_id
        assert loaded_config.exp_name == original_config.exp_name
        assert loaded_config.max_steps == original_config.max_steps
        assert loaded_config.output_dir == original_config.output_dir
        assert loaded_config.agent_config == original_config.agent_config
        assert loaded_config.tool_config == original_config.tool_config

    def test_load_episode_config_not_found(self, tmp_dir):
        """Test load_episode_config raises error for non-existent file."""
        storage = FileStorage(tmp_dir)
        config_path = Path(tmp_dir) / "episode_configs" / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            storage.load_episode_config(config_path)

    def test_list_episode_configs(self, tmp_dir, mock_agent_config, mock_tool_config):
        """Test list_episode_configs returns all config files."""
        from agentlab2.episode import EpisodeConfig

        storage = FileStorage(tmp_dir)

        # Save multiple configs
        for i in range(3):
            config = EpisodeConfig(
                id=i,
                task_id=f"task_{i}",
                agent_config=mock_agent_config,
                tool_config=mock_tool_config,
                exp_name="test_exp",
                output_dir=tmp_dir,
                max_steps=100,
            )
            storage.save_episode_config(config)

        # List configs
        config_files = storage.list_episode_configs()

        assert len(config_files) == 3
        # Verify all files exist and have correct naming pattern
        for config_file in config_files:
            assert config_file.exists()
            assert config_file.name.startswith("episode_")
            assert "_task_" in config_file.name
            assert config_file.name.endswith(".json")

    def test_list_episode_configs_empty_directory(self, tmp_dir):
        """Test list_episode_configs returns empty list when no configs exist."""
        storage = FileStorage(tmp_dir)
        config_files = storage.list_episode_configs()

        assert config_files == []

    def test_episode_config_filename_parsing(self, tmp_dir, mock_agent_config, mock_tool_config):
        """Test episode config filename format is correct for parsing."""
        from agentlab2.episode import EpisodeConfig

        storage = FileStorage(tmp_dir)

        # Save config with task_id that contains underscores
        config = EpisodeConfig(
            id=10,
            task_id="task_with_underscores_123",
            agent_config=mock_agent_config,
            tool_config=mock_tool_config,
            exp_name="test_exp",
            output_dir=tmp_dir,
            max_steps=100,
        )

        storage.save_episode_config(config)

        # Verify filename format
        config_path = Path(tmp_dir) / "episode_configs" / "episode_10_task_task_with_underscores_123.json"
        assert config_path.exists()

        # Load it back
        loaded = storage.load_episode_config(config_path)
        assert loaded.id == 10
        assert loaded.task_id == "task_with_underscores_123"


class TestFileStorageOverwrite:
    """Tests for save_trajectory overwrite / archive behavior."""

    def test_save_trajectory_raises_on_duplicate(self, tmp_dir) -> None:
        """Saving a trajectory with the same ID twice (different session) raises FileExistsError."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="dup_test", metadata={"task_id": "t1"})
        storage.save_trajectory(traj)

        # New storage instance (simulates a new session)
        storage2 = FileStorage(tmp_dir)
        with pytest.raises(FileExistsError, match="dup_test"):
            storage2.save_trajectory(traj)

    def test_save_trajectory_allows_resave_same_session(self, tmp_dir) -> None:
        """Re-saving within the same session (e.g. end_time update) succeeds without allow_overwrite."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="resave_test", metadata={"task_id": "t1"})
        storage.save_trajectory(traj)
        # Second save in same session — should not raise
        traj.end_time = 999.0
        storage.save_trajectory(traj)

    def test_save_trajectory_archives_on_overwrite(self, tmp_dir) -> None:
        """With allow_overwrite=True, old trajectory files are archived before saving."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="archive_test", metadata={"task_id": "t1"})
        obs = Observation.from_text("old data")
        env_out = EnvironmentOutput(obs=obs, reward=0.5)
        traj.steps.append(TrajectoryStep(output=env_out))
        storage.save_trajectory(traj)

        traj_dir = Path(tmp_dir) / "trajectories"

        # New storage instance with allow_overwrite
        storage2 = FileStorage(tmp_dir)
        traj2 = Trajectory(id="archive_test", metadata={"task_id": "t1"})
        storage2.save_trajectory(traj2, allow_overwrite=True)

        # Archived files should exist
        archived_metadata = list(traj_dir.glob("archive_test.archived_*.metadata.json"))
        archived_jsonl = list(traj_dir.glob("archive_test.archived_*.jsonl"))
        assert len(archived_metadata) == 1
        assert len(archived_jsonl) == 1

        # New files should also exist
        assert (traj_dir / "archive_test.metadata.json").exists()
        assert (traj_dir / "archive_test.jsonl").exists()

        # Archived JSONL should contain the old step data
        with open(archived_jsonl[0]) as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert "0.5" in lines[0]  # old reward

        # New JSONL should be empty (no steps in traj2)
        with open(traj_dir / "archive_test.jsonl") as f:
            assert f.read() == ""

    def test_save_trajectory_overwrite_false_is_default(self, tmp_dir) -> None:
        """allow_overwrite defaults to False."""
        storage = FileStorage(tmp_dir)
        traj = Trajectory(id="default_test", metadata={"task_id": "t1"})
        storage.save_trajectory(traj)

        storage2 = FileStorage(tmp_dir)
        # Should raise without explicit allow_overwrite=True
        with pytest.raises(FileExistsError):
            storage2.save_trajectory(traj)
