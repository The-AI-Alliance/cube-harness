"""Tests for agentlab2.episode module."""

import json

import pytest

from agentlab2.core import Action, AgentOutput, EnvironmentOutput, Observation, Trajectory, TrajectoryStep
from agentlab2.episode import MAX_STEPS, Episode
from tests.conftest import MockAgent


class TestEpisode:
    """Tests for Episode class."""

    def test_episode_creation(self, mock_episode, tmp_dir):
        """Test Episode creation."""
        assert mock_episode.config.id == 0
        assert mock_episode.config.output_dir == tmp_dir
        assert mock_episode.config.max_steps == MAX_STEPS

    def test_episode_custom_max_steps(self, tmp_dir, mock_agent_config, mock_env_config):
        """Test Episode with custom max_steps."""
        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            env_config=mock_env_config,
            max_steps=10,
        )

        assert episode.config.max_steps == 10

    def test_episode_run_completes(self, mock_episode):
        """Test Episode run completes successfully."""
        trajectory = mock_episode.run()

        assert isinstance(trajectory, Trajectory)
        assert "task_id" in trajectory.metadata
        # Should have initial env output + agent output + final env output
        assert len(trajectory.steps) >= 2

    def test_episode_run_saves_trajectory(self, mock_episode, tmp_dir):
        """Test Episode run saves trajectory files."""
        mock_episode.run()

        # Check trajectory files exist
        traj_dir = tmp_dir
        assert traj_dir.exists()

        # Should have metadata and jsonl files
        files = list(traj_dir.iterdir())
        assert any(".metadata.json" in f.name for f in files)
        assert any(".jsonl" in f.name for f in files)

    def test_episode_run_metadata_file_content(self, mock_episode, tmp_dir):
        """Test Episode run creates correct metadata file."""
        mock_episode.run()

        # Read metadata file
        traj_dir = tmp_dir
        metadata_files = [f for f in traj_dir.iterdir() if ".metadata.json" in f.name]
        assert len(metadata_files) > 0, "No metadata file found"

        with open(metadata_files[0]) as f:
            metadata = json.load(f)["metadata"]

        assert "task_id" in metadata

    def test_episode_run_jsonl_content(self, mock_episode, tmp_dir):
        """Test Episode run creates correct JSONL file."""
        mock_episode.run()

        # Read JSONL file
        traj_dir = tmp_dir
        jsonl_files = [f for f in traj_dir.iterdir() if ".jsonl" in f.name]
        assert len(jsonl_files) > 0, "No JSONL file found"

        with open(jsonl_files[0]) as f:
            lines = f.readlines()

        # Should have at least one step saved
        assert len(lines) >= 1

        # Each line should be valid JSON
        for line in lines:
            if line.strip():
                data = json.loads(line)
                assert isinstance(data, dict)

    def test_episode_run_respects_max_steps(self, tmp_dir, mock_agent_config, mock_env_config):
        """Test Episode run respects max_steps limit."""

        # Create an agent that never stops
        class NeverStopsAgent(MockAgent):
            def step(self, obs):
                self.step_count += 1
                # Return non-stop action
                return AgentOutput(actions=[Action(name="click", arguments={"element_id": "btn"})])

        class NeverStopsConfig(type(mock_agent_config)):
            def make(self, *args):
                agent = NeverStopsAgent(config=self)
                return agent

        config = NeverStopsConfig()

        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=config,
            env_config=mock_env_config,
            max_steps=3,
        )

        trajectory = episode.run()

        # Should have stopped at max_steps
        # Steps: initial_env + (agent + env) * max_steps
        # But it's limited by max_steps, so agent should only step 3 times
        agent_steps = sum(1 for step in trajectory.steps if isinstance(step, AgentOutput))
        assert agent_steps <= 3

    def test_episode_run_stops_on_done(self, tmp_dir, mock_agent_config, mock_env_config):
        """Test Episode run stops when done=True."""
        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            env_config=mock_env_config,
            max_steps=100,  # High limit
        )

        trajectory = episode.run()

        # Should stop before max_steps because agent returns final_step
        last_env_step = trajectory.last_env_step()
        assert last_env_step.done is True

    def test_storage_save_trajectory_creates_directory(self, mock_episode, tmp_dir):
        """Test save_trajectory creates trajectory directory."""
        trajectory = Trajectory(id="test_traj", metadata={"task_id": "test"})
        mock_episode.storage.save_trajectory(trajectory)

        traj_dir = tmp_dir
        assert traj_dir.exists()

    def test_storage_save_step_without_trajectory(self, mock_episode):
        """Test save_step raises error if called before save_trajectory."""
        obs = Observation.from_text("test")
        step = TrajectoryStep(output=EnvironmentOutput(obs=obs))

        with pytest.raises(ValueError, match="Trajectory path not set"):
            mock_episode.storage.save_step(step, "nonexistent_traj", 0)

    def test_storage_save_step_appends(self, mock_episode, tmp_dir):
        """Test save_step appends to JSONL file."""
        trajectory = Trajectory(id="test_traj", metadata={"task_id": "test"})
        mock_episode.storage.save_trajectory(trajectory)

        # Save multiple steps
        for i in range(3):
            obs = Observation.from_text(f"step {i}")
            step = TrajectoryStep(output=EnvironmentOutput(obs=obs))
            mock_episode.storage.save_step(step, trajectory.id, i)

        # Read JSONL file
        traj_dir = tmp_dir
        jsonl_files = [f for f in traj_dir.iterdir() if ".jsonl" in f.name]
        assert len(jsonl_files) > 0, "No JSONL file found"

        with open(jsonl_files[0]) as f:
            lines = f.readlines()

        assert len(lines) == 3

    def test_episode_closes_env_on_completion(self, mock_episode, mock_task):
        """Test Episode closes environment after run."""
        mock_episode.run()

        # Task teardown should have been called
        assert mock_task.teardown_called

    def test_episode_closes_env_on_error(self, tmp_dir, mock_agent_config, mock_task, mock_env_config):
        """Test Episode closes environment even when error occurs."""

        class ErrorAgent(MockAgent):
            def step(self, obs):
                raise RuntimeError("Test error")

        class ErrorConfig(type(mock_agent_config)):
            def make(self, *args) -> "ErrorAgent":
                return ErrorAgent(config=self)

        config = ErrorConfig()

        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=config,
            env_config=mock_env_config,
        )

        with pytest.raises(RuntimeError, match="Test error"):
            episode.run()

        # Environment should still be closed
        assert mock_task.teardown_called

    def test_episode_output_filename(self, tmp_dir, mock_agent_config, mock_env_config):
        """Test Episode generates correct output filename."""
        episode = Episode(
            id=42,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            env_config=mock_env_config,
        )

        episode.run()

        traj_dir = tmp_dir
        files = [f.name for f in traj_dir.iterdir()]

        # Should contain run id
        assert any("ep42" in f for f in files)

    def test_episode_captures_agent_error(self, tmp_dir, mock_agent_config, mock_env_config):
        """Test Episode captures agent errors correctly in trajectory."""

        class ErrorAgent(MockAgent):
            def step(self, obs):
                raise RuntimeError("Agent step failed")

        class ErrorConfig(type(mock_agent_config)):
            def make(self, *args) -> "ErrorAgent":
                return ErrorAgent(config=self)

        config = ErrorConfig()

        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=config,
            env_config=mock_env_config,
        )

        # Episode should raise the error
        with pytest.raises(RuntimeError, match="Agent step failed"):
            trajectory = episode.run()

        # But error should be saved in trajectory before raising
        # Load the trajectory to verify
        from agentlab2.storage import FileStorage

        storage = FileStorage(tmp_dir)
        traj_id = f"{episode.config.task_id}_ep{episode.config.id}"
        trajectory = storage.load_trajectory(traj_id)

        # Find the agent output step with error
        agent_steps = [s for s in trajectory.steps if isinstance(s.output, AgentOutput)]
        assert len(agent_steps) > 0, "No agent steps found in trajectory"

        error_step = next((s for s in agent_steps if s.output.error is not None), None)
        assert error_step is not None, "No error found in agent steps"
        assert error_step.output.error.error_type == "RuntimeError"
        assert "Agent step failed" in error_step.output.error.exception_str

    def test_episode_captures_env_error(self, tmp_dir, mock_agent_config, mock_tool_config):
        """Test Episode captures environment errors correctly in trajectory."""

        from agentlab2.core import ActionSchema, Task

        class ErrorTask(Task):
            id = "error_task"
            validate_per_step = True

            def setup(self, tool):
                from agentlab2.core import Observation

                return Observation.from_text("Start"), {}

            def validate_task(self, obs):
                raise ValueError("Environment validation failed")

            def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
                return actions

            def accept_agent_stop(self) -> bool:
                return True

            def teardown(self) -> None:
                pass

        from agentlab2.environment import EnvConfig

        error_task = ErrorTask()
        env_config = EnvConfig(task=error_task, tool_config=mock_tool_config)

        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            env_config=env_config,
        )

        # Episode should raise the error
        with pytest.raises(ValueError, match="Environment validation failed"):
            trajectory = episode.run()

        # But error should be saved in trajectory before raising
        from agentlab2.storage import FileStorage

        storage = FileStorage(tmp_dir)
        traj_id = f"{episode.config.task_id}_ep{episode.config.id}"
        trajectory = storage.load_trajectory(traj_id)

        # Find the environment output step with error
        env_steps = [s for s in trajectory.steps if isinstance(s.output, EnvironmentOutput)]
        assert len(env_steps) > 0, "No env steps found in trajectory"

        error_step = next((s for s in env_steps if s.output.error is not None), None)
        assert error_step is not None, "No error found in env steps"
        assert error_step.output.error.error_type == "ValueError"
        assert "Environment validation failed" in error_step.output.error.exception_str

    def test_episode_run_raises_on_duplicate_trajectory(self, tmp_dir, mock_agent_config, mock_env_config) -> None:
        """Running the same episode twice raises FileExistsError (prevents accidental overwrites)."""
        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            env_config=mock_env_config,
        )
        episode.run()

        # Second run with a fresh Episode (same ID, new storage session)
        episode2 = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            env_config=mock_env_config,
        )
        with pytest.raises(FileExistsError):
            episode2.run()

    def test_episode_relaunch_archives_old_trajectory(self, tmp_dir, mock_agent_config, mock_env_config) -> None:
        """An episode loaded from config (_allow_overwrite=True) archives the old trajectory."""
        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            env_config=mock_env_config,
        )
        episode.run()

        # Simulate a relaunch by creating a new episode with _allow_overwrite=True
        episode2 = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            env_config=mock_env_config,
        )
        episode2.allow_overwrite = True
        episode2.run()

        # Both archived and current files should exist
        traj_dir = tmp_dir
        traj_id = f"{episode.config.task_id}_ep{episode.config.id}"
        archived = list(traj_dir.glob(f"{traj_id}.archived_*.metadata.json"))
        assert len(archived) == 1
        assert (traj_dir / f"{traj_id}.metadata.json").exists()
