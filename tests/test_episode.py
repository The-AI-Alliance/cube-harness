"""Tests for agentlab2.episode module."""

import json
import os
import tempfile

import pytest

from agentlab2.core import Action, AgentOutput, EnvironmentOutput, Observation, Trajectory
from agentlab2.episode import MAX_STEPS, Episode
from tests.conftest import MockAgent


class TestEpisode:
    """Tests for Episode class."""

    def test_episode_creation(self, mock_agent_config, mock_task, mock_env_config):
        """Test Episode creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Episode(
                id=0,
                exp_name="test_exp",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                task=mock_task,
                env_config=mock_env_config,
            )

            assert episode.id == 0
            assert episode.exp_name == "test_exp"
            assert episode.output_dir == tmpdir
            assert episode.max_steps == MAX_STEPS

    def test_episode_custom_max_steps(self, mock_agent_config, mock_task, mock_env_config):
        """Test Episode with custom max_steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Episode(
                id=0,
                exp_name="test_exp",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                task=mock_task,
                env_config=mock_env_config,
                max_steps=10,
            )

            assert episode.max_steps == 10

    def test_episode_run_completes(self, mock_agent_config, mock_task, mock_env_config):
        """Test Episode run completes successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Episode(
                id=0,
                exp_name="test_exp",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                task=mock_task,
                env_config=mock_env_config,
            )

            trajectory = episode.run()

            assert isinstance(trajectory, Trajectory)
            assert trajectory.metadata["task_id"] == mock_task.id
            # Should have initial env output + agent output + final env output
            assert len(trajectory.steps) >= 2

    def test_episode_run_saves_trajectory(self, mock_agent_config, mock_task, mock_env_config):
        """Test Episode run saves trajectory files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Episode(
                id=0,
                exp_name="test_exp",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                task=mock_task,
                env_config=mock_env_config,
            )

            episode.run()

            # Check trajectory files exist
            traj_dir = os.path.join(tmpdir, "trajectories")
            assert os.path.exists(traj_dir)

            # Should have metadata and jsonl files
            files = os.listdir(traj_dir)
            assert any(".metadata.json" in f for f in files)
            assert any(".jsonl" in f for f in files)

    def test_episode_run_metadata_file_content(self, mock_agent_config, mock_task, mock_env_config):
        """Test Episode run creates correct metadata file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Episode(
                id=0,
                exp_name="test_exp",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                task=mock_task,
                env_config=mock_env_config,
            )

            episode.run()

            # Read metadata file
            traj_dir = os.path.join(tmpdir, "trajectories")
            metadata_file = [f for f in os.listdir(traj_dir) if ".metadata.json" in f][0]
            with open(os.path.join(traj_dir, metadata_file)) as f:
                metadata = json.load(f)

            assert metadata["task_id"] == mock_task.id

    def test_episode_run_jsonl_content(self, mock_agent_config, mock_task, mock_env_config):
        """Test Episode run creates correct JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Episode(
                id=0,
                exp_name="test_exp",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                task=mock_task,
                env_config=mock_env_config,
            )

            episode.run()

            # Read JSONL file
            traj_dir = os.path.join(tmpdir, "trajectories")
            jsonl_file = [f for f in os.listdir(traj_dir) if ".jsonl" in f][0]
            with open(os.path.join(traj_dir, jsonl_file)) as f:
                lines = f.readlines()

            # Should have at least one step saved
            assert len(lines) >= 1

            # Each line should be valid JSON
            for line in lines:
                if line.strip():
                    data = json.loads(line)
                    assert isinstance(data, dict)

    def test_episode_run_respects_max_steps(self, mock_agent_config, mock_task, mock_env_config):
        """Test Episode run respects max_steps limit."""

        # Create an agent that never stops
        class NeverStopsAgent(MockAgent):
            def step(self, obs):
                self.step_count += 1
                # Return non-stop action
                return AgentOutput(actions=[Action(name="click", arguments={"element_id": "btn"})])

        class NeverStopsConfig(type(mock_agent_config)):
            def make(self, **kwargs):
                agent = NeverStopsAgent(config=self)
                return agent

        config = NeverStopsConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Episode(
                id=0,
                exp_name="test_exp",
                output_dir=tmpdir,
                agent_config=config,
                task=mock_task,
                env_config=mock_env_config,
                max_steps=3,
            )

            trajectory = episode.run()

            # Should have stopped at max_steps
            # Steps: initial_env + (agent + env) * max_steps
            # But it's limited by max_steps, so agent should only step 3 times
            agent_steps = sum(1 for step in trajectory.steps if isinstance(step, AgentOutput))
            assert agent_steps <= 3

    def test_episode_run_stops_on_done(self, mock_agent_config, mock_task, mock_env_config):
        """Test Episode run stops when done=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Episode(
                id=0,
                exp_name="test_exp",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                task=mock_task,
                env_config=mock_env_config,
                max_steps=100,  # High limit
            )

            trajectory = episode.run()

            # Should stop before max_steps because agent returns final_step
            last_env_step = trajectory.last_env_step()
            assert last_env_step.done is True

    def test_episode_save_trajectory_creates_directory(self, mock_agent_config, mock_task, mock_env_config):
        """Test save_trajectory creates trajectory directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Episode(
                id=0,
                exp_name="test_exp",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                task=mock_task,
                env_config=mock_env_config,
            )

            trajectory = Trajectory(metadata={"task_id": "test"})
            episode.save_trajectory(trajectory)

            traj_dir = os.path.join(tmpdir, "trajectories")
            assert os.path.exists(traj_dir)

    def test_episode_save_step_without_trajectory(self, mock_agent_config, mock_task, mock_env_config):
        """Test save_step raises error if called before save_trajectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Episode(
                id=0,
                exp_name="test_exp",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                task=mock_task,
                env_config=mock_env_config,
            )

            obs = Observation.from_text("test")
            step = EnvironmentOutput(obs=obs)

            with pytest.raises(ValueError, match="Trajectory path not set"):
                episode.save_step(step)

    def test_episode_save_step_appends(self, mock_agent_config, mock_task, mock_env_config):
        """Test save_step appends to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Episode(
                id=0,
                exp_name="test_exp",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                task=mock_task,
                env_config=mock_env_config,
            )

            trajectory = Trajectory(metadata={"task_id": "test"})
            episode.save_trajectory(trajectory)

            # Save multiple steps
            for i in range(3):
                obs = Observation.from_text(f"step {i}")
                step = EnvironmentOutput(obs=obs)
                episode.save_step(step)

            # Read JSONL file
            traj_dir = os.path.join(tmpdir, "trajectories")
            jsonl_file = [f for f in os.listdir(traj_dir) if ".jsonl" in f][0]
            with open(os.path.join(traj_dir, jsonl_file)) as f:
                lines = f.readlines()

            assert len(lines) == 3

    def test_episode_closes_env_on_completion(self, mock_agent_config, mock_task, mock_env_config):
        """Test Episode closes environment after run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Episode(
                id=0,
                exp_name="test_exp",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                task=mock_task,
                env_config=mock_env_config,
            )

            episode.run()

            # Task teardown should have been called
            assert mock_task.teardown_called

    def test_episode_closes_env_on_error(self, mock_agent_config, mock_task, mock_env_config):
        """Test Episode closes environment even when error occurs."""

        class ErrorAgent(MockAgent):
            def step(self, obs):
                raise RuntimeError("Test error")

        class ErrorConfig(type(mock_agent_config)):
            def make(self, **kwargs):
                return ErrorAgent(config=self)

        config = ErrorConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Episode(
                id=0,
                exp_name="test_exp",
                output_dir=tmpdir,
                agent_config=config,
                task=mock_task,
                env_config=mock_env_config,
            )

            with pytest.raises(RuntimeError, match="Test error"):
                episode.run()

            # Environment should still be closed
            assert mock_task.teardown_called

    def test_episode_output_filename(self, mock_agent_config, mock_task, mock_env_config):
        """Test Episode generates correct output filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Episode(
                id=42,
                exp_name="test_exp",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                task=mock_task,
                env_config=mock_env_config,
            )

            episode.run()

            traj_dir = os.path.join(tmpdir, "trajectories")
            files = os.listdir(traj_dir)

            # Should contain run id and task id
            assert any("run42" in f for f in files)
            assert any(mock_task.id in f for f in files)
