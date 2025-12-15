"""Tests for agentlab2.experiment module."""

import json
import os
import tempfile

from agentlab2.core import EnvironmentOutput, Observation, Trajectory
from agentlab2.environment import EnvironmentConfig, ToolboxEnv
from agentlab2.episode import Episode
from agentlab2.experiment import Experiment, ExpResult
from tests.conftest import MockBenchmark, MockTask


class TestExpResult:
    """Tests for ExpResult class."""

    def test_exp_result_creation(self):
        """Test ExpResult creation."""
        result = ExpResult(exp_id="test_exp_123", tasks_num=10)

        assert result.exp_id == "test_exp_123"
        assert result.tasks_num == 10
        assert result.config == {}
        assert result.trajectories == {}
        assert result.failures == {}

    def test_exp_result_with_trajectories(self):
        """Test ExpResult with trajectories."""
        obs = Observation.from_text("done")
        traj = Trajectory(metadata={"task_id": "task_1"}, steps=[EnvironmentOutput(obs=obs, reward=1.0, done=True)])

        result = ExpResult(exp_id="test_exp", tasks_num=1, trajectories={"task_1": traj})

        assert len(result.trajectories) == 1
        assert "task_1" in result.trajectories

    def test_exp_result_with_failures(self):
        """Test ExpResult with failures."""
        result = ExpResult(
            exp_id="test_exp",
            tasks_num=3,
            failures={"task_2": "Connection error", "task_3": "Timeout"},
        )

        assert len(result.failures) == 2
        assert result.failures["task_2"] == "Connection error"

    def test_exp_result_with_config(self):
        """Test ExpResult with config."""
        config = {"model": "gpt-4", "temperature": 0.7}
        result = ExpResult(exp_id="test_exp", tasks_num=5, config=config)

        assert result.config["model"] == "gpt-4"
        assert result.config["temperature"] == 0.7

    def test_exp_result_serialization(self):
        """Test ExpResult JSON serialization."""
        result = ExpResult(exp_id="test_exp", tasks_num=5, config={"key": "value"})
        json_str = result.model_dump_json()
        data = json.loads(json_str)

        assert data["exp_id"] == "test_exp"
        assert data["tasks_num"] == 5
        assert data["config"]["key"] == "value"


class TestExperiment:
    """Tests for Experiment class."""

    def test_experiment_creation(self, mock_agent_config, mock_benchmark):
        """Test Experiment creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(
                name="test_experiment",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                benchmark=mock_benchmark,
            )

            assert exp.name == "test_experiment"
            assert exp.output_dir == tmpdir
            assert exp.agent_config == mock_agent_config
            assert exp.benchmark == mock_benchmark

    def test_experiment_config_property(self, mock_agent_config, mock_benchmark):
        """Test Experiment config property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(
                name="test_experiment",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                benchmark=mock_benchmark,
            )

            config = exp.config
            assert config["name"] == "test_experiment"
            assert config["output_dir"] == tmpdir
            assert "agent_config" in config
            assert "benchmark" in config

    def test_experiment_create_episodes(self, mock_agent_config, mock_benchmark):
        """Test Experiment create_episodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(
                name="test_experiment",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                benchmark=mock_benchmark,
            )

            episodes = exp.create_episodes()

            assert len(episodes) == len(mock_benchmark.tasks())
            for i, episode in enumerate(episodes):
                assert isinstance(episode, Episode)
                assert episode.id == i
                assert episode.exp_name == "test_experiment"
                assert episode.output_dir == tmpdir

    def test_experiment_create_episodes_multiple_tasks(self, mock_agent_config, mock_env_config):
        """Test Experiment create_episodes with multiple tasks."""

        tasks = [MockTask(goal=f"Goal {i}") for i in range(5)]
        benchmark = MockBenchmark(tasks_list=tasks, env_config=mock_env_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(
                name="multi_task_exp",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                benchmark=benchmark,
            )

            episodes = exp.create_episodes()
            assert len(episodes) == 5

    def test_experiment_save_config(self, mock_agent_config, mock_task):
        """Test Experiment save_config."""

        # Use a serializable env config (without tools)
        class SerializableEnvConfig(EnvironmentConfig):
            def make(self, task):
                return ToolboxEnv(task=task, tools=[])

        benchmark = MockBenchmark(tasks_list=[mock_task], env_config=SerializableEnvConfig())

        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(
                name="test_experiment",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                benchmark=benchmark,
            )

            exp.save_config()

            config_path = os.path.join(tmpdir, "experiment_config.json")
            assert os.path.exists(config_path)

            with open(config_path) as f:
                saved_config = json.load(f)

            assert saved_config["name"] == "test_experiment"

    def test_experiment_save_config_creates_directory(self, mock_agent_config, mock_task):
        """Test Experiment save_config creates output directory."""

        # Use a serializable env config (without tools)
        class SerializableEnvConfig(EnvironmentConfig):
            def make(self, task):
                return ToolboxEnv(task=task, tools=[])

        benchmark = MockBenchmark(tasks_list=[mock_task], env_config=SerializableEnvConfig())

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested", "output")
            exp = Experiment(
                name="test_experiment",
                output_dir=nested_dir,
                agent_config=mock_agent_config,
                benchmark=benchmark,
            )

            exp.save_config()

            assert os.path.exists(nested_dir)
            assert os.path.exists(os.path.join(nested_dir, "experiment_config.json"))

    def test_experiment_print_stats_no_trajectories(self, mock_agent_config, mock_benchmark, capsys):
        """Test Experiment print_stats with no trajectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(
                name="test_experiment",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                benchmark=mock_benchmark,
            )

            result = ExpResult(exp_id="test", tasks_num=0)
            exp.print_stats(result)

            # Should not crash, just log that there are no trajectories

    def test_experiment_print_stats_with_trajectories(self, mock_agent_config, mock_benchmark):
        """Test Experiment print_stats with trajectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(
                name="test_experiment",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                benchmark=mock_benchmark,
            )

            # Create trajectories
            trajectories = {}
            for i in range(3):
                obs = Observation.from_text("done")
                traj = Trajectory(
                    metadata={"task_id": f"task_{i}"},
                    steps=[
                        EnvironmentOutput(obs=obs, reward=0.0),
                        EnvironmentOutput(obs=obs, reward=float(i) / 2, done=True),
                    ],
                )
                trajectories[f"task_{i}"] = traj

            result = ExpResult(exp_id="test", tasks_num=3, trajectories=trajectories)

            # Should not crash
            exp.print_stats(result)

    def test_experiment_print_stats_with_failures(self, mock_agent_config, mock_benchmark):
        """Test Experiment print_stats with failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(
                name="test_experiment",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                benchmark=mock_benchmark,
            )

            obs = Observation.from_text("done")
            traj = Trajectory(metadata={"task_id": "task_0"}, steps=[EnvironmentOutput(obs=obs, reward=1.0, done=True)])

            result = ExpResult(
                exp_id="test",
                tasks_num=3,
                trajectories={"task_0": traj},
                failures={"task_1": "Error 1", "task_2": "Error 2"},
            )

            # Should not crash
            exp.print_stats(result)

    def test_experiment_serialization(self, mock_agent_config, mock_task):
        """Test Experiment JSON serialization."""

        # Use a serializable env config (without tools)
        class SerializableEnvConfig(EnvironmentConfig):
            def make(self, task):
                return ToolboxEnv(task=task, tools=[])

        benchmark = MockBenchmark(tasks_list=[mock_task], env_config=SerializableEnvConfig())

        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(
                name="test_experiment",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                benchmark=benchmark,
            )

            json_str = exp.model_dump_json(serialize_as_any=True)
            data = json.loads(json_str)

            assert data["name"] == "test_experiment"
            assert "agent_config" in data
            assert "benchmark" in data

    def test_experiment_episodes_have_correct_env_config(self, mock_agent_config, mock_benchmark):
        """Test that created episodes have correct env_config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(
                name="test_experiment",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                benchmark=mock_benchmark,
            )

            episodes = exp.create_episodes()

            for episode in episodes:
                assert episode.env_config == mock_benchmark.env_config

    def test_experiment_episodes_have_tasks_from_benchmark(self, mock_agent_config, mock_benchmark):
        """Test that created episodes have tasks from benchmark."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(
                name="test_experiment",
                output_dir=tmpdir,
                agent_config=mock_agent_config,
                benchmark=mock_benchmark,
            )

            episodes = exp.create_episodes()
            tasks = mock_benchmark.tasks()

            for episode, task in zip(episodes, tasks):
                assert episode.task == task
