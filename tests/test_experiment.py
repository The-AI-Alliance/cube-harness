"""Tests for agentlab2.experiment module."""

import json

from agentlab2.core import EnvironmentOutput, Observation, Trajectory, TrajectoryStep
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
        step = TrajectoryStep(output=EnvironmentOutput(obs=obs, reward=1.0, done=True))
        traj = Trajectory(id="test_traj", metadata={"task_id": "task_1"}, steps=[step])

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

    def test_experiment_creation(self, tmp_dir, mock_agent_config, mock_benchmark):
        """Test Experiment creation."""
        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_benchmark,
        )

        assert exp.name == "test_experiment"
        assert exp.output_dir == tmp_dir
        assert exp.agent_config == mock_agent_config
        assert exp.benchmark == mock_benchmark

    def test_experiment_config_property(self, tmp_dir, mock_agent_config, mock_benchmark):
        """Test Experiment config property."""
        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_benchmark,
        )

        config = exp.config
        assert config["name"] == "test_experiment"
        assert config["output_dir"] == tmp_dir
        assert "agent_config" in config
        assert "benchmark" in config

    def test_experiment_create_episodes(self, tmp_dir, mock_agent_config, mock_benchmark):
        """Test Experiment create_episodes."""
        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_benchmark,
        )

        episodes = exp.create_episodes()

        assert len(episodes) == len(mock_benchmark.env_configs())
        for i, episode in enumerate(episodes):
            assert isinstance(episode, Episode)
            assert episode.config.id == i
            assert episode.config.output_dir == tmp_dir
            assert episode.env_config.task is not None

    def test_experiment_create_episodes_multiple_tasks(self, tmp_dir, mock_agent_config, mock_tool_config):
        """Test Experiment create_episodes with multiple tasks."""
        tasks = [MockTask(goal=f"Goal {i}") for i in range(5)]
        benchmark = MockBenchmark(tasks_list=tasks, tool_config=mock_tool_config)

        exp = Experiment(
            name="multi_task_exp",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=benchmark,
        )

        episodes = exp.create_episodes()
        assert len(episodes) == 5
        for i, episode in enumerate(episodes):
            assert episode.env_config.task == tasks[i]

    def test_experiment_save_config(self, tmp_dir, mock_agent_config, mock_benchmark):
        """Test Experiment save_config."""

        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_benchmark,
        )

        exp.save_config()

        config_path = tmp_dir / "experiment_config.json"
        assert config_path.exists()

        with open(config_path) as f:
            saved_config = json.load(f)

        assert saved_config["name"] == "test_experiment"

    def test_experiment_save_config_creates_directory(self, tmp_dir, mock_agent_config, mock_benchmark):
        """Test Experiment save_config creates output directory."""

        nested_dir = tmp_dir / "nested" / "output"
        exp = Experiment(
            name="test_experiment",
            output_dir=nested_dir,
            agent_config=mock_agent_config,
            benchmark=mock_benchmark,
        )

        exp.save_config()

        assert nested_dir.exists()
        assert (nested_dir / "experiment_config.json").exists()

    def test_experiment_serialization(self, tmp_dir, mock_agent_config, mock_benchmark):
        """Test Experiment JSON serialization."""

        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_benchmark,
        )

        json_str = exp.model_dump_json(serialize_as_any=True)
        data = json.loads(json_str)

        assert data["name"] == "test_experiment"
        assert "agent_config" in data
        assert "benchmark" in data

    def test_experiment_episodes_have_correct_config(self, tmp_dir, mock_agent_config, mock_benchmark):
        """Test that created episodes have correct env_config."""
        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_benchmark,
        )

        episodes = exp.create_episodes()

        for episode in episodes:
            assert episode.env_config.tool_config == mock_benchmark.tool_config

    def test_experiment_episodes_have_tasks_from_benchmark(self, tmp_dir, mock_agent_config, mock_benchmark):
        """Test that created episodes have tasks from benchmark."""
        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_benchmark,
        )

        episodes = exp.create_episodes()
        env_configs = mock_benchmark.env_configs()

        for episode, env_config in zip(episodes, env_configs):
            assert episode.config.task_id == env_config.task.id

    def test_experiment_relaunch_failed_episodes(self, tmp_dir, mock_agent_config, mock_tool_config):
        """Test relaunch_failed_episodes correctly identifies and relaunches failed episodes."""
        from agentlab2.core import StepError

        # Create tasks
        tasks = [MockTask(goal=f"Task {i}") for i in range(3)]
        benchmark = MockBenchmark(tasks_list=tasks, tool_config=mock_tool_config)

        exp = Experiment(
            name="test_relaunch_failed",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=benchmark,
        )

        # Create episodes
        episodes = exp.create_episodes()

        # Run first episode to completion (successful)
        episodes[0].run()

        # Run second episode but simulate failure by creating a trajectory with error
        from agentlab2.core import Observation, Trajectory, TrajectoryStep
        from agentlab2.storage import FileStorage

        storage = FileStorage(tmp_dir)
        failed_traj = Trajectory(
            id=f"{episodes[1].config.task_id}_ep{episodes[1].config.id}",
            metadata={"task_id": episodes[1].config.task_id},
        )
        obs = Observation.from_text("test")
        from agentlab2.core import EnvironmentOutput

        # Add a step with error to simulate failure
        failed_env_output = EnvironmentOutput(obs=obs, error=StepError.from_exception(ValueError("Test error")))
        failed_traj.steps.append(TrajectoryStep(output=failed_env_output))
        storage.save_trajectory(failed_traj)

        # Third episode not started (no trajectory)

        # Test relaunch_failed_episodes
        benchmark.setup()
        try:
            results = exp.relaunch_failed_episodes()
            # Should find episode 1 as failed (has trajectory with error)
            assert results.tasks_num == 1
            # The relaunched episode should run successfully
            assert len(results.trajectories) == 1
            assert len(results.failures) == 0
        finally:
            benchmark.close()

    def test_experiment_relaunch_unstarted_episodes(self, tmp_dir, mock_agent_config, mock_tool_config):
        """Test relaunch_unstarted_episodes correctly identifies and relaunches unstarted episodes."""
        # Create tasks
        tasks = [MockTask(goal=f"Task {i}") for i in range(3)]
        benchmark = MockBenchmark(tasks_list=tasks, tool_config=mock_tool_config)

        exp = Experiment(
            name="test_relaunch_unstarted",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=benchmark,
        )

        # Create episodes (this saves configs)
        episodes = exp.create_episodes()

        # Run only first episode (leaving 2 and 3 unstarted)
        episodes[0].run()

        # Test relaunch_unstarted_episodes
        benchmark.setup()
        try:
            results = exp.relaunch_unstarted_episodes()
            # Should find episodes 1 and 2 as unstarted
            assert results.tasks_num == 2
            # Both should run successfully
            assert len(results.trajectories) == 2
            assert len(results.failures) == 0
        finally:
            benchmark.close()

    def test_experiment_relaunch_no_episodes(self, tmp_dir, mock_agent_config, mock_benchmark):
        """Test relaunch methods return empty results when no episodes to relaunch."""
        exp = Experiment(
            name="test_no_relaunch",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_benchmark,
        )

        # Create episodes and run all successfully
        episodes = exp.create_episodes()
        for episode in episodes:
            episode.run()

        # Test relaunch_failed_episodes - should find nothing
        mock_benchmark.setup()
        try:
            failed_results = exp.relaunch_failed_episodes()
            assert failed_results.tasks_num == 0
            assert len(failed_results.trajectories) == 0

            # Test relaunch_unstarted_episodes - should find nothing
            unstarted_results = exp.relaunch_unstarted_episodes()
            assert unstarted_results.tasks_num == 0
            assert len(unstarted_results.trajectories) == 0
        finally:
            mock_benchmark.close()
