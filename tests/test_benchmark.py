"""Tests for agentlab2.benchmark module."""

import json

from tests.conftest import MockBenchmark, MockTask, SerializableEnvConfig


class TestBenchmark:
    """Tests for Benchmark abstract class through MockBenchmark."""

    def test_benchmark_creation(self, mock_benchmark):
        """Test Benchmark creation."""
        assert mock_benchmark.env_config is not None
        assert mock_benchmark.metadata == {}

    def test_benchmark_with_metadata(self, mock_env_config, mock_task):
        """Test Benchmark with metadata."""

        benchmark = MockBenchmark(
            tasks_list=[mock_task], env_config=mock_env_config, metadata={"version": "1.0", "author": "test"}
        )
        assert benchmark.metadata["version"] == "1.0"
        assert benchmark.metadata["author"] == "test"

    def test_benchmark_setup(self, mock_benchmark):
        """Test Benchmark setup."""
        mock_benchmark.setup()
        assert mock_benchmark.setup_called

    def test_benchmark_close(self, mock_benchmark):
        """Test Benchmark close."""
        mock_benchmark.close()
        assert mock_benchmark.close_called

    def test_benchmark_env_configs(self, mock_benchmark, mock_task):
        """Test getting env_configs from benchmark."""
        env_configs = mock_benchmark.env_configs()
        assert len(env_configs) == 1
        assert env_configs[0]._task == mock_task

    def test_benchmark_multiple_tasks(self, mock_env_config):
        """Test benchmark with multiple tasks."""

        tasks = [MockTask(goal=f"Task {i}") for i in range(5)]
        benchmark = MockBenchmark(tasks_list=tasks, env_config=mock_env_config)  # type: ignore

        assert len(benchmark.env_configs()) == 5
        for i, env_config in enumerate(benchmark.env_configs()):
            assert env_config._task.goal == f"Task {i}"  # type: ignore

    def test_benchmark_empty_tasks(self, mock_env_config):
        """Test benchmark with no tasks."""

        benchmark = MockBenchmark(tasks_list=[], env_config=mock_env_config)
        assert len(benchmark.env_configs()) == 0

    def test_benchmark_install_default(self, mock_benchmark):
        """Test default install does nothing."""
        # Should not raise
        mock_benchmark.install()

    def test_benchmark_uninstall_default(self, mock_benchmark):
        """Test default uninstall does nothing."""
        # Should not raise
        mock_benchmark.uninstall()

    def test_benchmark_serialization(self, mock_task):
        """Test Benchmark JSON serialization."""
        benchmark = MockBenchmark(tasks_list=[mock_task], env_config=SerializableEnvConfig())
        json_str = benchmark.model_dump_json(serialize_as_any=True)
        data = json.loads(json_str)
        assert "env_config" in data
        assert "metadata" in data

    def test_benchmark_custom_install_uninstall(self, mock_env_config, mock_task):
        """Test benchmark with custom install/uninstall."""

        benchmark = MockBenchmark(tasks_list=[mock_task], env_config=mock_env_config)
        benchmark.install()
        assert benchmark.install_called

        benchmark.uninstall()
        assert benchmark.uninstall_called

    def test_benchmark_lifecycle(self, mock_benchmark):
        """Test full benchmark lifecycle."""
        # Install (optional)
        mock_benchmark.install()

        # Setup
        mock_benchmark.setup()
        assert mock_benchmark.setup_called

        # Get env_configs
        env_configs = mock_benchmark.env_configs()
        assert len(env_configs) > 0
        assert env_configs[0] is not None
        assert env_configs[0]._task is not None

        # Close
        mock_benchmark.close()
        assert mock_benchmark.close_called

        # Uninstall (optional)
        mock_benchmark.uninstall()

    def test_benchmark_env_config_access(self, mock_benchmark):
        """Test accessing env_config from benchmark."""
        env_config = mock_benchmark.env_configs()[0]
        assert env_config is not None
        # Should be able to make environment
        env = env_config.make()
        assert env is not None
        assert env.task is not None
        assert env.task.id == mock_benchmark.env_config._task.id
        assert env.task.id == "mock_task_1"
