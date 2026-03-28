"""Tests for cube_harness.benchmark module."""

import json

from tests.conftest import MockBenchmark, MockTask


class TestBenchmark:
    """Tests for Benchmark abstract class through MockBenchmark."""

    def test_benchmark_creation(self, mock_benchmark):
        """Test Benchmark creation."""
        assert mock_benchmark.tool_config is not None
        assert mock_benchmark.metadata == {}

    def test_benchmark_with_metadata(self, mock_tool_config, mock_task):
        """Test Benchmark with metadata."""

        benchmark = MockBenchmark(
            tasks_list=[mock_task], tool_config=mock_tool_config, metadata={"version": "1.0", "author": "test"}
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
        assert env_configs[0].task == mock_task

    def test_benchmark_multiple_tasks(self, mock_tool_config):
        """Test benchmark with multiple tasks."""

        tasks = [MockTask(goal=f"Task {i}") for i in range(5)]
        benchmark = MockBenchmark(tasks_list=tasks, tool_config=mock_tool_config)  # type: ignore

        assert len(benchmark.env_configs()) == 5
        for i, env_config in enumerate(benchmark.env_configs()):
            assert env_config.task.goal == f"Task {i}"  # type: ignore

    def test_benchmark_empty_tasks(self, mock_tool_config):
        """Test benchmark with no tasks."""

        benchmark = MockBenchmark(tasks_list=[], tool_config=mock_tool_config)
        assert len(benchmark.env_configs()) == 0

    def test_benchmark_serialization(self, mock_benchmark):
        """Test Benchmark JSON serialization."""
        json_str = mock_benchmark.model_dump_json(serialize_as_any=True)
        data = json.loads(json_str)
        assert "tool_config" in data
        assert "metadata" in data

    def test_benchmark_lifecycle(self, mock_benchmark):
        """Test full benchmark lifecycle."""
        # Setup
        mock_benchmark.setup()
        assert mock_benchmark.setup_called

        # Get env_configs
        env_configs = mock_benchmark.env_configs()
        assert len(env_configs) > 0
        assert env_configs[0] is not None
        assert env_configs[0].task is not None

        # Close
        mock_benchmark.close()
        assert mock_benchmark.close_called

    def test_benchmark_env_config_access(self, mock_benchmark):
        """Test accessing env_config from benchmark."""
        env_config = mock_benchmark.env_configs()[0]
        assert env_config is not None
        # Should be able to make environment
        env = env_config.make()
        assert env is not None
        assert env.task is not None
        assert env.task.id == env_config.task.id
        assert env.task.id == "mock_task_1"
