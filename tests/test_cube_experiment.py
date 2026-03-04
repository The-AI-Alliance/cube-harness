"""Tests for Experiment with the cube path (CubeBenchmark)."""

import warnings

import pytest

from agentlab2.experiment import Experiment
from tests.conftest import MockCubeTaskConfig


class TestCubeExperiment:
    """Tests for Experiment with the cube path (CubeBenchmark)."""

    def test_cube_benchmark_creates_task_config_episodes(self, tmp_dir, mock_agent_config, mock_cube_benchmark):
        """Experiment with CubeBenchmark creates episodes with task_config, no DeprecationWarning."""
        exp = Experiment(
            name="cube_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_cube_benchmark,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            episodes = exp.get_episodes_to_run()

        assert len(episodes) == len(mock_cube_benchmark.task_metadata)
        for episode in episodes:
            assert isinstance(episode.config.task_config, MockCubeTaskConfig)
            assert episode.config.tool_config is None

    def test_cube_benchmark_resume_reloads_without_benchmark_arg(self, tmp_dir, mock_agent_config, mock_cube_benchmark):
        """Experiment.resume with a cube benchmark reloads episodes without needing benchmark arg."""
        exp = Experiment(
            name="cube_resume",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_cube_benchmark,
            resume=True,
        )

        # First call: no configs on disk yet, creates all episodes from scratch.
        episodes = exp.get_episodes_to_run()
        assert len(episodes) == 2

        # Run only the first episode, leaving the second unstarted.
        episodes[0].run()

        # resume=True: only the unstarted episode should be returned.
        # _find_episodes_to_relaunch calls load_episode_from_config(path, self.benchmark)
        # for each config; for cube episodes the benchmark arg is ignored.
        resumed = exp.get_episodes_to_run()
        assert len(resumed) == 1
        assert resumed[0].config.task_config is not None
        assert resumed[0].config.task_id != episodes[0].config.task_id

    def test_legacy_benchmark_emits_deprecation_warning(self, tmp_dir, mock_agent_config, mock_benchmark):
        """Experiment with a legacy AL2Benchmark emits DeprecationWarning; episodes still created."""
        exp = Experiment(
            name="legacy_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_benchmark,
        )

        with pytest.warns(DeprecationWarning):
            episodes = exp.get_episodes_to_run()

        assert len(episodes) == len(mock_benchmark.env_configs())
        for episode in episodes:
            assert episode.config.task_config is None
            assert episode.config.tool_config is not None
