"""Tests for Experiment with the cube path (CubeBenchmark)."""

import warnings

from cube_harness.experiment import Experiment
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
        resumed = exp.get_episodes_to_run()
        assert len(resumed) == 1
        assert resumed[0].config.task_config is not None
        assert resumed[0].config.task_config.task_id != episodes[0].config.task_config.task_id

    def test_experiment_load_config_round_trip(self, tmp_dir, mock_agent_config, mock_cube_benchmark):
        """Experiment.save_config / load_config round-trip preserves benchmark type."""
        exp = Experiment(
            name="cube_roundtrip",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_cube_benchmark,
        )
        exp.save_config()

        from cube_harness.experiment import Experiment as Exp

        restored = Exp.load_config(str(tmp_dir / "experiment_config.json"))
        assert restored.name == "cube_roundtrip"
        assert type(restored.benchmark) is type(mock_cube_benchmark)

    def test_episode_is_self_contained_without_benchmark(self, tmp_dir, mock_agent_config, mock_cube_benchmark):
        """Episodes created from a cube benchmark can be reloaded without the benchmark arg."""
        exp = Experiment(
            name="self_contained",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_cube_benchmark,
        )
        episodes = exp.get_episodes_to_run()
        assert len(episodes) > 0

        # Pick the first episode config file and reload it without benchmark
        from cube_harness.episode import Episode as Ep
        from cube_harness.storage import FileStorage

        storage = FileStorage(tmp_dir)
        config_files = storage.list_episode_configs()
        assert config_files

        reloaded = Ep.load_episode_from_config(config_files[0])  # no benchmark
        assert reloaded.config.task_config is not None
