"""Tests for cube_harness.experiment module."""

import json
import time

from cube.benchmark import Benchmark as CubeBenchmark
from cube.benchmark import BenchmarkMetadata
from cube.core import EnvironmentOutput, Observation
from cube.task import TaskMetadata

from cube_harness.core import Trajectory, TrajectoryStep
from cube_harness.episode import Episode
from cube_harness.exp_runner import run_sequentially
from cube_harness.episode_status import EpisodeStatus
from cube_harness.experiment import Experiment, ExpResult, sweep_stale_statuses
from cube_harness.storage import FileStorage
from tests.conftest import MockCubeBenchmark, MockCubeTaskConfig


def _make_benchmark(n: int) -> CubeBenchmark:
    """Create a cube benchmark with n tasks for testing."""
    task_meta = {f"task_{i}": TaskMetadata(id=f"task_{i}") for i in range(n)}

    class _NTaskBenchmark(MockCubeBenchmark):
        benchmark_metadata = BenchmarkMetadata(name=f"n{n}-task", version="0.1.0", description="test")
        task_metadata = task_meta
        task_config_class = MockCubeTaskConfig

    return _NTaskBenchmark()


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

    def test_experiment_creation(self, tmp_dir, mock_agent_config, mock_cube_benchmark):
        """Test Experiment creation."""
        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_cube_benchmark,
        )

        assert exp.name == "test_experiment"
        assert exp.output_dir == tmp_dir
        assert exp.agent_config == mock_agent_config
        assert exp.benchmark == mock_cube_benchmark

    def test_experiment_config_property(self, tmp_dir, mock_agent_config, mock_cube_benchmark):
        """Test Experiment config property."""
        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_cube_benchmark,
        )

        config = exp.config
        assert config["name"] == "test_experiment"
        assert config["output_dir"] == tmp_dir
        assert "agent_config" in config
        assert "benchmark" in config

    def test_experiment_create_episodes(self, tmp_dir, mock_agent_config, mock_cube_benchmark):
        """Test Experiment creates one episode per task in the benchmark."""
        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_cube_benchmark,
        )

        episodes = exp.get_episodes_to_run()

        assert len(episodes) == len(mock_cube_benchmark.task_metadata)
        for i, episode in enumerate(episodes):
            assert isinstance(episode, Episode)
            assert episode.config.id == i
            assert episode.config.output_dir == tmp_dir
            assert episode.config.task_config is not None

    def test_experiment_create_episodes_multiple_tasks(self, tmp_dir, mock_agent_config):
        """Test Experiment create_episodes with multiple tasks."""
        benchmark = _make_benchmark(5)

        exp = Experiment(
            name="multi_task_exp",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=benchmark,
        )

        episodes = exp.get_episodes_to_run()
        assert len(episodes) == 5
        task_ids = {e.config.task_config.task_id for e in episodes}
        assert task_ids == {f"task_{i}" for i in range(5)}

    def test_experiment_save_config(self, tmp_dir, mock_agent_config, mock_cube_benchmark):
        """Test Experiment save_config."""

        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_cube_benchmark,
        )

        exp.save_config()

        config_path = tmp_dir / "experiment_config.json"
        assert config_path.exists()

        with open(config_path) as f:
            saved_config = json.load(f)

        assert saved_config["name"] == "test_experiment"

    def test_experiment_save_config_creates_directory(self, tmp_dir, mock_agent_config, mock_cube_benchmark):
        """Test Experiment save_config creates output directory."""

        nested_dir = tmp_dir / "nested" / "output"
        exp = Experiment(
            name="test_experiment",
            output_dir=nested_dir,
            agent_config=mock_agent_config,
            benchmark=mock_cube_benchmark,
        )

        exp.save_config()

        assert nested_dir.exists()
        assert (nested_dir / "experiment_config.json").exists()

    def test_experiment_serialization(self, tmp_dir, mock_agent_config, mock_cube_benchmark):
        """Test Experiment JSON serialization."""

        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_cube_benchmark,
        )

        json_str = exp.model_dump_json(serialize_as_any=True)
        data = json.loads(json_str)

        assert data["name"] == "test_experiment"
        assert "agent_config" in data
        assert "benchmark" in data

    def test_experiment_episodes_have_tasks_from_benchmark(self, tmp_dir, mock_agent_config, mock_cube_benchmark):
        """Test that created episodes have tasks from benchmark."""
        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_cube_benchmark,
        )

        episodes = exp.get_episodes_to_run()
        expected_task_ids = set(mock_cube_benchmark.task_metadata.keys())
        actual_task_ids = {e.config.task_config.task_id for e in episodes}
        assert actual_task_ids == expected_task_ids

    def test_retry_failed_episodes(self, tmp_dir, mock_agent_config):
        """resume=True returns FAILED + missing-status episodes (gated by max_retries)."""
        benchmark = _make_benchmark(3)

        exp = Experiment(
            name="test_retry_failed",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=benchmark,
        )

        episodes = exp.get_episodes_to_run()

        # Episode 0: complete successfully → COMPLETED status
        episodes[0].run()

        # Episode 1: write a FAILED status manually
        storage = FileStorage(tmp_dir)
        traj_id_1 = f"{episodes[1].config.task_config.task_id}_ep{episodes[1].config.id}"
        storage.write_episode_status(
            traj_id_1,
            EpisodeStatus(
                status="FAILED",
                task_id=episodes[1].config.task_config.task_id,
                episode_id=episodes[1].config.id,
                started_at=time.time() - 10,
                ended_at=time.time(),
                last_heartbeat_at=time.time(),
                error_type="RuntimeError",
                error_message="boom",
                retry_count=0,
            ),
        )

        # Episode 2: no status.json (never started)

        exp.resume = True
        failed_episodes = exp.get_episodes_to_run()
        # Both episode 1 (FAILED) and episode 2 (missing status) qualify
        ids = {ep.config.id for ep in failed_episodes}
        assert ids == {episodes[1].config.id, episodes[2].config.id}

    def test_retry_respects_max_retries(self, tmp_dir, mock_agent_config):
        """retry_count >= max_retries excludes the episode from retry."""
        benchmark = _make_benchmark(2)
        exp = Experiment(
            name="test_max_retries",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=benchmark,
            max_retries=2,
        )
        episodes = exp.get_episodes_to_run()
        storage = FileStorage(tmp_dir)
        # Episode 0: capped (retry_count == max_retries)
        storage.write_episode_status(
            f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}",
            EpisodeStatus(
                status="FAILED",
                task_id=episodes[0].config.task_config.task_id,
                episode_id=episodes[0].config.id,
                started_at=time.time(),
                retry_count=2,
            ),
        )
        # Episode 1: still under cap
        storage.write_episode_status(
            f"{episodes[1].config.task_config.task_id}_ep{episodes[1].config.id}",
            EpisodeStatus(
                status="FAILED",
                task_id=episodes[1].config.task_config.task_id,
                episode_id=episodes[1].config.id,
                started_at=time.time(),
                retry_count=1,
            ),
        )
        exp.resume = True
        retried = exp.get_episodes_to_run()
        ids = {ep.config.id for ep in retried}
        assert ids == {episodes[1].config.id}

    def test_resume_returns_unstarted(self, tmp_dir, mock_agent_config):
        """Test resume=True returns only unstarted episodes."""
        benchmark = _make_benchmark(3)

        exp = Experiment(
            name="test_resume_unstarted",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=benchmark,
        )

        # First call creates all episodes and saves configs
        episodes = exp.get_episodes_to_run()
        assert len(episodes) == 3

        # Run only first episode (leaving 2 and 3 unstarted)
        episodes[0].run()

        # With resume=True, should return only unstarted episodes
        exp.resume = True
        resumed_episodes = exp.get_episodes_to_run()
        assert len(resumed_episodes) == 2

    def test_run_sequentially(self, tmp_dir, mock_agent_config, mock_cube_benchmark):
        """run_sequentially completes all episodes and returns trajectories keyed by task_id."""
        exp = Experiment(
            name="test_run_sequential",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_cube_benchmark,
        )

        result = run_sequentially(exp)

        expected_task_ids = set(mock_cube_benchmark.task_metadata.keys())
        assert set(result.trajectories.keys()) == expected_task_ids
        assert result.failures == {}

    def test_resume_and_retry_empty_when_all_succeeded(self, tmp_dir, mock_agent_config, mock_cube_benchmark):
        """resume returns empty when all episodes succeeded."""
        exp = Experiment(
            name="test_no_relaunch",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=mock_cube_benchmark,
        )

        # Create episodes and run all successfully
        episodes = exp.get_episodes_to_run()
        for episode in episodes:
            episode.run()

        exp.resume = True
        exp.resume = True
        assert exp.get_episodes_to_run() == []


class TestStatusBasedSelection:
    """Focused tests for the status-driven selection logic."""

    def test_stale_sweep_marks_orphaned_running(self, tmp_dir, mock_agent_config):
        """A RUNNING entry with a stale heartbeat is swept to STALE and becomes retriable."""
        benchmark = _make_benchmark(1)
        exp = Experiment(
            name="test_stale_sweep",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=benchmark,
        )
        episodes = exp.get_episodes_to_run()
        traj_id = f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}"
        storage = FileStorage(tmp_dir)
        # Heartbeat 1 hour ago — older than step_timeout(1s) + cancel_grace(1s)
        storage.write_episode_status(
            traj_id,
            EpisodeStatus(
                status="RUNNING",
                task_id=episodes[0].config.task_config.task_id,
                episode_id=episodes[0].config.id,
                started_at=time.time() - 3600,
                last_heartbeat_at=time.time() - 3600,
                current_step=5,
                retry_count=0,
            ),
        )

        swept = sweep_stale_statuses(storage, step_timeout_s=1.0, cancel_grace_s=1.0, orphan_threshold_s=10.0)
        assert swept == [traj_id]
        status = storage.read_episode_status(traj_id)
        assert status is not None
        assert status.status == "STALE"

        exp.resume = True
        retried = exp.get_episodes_to_run(step_timeout_s=1.0, cancel_grace_s=1.0, orphan_threshold_s=10.0)
        assert len(retried) == 1
        assert retried[0].config.id == episodes[0].config.id

    def test_queued_orphan_swept_to_stale(self, tmp_dir, mock_agent_config):
        """A QUEUED entry older than `orphan_threshold_s` is swept to STALE."""
        benchmark = _make_benchmark(1)
        exp = Experiment(
            name="test_queued_orphan",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=benchmark,
        )
        episodes = exp.get_episodes_to_run()
        traj_id = f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}"
        storage = FileStorage(tmp_dir)
        # QUEUED with started_at way in the past — Ray never picked it up.
        storage.write_episode_status(
            traj_id,
            EpisodeStatus(
                status="QUEUED",
                task_id=episodes[0].config.task_config.task_id,
                episode_id=episodes[0].config.id,
                started_at=time.time() - 7200,
                last_heartbeat_at=None,
                current_step=0,
            ),
        )
        swept = sweep_stale_statuses(storage, step_timeout_s=1.0, cancel_grace_s=1.0, orphan_threshold_s=10.0)
        assert swept == [traj_id]
        assert storage.read_episode_status(traj_id).status == "STALE"

    def test_queued_fresh_not_swept(self, tmp_dir, mock_agent_config):
        """A QUEUED entry younger than `orphan_threshold_s` is left alone (still queued)."""
        benchmark = _make_benchmark(1)
        exp = Experiment(
            name="test_queued_fresh",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=benchmark,
        )
        episodes = exp.get_episodes_to_run()
        traj_id = f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}"
        storage = FileStorage(tmp_dir)
        storage.write_episode_status(
            traj_id,
            EpisodeStatus(
                status="QUEUED",
                task_id=episodes[0].config.task_config.task_id,
                episode_id=episodes[0].config.id,
                started_at=time.time(),
                last_heartbeat_at=None,
                current_step=0,
            ),
        )
        swept = sweep_stale_statuses(storage, step_timeout_s=60.0, cancel_grace_s=60.0, orphan_threshold_s=3600.0)
        assert swept == []
        # In-flight (QUEUED) is never returned by resume.
        exp.resume = True
        assert exp.get_episodes_to_run() == []

    def test_stale_sweep_keeps_fresh_running(self, tmp_dir, mock_agent_config):
        """A RUNNING entry with a fresh heartbeat is left alone."""
        benchmark = _make_benchmark(1)
        exp = Experiment(
            name="test_fresh_running",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=benchmark,
        )
        episodes = exp.get_episodes_to_run()
        traj_id = f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}"
        storage = FileStorage(tmp_dir)
        storage.write_episode_status(
            traj_id,
            EpisodeStatus(
                status="RUNNING",
                task_id=episodes[0].config.task_config.task_id,
                episode_id=episodes[0].config.id,
                started_at=time.time(),
                last_heartbeat_at=time.time(),
                current_step=1,
            ),
        )
        swept = sweep_stale_statuses(storage, step_timeout_s=60.0, cancel_grace_s=60.0, orphan_threshold_s=3600.0)
        assert swept == []
        # Fresh RUNNING is never returned, even with resume=True.
        exp.resume = True
        assert exp.get_episodes_to_run() == []

    def test_resume_returns_unstarted_and_failed_skipping_completed(self, tmp_dir, mock_agent_config):
        """resume=True returns missing-status + retriable failures, skipping COMPLETED."""
        benchmark = _make_benchmark(3)
        exp = Experiment(
            name="test_resume_mixed",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=benchmark,
        )
        episodes = exp.get_episodes_to_run()
        storage = FileStorage(tmp_dir)
        # Episode 0: COMPLETED — should NOT be returned
        storage.write_episode_status(
            f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}",
            EpisodeStatus(
                status="COMPLETED",
                task_id=episodes[0].config.task_config.task_id,
                episode_id=episodes[0].config.id,
                started_at=time.time(),
                ended_at=time.time(),
                last_heartbeat_at=time.time(),
                reward=1.0,
            ),
        )
        # Episode 1: FAILED — SHOULD be returned (resume covers retriable failures)
        storage.write_episode_status(
            f"{episodes[1].config.task_config.task_id}_ep{episodes[1].config.id}",
            EpisodeStatus(
                status="FAILED",
                task_id=episodes[1].config.task_config.task_id,
                episode_id=episodes[1].config.id,
                started_at=time.time(),
            ),
        )
        # Episode 2: no status.json — SHOULD be returned (never started)

        exp.resume = True
        resumed = exp.get_episodes_to_run()
        ids = {ep.config.id for ep in resumed}
        assert ids == {episodes[1].config.id, episodes[2].config.id}

    def test_resume_skips_max_steps_reached(self, tmp_dir, mock_agent_config):
        """resume=True does NOT re-run MAX_STEPS_REACHED — it's a legitimate outcome."""
        benchmark = _make_benchmark(2)
        exp = Experiment(
            name="test_max_steps_skip",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark=benchmark,
        )
        episodes = exp.get_episodes_to_run()
        storage = FileStorage(tmp_dir)
        # Episode 0: MAX_STEPS_REACHED → skipped on resume
        storage.write_episode_status(
            f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}",
            EpisodeStatus(
                status="MAX_STEPS_REACHED",
                task_id=episodes[0].config.task_config.task_id,
                episode_id=episodes[0].config.id,
                started_at=time.time(),
                ended_at=time.time(),
                last_heartbeat_at=time.time(),
                reward=0.0,
            ),
        )
        # Episode 1: missing → included
        exp.resume = True
        resumed = exp.get_episodes_to_run()
        assert {ep.config.id for ep in resumed} == {episodes[1].config.id}
