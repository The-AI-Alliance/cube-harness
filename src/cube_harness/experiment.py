import json
import logging
import time
from pathlib import Path
from typing import Self

from cube.benchmark import Benchmark
from cube.core import TypedBaseModel
from pydantic import Field

from cube_harness.agent import AgentConfig
from cube_harness.core import Trajectory
from cube_harness.episode import MAX_STEPS, Episode
from cube_harness.episode_status import RETRIABLE_STATUSES, EpisodeStatus
from cube_harness.storage import FileStorage

logger = logging.getLogger(__name__)


class ExpResult(TypedBaseModel):
    exp_id: str
    tasks_num: int
    config: dict = Field(default_factory=dict)
    trajectories: dict[str, Trajectory] = Field(default_factory=dict)
    failures: dict[str, str] = Field(default_factory=dict)


class Experiment(TypedBaseModel):
    name: str
    output_dir: Path
    agent_config: AgentConfig
    benchmark: Benchmark
    resume: bool = False
    retry_failed: bool = False
    max_steps: int = MAX_STEPS
    max_retries: int = 3

    @property
    def config(self) -> dict:
        return self.model_dump(serialize_as_any=True)

    def get_episodes_to_run(
        self,
        *,
        step_timeout_s: float = 1800.0,
        cancel_grace_s: float = 120.0,
        orphan_threshold_s: float = 3600.0,
    ) -> list[Episode]:
        """Get episodes to run based on resume/retry_failed flags.

        Decisions are driven by `status.json` per episode (no trajectory deserialisation).

        - Neither flag set: creates all episodes from scratch.
        - resume=True: episodes with no status.json (never started).
        - retry_failed=True: episodes with status in {FAILED, STALE, CANCELLED}, OR
          missing status.json, gated by `retry_count < max_retries`.
        - Both flags: union.

        `RUNNING` (with fresh heartbeat) is never returned. Stale `RUNNING` entries are
        first swept to `STALE` so they become eligible for retry.
        """
        if not self.resume and not self.retry_failed:
            return self._create_all_episodes()

        storage = FileStorage(self.output_dir)
        config_files = storage.list_episode_configs()
        if not config_files:
            logger.warning(f"No episode configs found in {self.output_dir}, creating from scratch")
            return self._create_all_episodes()

        # Sweep stale RUNNING entries first so they show up as STALE in retry selection.
        sweep_stale_statuses(
            storage,
            step_timeout_s=step_timeout_s,
            cancel_grace_s=cancel_grace_s,
            orphan_threshold_s=orphan_threshold_s,
        )

        statuses = storage.list_episode_statuses()
        episodes: list[Episode] = []

        for config_file in config_files:
            trajectory_id = self._trajectory_id_from_config(config_file)
            if trajectory_id is None:
                logger.warning(f"Could not parse task_id from config filename: {config_file.name}")
                continue
            status = statuses.get(trajectory_id)
            if not self._should_relaunch(status):
                continue
            try:
                episode = Episode.load_episode_from_config(config_file, self.benchmark)
            except Exception:
                logger.exception(f"Failed to load episode config {config_file}")
                continue
            # Existing trajectory (if any) will be archived on the next save_trajectory.
            episode.allow_overwrite = status is not None
            episodes.append(episode)

        logger.info(
            f"Selected {len(episodes)} episode(s) to run (out of {len(config_files)} total) "
            f"with resume={self.resume}, retry_failed={self.retry_failed}"
        )
        return episodes

    def _should_relaunch(self, status: EpisodeStatus | None) -> bool:
        # Missing status.json
        if status is None:
            # resume picks it up unconditionally; retry_failed gates by max_retries
            # but with retry_count=0 (the default) it always passes for missing status.
            return self.resume or self.retry_failed
        if status.status == "RUNNING":
            return False  # alive (sweep ran first; surviving RUNNING is fresh)
        if status.status == "COMPLETED":
            return False
        # FAILED / CANCELLED / STALE
        if not self.retry_failed:
            return False
        return status.retry_count < self.max_retries

    def _create_all_episodes(self) -> list[Episode]:
        """Create all episodes from scratch and save their configs to disk."""
        task_configs = list(self.benchmark.get_task_configs())
        episodes = [
            Episode(
                id=i,
                output_dir=self.output_dir,
                agent_config=self.agent_config,
                task_config=tc,
                exp_name=self.name,
                max_steps=self.max_steps,
                runtime_context=self.benchmark._runtime_context,
                container_backend=self.benchmark.container_backend,
            )
            for i, tc in enumerate(task_configs)
        ]
        for episode in episodes:
            episode.storage.save_episode_config(episode.config)
        logger.info(f"Prepared {len(episodes)} episodes for experiment '{self.name}'")
        return episodes

    def save_config(self) -> None:
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config_path = output_path / "experiment_config.json"
        with open(config_path, "w") as f:
            f.write(self.model_dump_json(indent=2, serialize_as_any=True))
        logger.info(f"Saved experiment config to {config_path}")

    @classmethod
    def load_config(cls, path: str) -> Self:
        """Load experiment from a JSON config file."""
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    def _trajectory_id_from_config(self, config_file: Path) -> str | None:
        """Return trajectory_id for a config file, handling both V2 and V1 filename formats."""
        if config_file.name == "episode_config.json":
            return config_file.parent.name
        parsed = self._parse_episode_config_filename(config_file)
        if parsed:
            episode_id, task_id = parsed
            return f"{task_id}_ep{episode_id}"
        return None

    def _parse_episode_config_filename(self, config_file: Path) -> tuple[int, str] | None:
        """
        Parse episode config filename to extract episode id and task_id.

        Args:
            config_file: Path to the episode config file (e.g., episode_0_task_my_task.json)

        Returns:
            Tuple of (episode_id, task_id) if parsing succeeds, None otherwise.
        """
        parts = config_file.stem.split("_task_", 1)
        if len(parts) == 2:
            try:
                episode_id_str = parts[0].replace("episode_", "")
                episode_id = int(episode_id_str)
                task_id = parts[1]
                return (episode_id, task_id)
            except (ValueError, AttributeError):
                return None
        return None

    def print_stats(self, results: ExpResult) -> None:
        if not results.trajectories:
            logger.info("No trajectories to compute stats")
            return

        total_steps = sum(len(trajectory.steps) for trajectory in results.trajectories.values())
        avg_steps = total_steps / len(results.trajectories)

        rewards = []
        for traj in results.trajectories.values():
            rewards.append(traj.last_env_step().reward)

        accuracy = sum(rewards) / len(rewards) if rewards else 0.0

        logger.info(f"Experiment '{self.name}' stats:")
        logger.info(f"  Total trajectories: {len(results.trajectories)}")
        logger.info(f"  Avg steps per trajectory: {avg_steps:.2f}")
        logger.info(f"  Accuracy (avg. final reward): {accuracy:.4f}")
        logger.info(f"  Failed tasks: {len(results.failures)}")
        logger.info(f"Saved to: {self.output_dir}")


def sweep_stale_statuses(
    storage: FileStorage,
    *,
    step_timeout_s: float,
    cancel_grace_s: float,
    orphan_threshold_s: float,
) -> list[str]:
    """Mark `RUNNING` episodes whose worker is dead as `STALE`.

    A `RUNNING` entry is stale if either:
    - `last_heartbeat_at` is set and older than `step_timeout_s + cancel_grace_s`, or
    - `last_heartbeat_at` is `None` (queued but never picked up) and `started_at` is
      older than `orphan_threshold_s`.

    Returns the list of trajectory_ids that were swept.
    """
    now = time.time()
    swept: list[str] = []
    for trajectory_id, status in storage.list_episode_statuses().items():
        if status.status != "RUNNING":
            continue
        is_stale = False
        if status.last_heartbeat_at is not None:
            if now - status.last_heartbeat_at > step_timeout_s + cancel_grace_s:
                is_stale = True
        else:
            if now - status.started_at > orphan_threshold_s:
                is_stale = True
        if not is_stale:
            continue
        status.status = "STALE"
        status.ended_at = now
        status.error_type = status.error_type or "WorkerDied"
        status.error_message = status.error_message or (
            "Worker died without writing terminal status (no fresh heartbeat)"
        )
        storage.write_episode_status(trajectory_id, status)
        swept.append(trajectory_id)
        logger.warning(f"Swept stale RUNNING -> STALE for {trajectory_id}")
    return swept


def is_retriable(status: EpisodeStatus | None, max_retries: int) -> bool:
    """Standalone helper: True if a status (or missing status) qualifies for retry."""
    if status is None:
        return True
    if status.status not in RETRIABLE_STATUSES:
        return False
    return status.retry_count < max_retries
