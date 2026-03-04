import json
import logging
import warnings
from pathlib import Path
from typing import Self

from cube.benchmark import Benchmark as CubeBenchmark
from cube.core import EnvironmentOutput, TypedBaseModel
from pydantic import Field

from agentlab2.agent import AgentConfig
from agentlab2.core import AgentOutput, Trajectory
from agentlab2.episode import MAX_STEPS, Episode
from agentlab2.legacy import Benchmark as AL2Benchmark
from agentlab2.storage import FileStorage

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
    benchmark: AL2Benchmark | CubeBenchmark
    resume: bool = False
    retry_failed: bool = False
    max_steps: int = MAX_STEPS

    @property
    def config(self) -> dict:
        return self.model_dump(serialize_as_any=True)

    def get_episodes_to_run(self) -> list[Episode]:
        """Get episodes to run based on resume/retry_failed flags.

        - Neither flag set: creates all episodes from scratch.
        - resume=True: returns only unstarted episodes (have configs but no trajectory files).
        - retry_failed=True: returns only failed episodes (started but not successful).
        - Both flags: returns unstarted + failed episodes.
        """
        if not self.resume and not self.retry_failed:
            return self._create_all_episodes()

        storage = FileStorage(self.output_dir)
        config_files = storage.list_episode_configs()
        if not config_files:
            logger.warning(f"No episode configs found in {self.output_dir / 'episode_configs'}, creating from scratch")
            return self._create_all_episodes()

        started_ids = self._load_started_trajectory_ids()
        episodes: list[Episode] = []

        if self.resume:
            unstarted = self._find_episodes_to_relaunch(config_files, started_ids, include=False)
            logger.info(f"Resuming: {len(unstarted)} unstarted episodes (out of {len(config_files)} total)")
            episodes.extend(unstarted)

        if self.retry_failed:
            successful_ids = self._load_successful_trajectory_ids(storage)
            failed_ids = started_ids - successful_ids
            failed = self._find_episodes_to_relaunch(config_files, failed_ids, include=True)
            for episode in failed:
                episode.allow_overwrite = True  # Allow overwriting existing trajectory since this is a retried episode
            logger.info(f"Retrying: {len(failed)} failed episodes (out of {len(config_files)} total)")
            episodes.extend(failed)

        return episodes

    def _create_all_episodes(self) -> list[Episode]:
        """Create all episodes from scratch and save their configs to disk."""
        if isinstance(self.benchmark, CubeBenchmark):
            task_configs = list(self.benchmark.get_task_configs())
            episodes = [
                Episode(
                    id=i,
                    output_dir=self.output_dir,
                    agent_config=self.agent_config,
                    task_config=tc,
                    exp_name=self.name,
                    max_steps=self.max_steps,
                )
                for i, tc in enumerate(task_configs)
            ]
        elif isinstance(self.benchmark, AL2Benchmark):
            warnings.warn(
                f"{type(self.benchmark).__name__} does not implement get_task_configs(). "
                "Falling back to deprecated env_configs(). "
                "Implement get_task_configs() to use the cube.Task path.",
                DeprecationWarning,
                stacklevel=2,
            )
            episodes = [
                Episode(
                    id=i,
                    output_dir=self.output_dir,
                    agent_config=self.agent_config,
                    env_config=ec,
                    exp_name=self.name,
                    max_steps=self.max_steps,
                )
                for i, ec in enumerate(self.benchmark.env_configs())
            ]
        else:
            raise ValueError(f"Unsupported benchmark type: {type(self.benchmark)}")
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

    def _parse_episode_config_filename(self, config_file: Path) -> tuple[int, str] | None:
        """
        Parse episode config filename to extract episode id and task_id.

        Args:
            config_file: Path to the episode config file (e.g., episode_0_task_my_task.json)

        Returns:
            Tuple of (episode_id, task_id) if parsing succeeds, None otherwise.
            The episode_id is extracted from the filename prefix.
        """
        # Extract task_id from filename: episode_{id}_task_{task_id}.json
        # Use split with maxsplit=1 to handle task_ids that contain "_task_"
        # Split from left so we only split on the FIRST "_task_" (the delimiter)
        parts = config_file.stem.split("_task_", 1)
        if len(parts) == 2:
            try:
                # Extract episode id from "episode_{id}"
                episode_id_str = parts[0].replace("episode_", "")
                episode_id = int(episode_id_str)
                task_id = parts[1]
                return (episode_id, task_id)
            except (ValueError, AttributeError):
                return None
        return None

    def _is_trajectory_successful(self, trajectory: Trajectory) -> bool:
        """Check if a trajectory completed successfully.

        A trajectory is successful if the last env step has done=True and no steps contain errors.
        """
        last_env_step = trajectory.last_env_step()
        for step in trajectory.steps:
            if isinstance(step.output, (EnvironmentOutput, AgentOutput)) and step.output.error:
                return False
        return last_env_step.done

    def _load_successful_trajectory_ids(self, storage: FileStorage) -> set[str]:
        """Load trajectory IDs for episodes that completed successfully.

        Args:
            storage: FileStorage instance to load trajectories from.

        Returns:
            Set of trajectory IDs that completed successfully.
        """
        successful = set()
        traj_dir = self.output_dir / "trajectories"
        if traj_dir.exists():
            for metadata_file in traj_dir.glob("*.metadata.json"):
                trajectory_id = metadata_file.stem.replace(".metadata", "")
                try:
                    trajectory = storage.load_trajectory(trajectory_id)
                    if self._is_trajectory_successful(trajectory):
                        successful.add(trajectory_id)
                except Exception as e:
                    logger.debug(f"Failed to load trajectory {trajectory_id}: {e}")
        return successful

    def _load_started_trajectory_ids(self) -> set[str]:
        """Load trajectory IDs for episodes that have been started.

        Returns:
            Set of trajectory IDs that have metadata files on disk.
        """
        started = set()
        traj_dir = self.output_dir / "trajectories"
        if traj_dir.exists():
            for metadata_file in traj_dir.glob("*.metadata.json"):
                started.add(metadata_file.stem.replace(".metadata", ""))
        return started

    def _find_episodes_to_relaunch(
        self, config_files: list[Path], filter_trajectory_ids: set[str], include: bool = True
    ) -> list[Episode]:
        """Find episodes to relaunch based on trajectory ID filter.

        Args:
            config_files: List of episode config file paths.
            filter_trajectory_ids: Set of trajectory IDs to filter by.
            include: If True, include episodes whose trajectory ID is in the set.
                    If False, exclude episodes whose trajectory ID is in the set.

        Returns:
            List of Episode objects to relaunch.
        """
        episodes = []
        for config_file in config_files:
            parsed = self._parse_episode_config_filename(config_file)
            if parsed:
                episode_id, task_id = parsed
                trajectory_id = f"{task_id}_ep{episode_id}"
                should_include = (
                    trajectory_id in filter_trajectory_ids if include else trajectory_id not in filter_trajectory_ids
                )
                if should_include:
                    try:
                        episode = Episode.load_episode_from_config(config_file, self.benchmark)
                        episodes.append(episode)
                    except Exception as e:
                        logger.exception(f"Failed to load episode config {config_file}: {e}")
            else:
                logger.warning(f"Could not parse task_id from config filename: {config_file.name}")
        return episodes

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
