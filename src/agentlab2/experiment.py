import json
import logging
from pathlib import Path
from typing import Self

from pydantic import Field

from agentlab2.agent import AgentConfig
from agentlab2.benchmark import Benchmark
from agentlab2.core import AgentOutput, EnvironmentOutput, Trajectory, TypedBaseModel
from agentlab2.episode import Episode
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
    benchmark: Benchmark

    @property
    def config(self) -> dict:
        return self.model_dump(serialize_as_any=True)

    def create_episodes(self):
        episodes = [
            Episode(
                id=i,
                output_dir=self.output_dir,
                agent_config=self.agent_config,
                env_config=env_config,
                exp_name=self.name,
            )
            for i, env_config in enumerate(self.benchmark.env_configs())
        ]
        # Save episode configs to disk for resumption
        for episode in episodes:
            episode.storage.save_episode_config(episode.config)
        logger.info(f"Prepared {len(episodes)} episodes for experiment '{self.name}'")
        return episodes

    def save_config(self) -> None:
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config_path = output_path / "experiment_config.json"
        with open(config_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
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

    def _load_successful_task_ids(self, storage: FileStorage) -> set[str]:
        """Load task IDs for episodes that completed successfully.

        Args:
            storage: FileStorage instance to load trajectories from.

        Returns:
            Set of task IDs that have successful trajectories.
        """
        successful_task_ids = set()
        traj_dir = self.output_dir / "trajectories"
        if traj_dir.exists():
            for metadata_file in traj_dir.glob("*.metadata.json"):
                trajectory_id = metadata_file.stem.replace(".metadata", "")
                try:
                    trajectory = storage.load_trajectory(trajectory_id)
                    # Check if trajectory completed successfully
                    # A trajectory is successful if:
                    # 1. The last env step has done=True
                    # 2. There are no errors in any step
                    last_env_step = trajectory.last_env_step()
                    has_error = False
                    # Check all steps for errors
                    for step in trajectory.steps:
                        if isinstance(step.output, EnvironmentOutput) and step.output.error:
                            has_error = True
                            break
                        if isinstance(step.output, AgentOutput) and step.output.error:
                            has_error = True
                            break

                    if last_env_step.done and not has_error and not last_env_step.error:
                        task_id = trajectory.metadata.get("task_id")
                        if task_id:
                            successful_task_ids.add(task_id)
                except Exception as e:
                    logger.debug(f"Failed to load trajectory {trajectory_id}: {e}")
        return successful_task_ids

    def _load_started_task_ids(self, storage: FileStorage) -> set[str]:
        """Load task IDs for episodes that have been started.

        Args:
            storage: FileStorage instance to load trajectories from.

        Returns:
            Set of task IDs that have trajectories (started at least once).
        """
        started_task_ids = set()
        traj_dir = self.output_dir / "trajectories"
        if traj_dir.exists():
            for metadata_file in traj_dir.glob("*.metadata.json"):
                trajectory_id = metadata_file.stem.replace(".metadata", "")
                try:
                    trajectory = storage.load_trajectory(trajectory_id)
                    task_id = trajectory.metadata.get("task_id")
                    if task_id:
                        started_task_ids.add(task_id)
                except Exception as e:
                    logger.debug(f"Failed to load trajectory {trajectory_id}: {e}")
        return started_task_ids

    def _find_episodes_to_relaunch(
        self, config_files: list[Path], filter_task_ids: set[str], include: bool = True
    ) -> list[Episode]:
        """Find episodes to relaunch based on task ID filter.

        Args:
            config_files: List of episode config file paths.
            filter_task_ids: Set of task IDs to filter by.
            include: If True, include episodes with task_ids in filter_task_ids.
                    If False, exclude episodes with task_ids in filter_task_ids.

        Returns:
            List of Episode objects to relaunch.
        """
        episodes = []
        for config_file in config_files:
            parsed = self._parse_episode_config_filename(config_file)
            if parsed:
                episode_id, task_id = parsed
                should_include = task_id in filter_task_ids if include else task_id not in filter_task_ids
                if should_include:
                    try:
                        episode = Episode.load_episode_from_config(config_file, self.benchmark)
                        episodes.append(episode)
                    except Exception as e:
                        logger.exception(f"Failed to load episode config {config_file}: {e}")
            else:
                logger.warning(f"Could not parse task_id from config filename: {config_file.name}")
        return episodes

    def _execute_episodes(self, episodes: list[Episode], exp_id_suffix: str) -> ExpResult:
        """Execute a list of episodes and return results.

        Args:
            episodes: List of Episode objects to execute.
            exp_id_suffix: Suffix to append to experiment name for the result exp_id.

        Returns:
            ExpResult containing trajectories and failures from executed episodes.
        """
        if not episodes:
            logger.info(f"No episodes to relaunch for {exp_id_suffix}")
            return ExpResult(
                exp_id=f"{self.name}_{exp_id_suffix}",
                tasks_num=0,
                config=self.config,
            )

        logger.info(f"Relaunching {len(episodes)} episodes ({exp_id_suffix})")
        self.benchmark.setup()
        try:
            trajectories = []
            failures = {}
            for episode in episodes:
                try:
                    trajectory = episode.run()
                    trajectories.append(trajectory)
                except Exception as e:
                    logger.exception(f"Episode {episode.config.id} (task {episode.config.task_id}) failed: {e}")
                    failures[episode.config.task_id] = str(e)

            results = ExpResult(
                exp_id=f"{self.name}_{exp_id_suffix}",
                tasks_num=len(episodes),
                trajectories={traj.metadata["task_id"]: traj for traj in trajectories},
                failures=failures,
                config=self.config,
            )
            self.print_stats(results)
            return results
        finally:
            self.benchmark.close()

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

    def relaunch_failed_episodes(self) -> ExpResult:
        """
        Relaunch all failed episodes from this experiment.
        Failed episodes are those that started but did not complete successfully.

        Returns:
            ExpResult with trajectories from relaunched episodes
        """
        storage = FileStorage(self.output_dir)
        config_files = storage.list_episode_configs()

        if not config_files:
            logger.warning(f"No episode configs found in {self.output_dir / 'episode_configs'}")
            return ExpResult(
                exp_id=f"{self.name}_relaunch",
                tasks_num=0,
                config=self.config,
            )

        # Find all successful trajectories
        successful_task_ids = self._load_successful_task_ids(storage)

        # Load and relaunch failed episodes (have configs and trajectories but not successful)
        failed_episodes = self._find_episodes_to_relaunch(config_files, successful_task_ids, include=False)

        return self._execute_episodes(failed_episodes, "relaunch")

    def relaunch_unstarted_episodes(self) -> ExpResult:
        """
        Relaunch all episodes that were never started (have configs but no trajectory files).

        Returns:
            ExpResult with trajectories from relaunched episodes
        """
        storage = FileStorage(self.output_dir)
        config_files = storage.list_episode_configs()

        if not config_files:
            logger.warning(f"No episode configs found in {self.output_dir / 'episode_configs'}")
            return ExpResult(
                exp_id=f"{self.name}_relaunch_unstarted",
                tasks_num=0,
                config=self.config,
            )

        # Find all tasks that have trajectories (started at least once)
        started_task_ids = self._load_started_task_ids(storage)

        # Load and relaunch unstarted episodes (have configs but no trajectories)
        unstarted_episodes = self._find_episodes_to_relaunch(config_files, started_task_ids, include=False)

        return self._execute_episodes(unstarted_episodes, "relaunch_unstarted")
