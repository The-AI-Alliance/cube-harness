import json
import logging
from pathlib import Path
from typing import Self

from pydantic import Field

from agentlab2.agent import AgentConfig
from agentlab2.benchmark import Benchmark
from agentlab2.core import Trajectory, TypedBaseModel
from agentlab2.episode import Episode

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
            episode.save_config()
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

        Returns:
            ExpResult with trajectories from relaunched episodes
        """
        config_dir = self.output_dir / "episode_configs"
        if not config_dir.exists():
            logger.warning(f"No episode configs found in {config_dir}")
            return ExpResult(
                exp_id=f"{self.name}_relaunch",
                tasks_num=0,
                config=self.config,
            )

        # Find all failed episodes (episodes with configs but no successful trajectory)
        traj_dir = self.output_dir / "trajectories"
        successful_task_ids = set()
        if traj_dir.exists():
            for traj_file in traj_dir.glob("*.jsonl"):
                # Extract task_id from filename: run{id}_task_{task_id}.jsonl
                parts = traj_file.stem.split("_task_")
                if len(parts) == 2:
                    task_id = parts[1]
                    # Check if trajectory completed successfully (has final env step without error)
                    try:
                        with open(traj_file) as f:
                            lines = f.readlines()
                            if lines:
                                # Check last line for successful completion
                                last_line = json.loads(lines[-1])
                                # Success: last step is EnvironmentOutput with done=True and no error
                                if (
                                    last_line.get("_type") == "agentlab2.core.EnvironmentOutput"
                                    and not last_line.get("error")
                                    and last_line.get("done")
                                ):
                                    successful_task_ids.add(task_id)
                    except Exception:
                        pass

        # Load and relaunch failed episodes
        failed_episodes = []
        for config_file in config_dir.glob("episode_*_task_*.json"):
            # Extract task_id from filename
            parts = config_file.stem.split("_task_")
            if len(parts) == 2:
                task_id = parts[1]
                if task_id not in successful_task_ids:
                    try:
                        episode = Episode.load_config(config_file, self.benchmark)
                        failed_episodes.append(episode)
                    except Exception as e:
                        logger.exception(f"Failed to load episode config {config_file}: {e}")

        if not failed_episodes:
            logger.info("No failed episodes to relaunch")
            return ExpResult(
                exp_id=f"{self.name}_relaunch",
                tasks_num=0,
                config=self.config,
            )

        logger.info(f"Relaunching {len(failed_episodes)} failed episodes")
        self.benchmark.setup()
        try:
            trajectories = []
            failures = {}
            for episode in failed_episodes:
                try:
                    trajectory = episode.run()
                    trajectories.append(trajectory)
                except Exception as e:
                    logger.exception(f"Episode {episode.id} (task {episode.task_id}) failed again: {e}")
                    failures[episode.task_id] = str(e)

            results = ExpResult(
                exp_id=f"{self.name}_relaunch",
                tasks_num=len(failed_episodes),
                trajectories={traj.metadata["task_id"]: traj for traj in trajectories},
                failures=failures,
                config=self.config,
            )
            self.print_stats(results)
            return results
        finally:
            self.benchmark.close()