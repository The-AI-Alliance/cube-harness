import json
import logging
from pathlib import Path
from typing import Self

from termcolor import colored

from agentlab2.agent import AgentConfig
from agentlab2.base import TypedBaseModel
from agentlab2.core import AgentOutput, EnvironmentOutput, StepError, Trajectory
from agentlab2.environment import EnvConfig
from agentlab2.metrics.tracer import get_tracer
from agentlab2.tool import ToolConfig

logger = logging.getLogger(__name__)

MAX_STEPS = 1000  # System-wide upper limit on steps


class EpisodeConfig(TypedBaseModel):
    """Configuration for an episode that can be saved and reloaded."""

    id: int
    task_id: str
    agent_config: AgentConfig
    tool_config: ToolConfig
    exp_name: str
    output_dir: Path
    max_steps: int


class Episode:
    """Manages the execution of an agent on a specific task in an environment."""

    def __init__(
        self,
        id: int,
        output_dir: Path,
        agent_config: AgentConfig,
        env_config: EnvConfig,
        exp_name: str = "default",
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.id = id
        self.output_dir = output_dir
        self.agent_config = agent_config
        self.task_id = env_config.task.id
        self.env_config = env_config
        self.exp_name = exp_name
        self.max_steps = max_steps
        self._output_name = ""

    def save_config(self) -> None:
        """Save episode configuration to disk for later resumption."""
        config_dir = self.output_dir / "episode_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"episode_{self.id}_task_{self.task_id}.json"

        episode_config = EpisodeConfig(
            id=self.id,
            task_id=self.task_id,
            agent_config=self.agent_config,
            tool_config=self.env_config.tool_config,
            exp_name=self.exp_name,
            output_dir=self.output_dir,
            max_steps=self.max_steps,
        )

        with open(config_path, "w") as f:
            f.write(episode_config.model_dump_json(indent=2))
        logger.info(f"Saved episode config to {config_path}")

    @classmethod
    def load_config(cls, config_path: Path, benchmark) -> Self:
        """
        Load episode configuration from disk and recreate the episode.

        Args:
            config_path: Path to the episode config JSON file
            benchmark: Benchmark instance to recreate the task

        Returns:
            Episode instance ready to run
        """
        with open(config_path) as f:
            data = json.load(f)

        episode_config = EpisodeConfig.model_validate(data)

        # Find the task in the benchmark
        tasks = benchmark.load_tasks()
        task = None
        for t in tasks:
            if t.id == episode_config.task_id:
                task = t
                break

        if task is None:
            raise ValueError(f"Task {episode_config.task_id} not found in benchmark")

        # Recreate EnvConfig
        env_config = EnvConfig(task=task, tool_config=episode_config.tool_config)

        # Create and return Episode
        return cls(
            id=episode_config.id,
            output_dir=episode_config.output_dir,
            agent_config=episode_config.agent_config,
            env_config=env_config,
            exp_name=episode_config.exp_name,
            max_steps=episode_config.max_steps,
        )

    def relaunch(self) -> Trajectory:
        """
        Relaunch a failed episode. This is the same as run() but provided for clarity.

        Returns:
            Trajectory containing the full history of the run.
        """
        return self.run()

    def run(self) -> Trajectory:
        """
        Main loop to run the agent on a single specific task.

        Returns:
            Trajectory containing the full history of the run.
        """
        tracer = get_tracer(self.exp_name)
        env = self.env_config.make()
        agent = self.agent_config.make(env.action_set)
        try:
            with tracer.episode(self.task_id, experiment=self.exp_name):
                env_output = env.setup()
                logger.info(colored(f"Initial env output: {env_output}", "blue"))
                trajectory = Trajectory(steps=[env_output], metadata={"task_id": self.task_id})
                self.save_trajectory(trajectory)
                turns = 0
                while not env_output.done and turns < self.max_steps:
                    with tracer.step(f"turn_{turns}") as span:
                        # Agent step
                        try:
                            agent_output = agent.step(env_output.obs)
                        except Exception as e:
                            logger.exception(f"Error in agent.step() at turn {turns}: {e}")
                            agent_output = AgentOutput(error=StepError.from_exception(e))
                            trajectory.append(agent_output)
                            self.save_step(agent_output)
                            raise e

                        logger.info(colored(f"Turn {turns} Agent output: {agent_output}", "magenta"))
                        trajectory.append(agent_output)
                        self.save_step(agent_output)

                        # Environment step
                        try:
                            env_output = env.step(agent_output.actions)
                        except Exception as e:
                            logger.exception(f"Error in env.step() at turn {turns}: {e}")
                            # Use previous observation since env.step() failed
                            prev_obs = trajectory.last_env_step().obs if trajectory.steps else Observation(contents=[])
                            env_output = EnvironmentOutput(
                                obs=prev_obs, error=StepError.from_exception(e)
                            )
                            trajectory.append(env_output)
                            self.save_step(env_output)
                            raise e

                        logger.info(colored(f"Turn {turns} Env output: {env_output}", "blue"))
                        trajectory.append(env_output)
                        self.save_step(env_output)

                        span.set_attribute("agent_output", agent_output.model_dump_json())
                        turns += 1
        except Exception as e:
            logger.exception(f"Error during agent run: {e}")
            raise e
        finally:
            env.close()
            tracer.shutdown()
        return trajectory

    def save_trajectory(self, trajectory: Trajectory) -> None:
        """Save the trajectory to the output directory."""
        # TODO: Replace with tracing implementation
        traj_dir = self.output_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        self._output_name = traj_dir / f"run{self.id}_task_{self.task_id}"
        with open(f"{self._output_name}.metadata.json", "w") as f:
            f.write(json.dumps(trajectory.metadata, indent=2))
        with open(f"{self._output_name}.jsonl", "a") as f:
            pass  # Create empty file for appending steps later
        logger.info(f"Saved trajectory for task {self.task_id} to {self._output_name}")

    def save_step(self, step: AgentOutput | EnvironmentOutput) -> None:
        """Append a single step to the trajectory JSONL file."""
        # TODO: Replace with tracing implementation
        if not self._output_name:
            raise ValueError("Trajectory path not set. Call save_trajectory first.")
        try:
            with open(f"{self._output_name}.jsonl", "a") as f:
                line = step.model_dump_json(serialize_as_any=True)
                f.write(f"{line}\n")
        except Exception as e:
            logger.exception(f"Error saving step to trajectory {self._output_name}: {e}")
            raise e
