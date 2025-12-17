import json
import logging
from pathlib import Path

from termcolor import colored

from agentlab2.agent import AgentConfig
from agentlab2.core import AgentOutput, EnvironmentOutput, Trajectory
from agentlab2.environment import EnvConfig

logger = logging.getLogger(__name__)

MAX_STEPS = 1000  # System-wide upper limit on steps


class Episode:
    """Manages the execution of an agent on a specific task in an environment."""

    def __init__(
        self,
        id: int,
        output_dir: Path,
        agent_config: AgentConfig,
        env_config: EnvConfig,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.id = id
        self.output_dir = output_dir
        self.agent_config = agent_config
        self.task_id = env_config.task.id
        self.env_config = env_config
        self.max_steps = max_steps
        self._output_name = ""

    def run(self) -> Trajectory:
        """
        Main loop to run the agent on a single specific task.

        Returns:
            Trajectory containing the full history of the run.
        """
        env = self.env_config.make()
        agent = self.agent_config.make(env.action_set)
        try:
            env_output = env.setup()
            logger.info(colored(f"Initial env output: {env_output}", "blue"))
            trajectory = Trajectory(steps=[env_output], metadata={"task_id": self.task_id})
            self.save_trajectory(trajectory)
            turns = 0
            while not env_output.done and turns < self.max_steps:
                # Agent step
                agent_output = agent.step(env_output.obs)
                logger.info(colored(f"Turn {turns} Agent output: {agent_output}", "magenta"))
                trajectory.append(agent_output)
                self.save_step(agent_output)

                # Environment step
                env_output = env.step(agent_output.actions)
                logger.info(colored(f"Turn {turns} Env output: {env_output}", "blue"))
                trajectory.append(env_output)
                self.save_step(env_output)

                turns += 1
        except Exception as e:
            logger.exception(f"Error during agent run: {e}")
            raise e
        finally:
            env.close()
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
