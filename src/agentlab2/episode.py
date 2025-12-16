import json
import logging
import time
from pathlib import Path

from termcolor import colored

from agentlab2.agent import AgentConfig
from agentlab2.core import Trajectory, TrajectoryStep
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
            start_time = time.time()
            env_output = env.setup()
            start_step = TrajectoryStep(output=env_output, start_time=start_time, end_time=time.time())
            trajectory = Trajectory(steps=[start_step], metadata={"task_id": self.task_id}, start_time=start_time)
            self.save_trajectory(trajectory)
            logger.info(colored(f"Start step: {start_step}", "blue"))
            turns = 0
            while not env_output.done and turns < self.max_steps:
                # Agent step
                ts = time.time()
                agent_output = agent.step(env_output.obs)
                logger.info(colored(f"Turn {turns} Agent output: {agent_output}", "magenta"))
                agent_step = TrajectoryStep(output=agent_output, start_time=ts, end_time=time.time())
                trajectory.steps.append(agent_step)
                self.save_step(agent_step)

                # Environment step
                env_ts = time.time()
                env_output = env.step(agent_output.actions)
                logger.info(colored(f"Turn {turns} Env output: {env_output}", "blue"))
                env_step = TrajectoryStep(output=env_output, start_time=env_ts, end_time=time.time())
                trajectory.steps.append(env_step)
                self.save_step(env_step)

                turns += 1
            trajectory.end_time = time.time()
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
        # save initial steps
        for step in trajectory.steps:
            with open(f"{self._output_name}.jsonl", "a") as f:
                line = step.model_dump_json(serialize_as_any=True)
                f.write(f"{line}\n")
        logger.info(f"Saved trajectory for task {self.task_id} to {self._output_name}")

    def save_step(self, step: TrajectoryStep) -> None:
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
