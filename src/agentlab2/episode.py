import json
import logging
from pathlib import Path

from termcolor import colored

from agentlab2.agent import AgentConfig
from agentlab2.core import Action, AgentOutput, EnvironmentOutput, Trajectory
from agentlab2.environment import STOP_ACTION, EnvConfig
from agentlab2.metrics.tracer import get_tracer

logger = logging.getLogger(__name__)

MAX_STEPS = 50  # System-wide fallback limit (agent's max_actions takes priority)


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
                # logger.info(colored(f"Initial env output: {env_output}", "blue"))
                trajectory = Trajectory(steps=[env_output], metadata={"task_id": self.task_id})
                self.save_trajectory(trajectory)
                turns = 0
                while not env_output.done and turns < self.max_steps:
                    # Check if agent has reached its max_actions limit
                    if hasattr(agent, "max_actions_reached") and agent.max_actions_reached():
                        logger.info(f"Agent reached max_actions limit at turn {turns}, forcing stop")
                        agent_output = AgentOutput(actions=[Action(name=STOP_ACTION.name, arguments={})])
                        trajectory.append(agent_output)
                        self.save_step(agent_output)
                        env_output = env.step(agent_output.actions)
                        trajectory.append(env_output)
                        self.save_step(env_output)
                        break

                    with tracer.step(f"turn_{turns}") as span:
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
