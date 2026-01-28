import logging
import time
from pathlib import Path

from termcolor import colored

from agentlab2.agent import AgentConfig
from agentlab2.core import AgentOutput, Trajectory, TrajectoryStep
from agentlab2.environment import EnvConfig
from agentlab2.metrics.tracer import get_tracer
from agentlab2.storage import FileStorage, Storage

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
        exp_name: str = "default",
        max_steps: int = MAX_STEPS,
        storage: Storage | None = None,
    ) -> None:
        self.id = id
        self.output_dir = output_dir
        self.agent_config = agent_config
        self.task_id = env_config.task.id
        self.env_config = env_config
        self.exp_name = exp_name
        self.max_steps = max_steps
        self.storage = storage or FileStorage(output_dir)

    def run(self) -> Trajectory:
        """Main loop to run the agent on a single specific task.

        Returns:
            Trajectory containing the full history of the run.

        """
        tracer = get_tracer(self.exp_name)
        env = self.env_config.make()
        agent = self.agent_config.make(env.action_set)
        try:
            with tracer.episode(self.task_id, experiment=self.exp_name):
                start_time = time.time()
                env_output = env.setup()
                trajectory = Trajectory(
                    id=f"{self.task_id}_ep{self.id}",
                    steps=[TrajectoryStep(output=env_output, start_time=start_time, end_time=time.time())],
                    metadata={"task_id": self.task_id, **env_output.info},
                    start_time=start_time,
                )
                self.storage.save_trajectory(trajectory)
                logger.info(colored(f"Start env output: {env_output}", "blue"))
                turns = 0
                while not env_output.done and turns < self.max_steps:
                    with tracer.step(f"turn_{turns}") as span:
                        # Agent step
                        ts = time.time()
                        agent_output = agent.step(env_output.obs)
                        self.log_agent_output(turns, agent_output)
                        agent_step = TrajectoryStep(output=agent_output, start_time=ts, end_time=time.time())
                        self.storage.save_step(agent_step, trajectory.id, len(trajectory.steps))
                        trajectory.steps.append(agent_step)

                        # Environment step
                        env_ts = time.time()
                        env_output = env.step(agent_output.actions)
                        logger.info(colored(f"Turn {turns} Env output: {env_output}", "blue"))
                        env_step = TrajectoryStep(output=env_output, start_time=env_ts, end_time=time.time())
                        self.storage.save_step(env_step, trajectory.id, len(trajectory.steps))
                        trajectory.steps.append(env_step)
                        span.set_attribute("agent_output", agent_output.model_dump_json())
                        turns += 1
                trajectory.end_time = time.time()
                self.storage.save_trajectory(trajectory)  # save final trajectory with end_time
        except Exception as e:
            logger.exception(f"Error during agent run: {e}")
            raise e
        finally:
            env.close()
            tracer.shutdown()
        return trajectory

    def log_agent_output(self, turns: int, agent_output: AgentOutput) -> None:
        for llm_call in agent_output.llm_calls:
            if llm_call.output.content:
                logger.info(colored(f"Turn {turns} LLM Response: {llm_call.output.content}", "green"))
            if hasattr(llm_call.output, "reasoning_content") and llm_call.output.reasoning_content:
                logger.info(colored(f"Turn {turns} LLM Reasoning: {llm_call.output.reasoning_content}", "cyan"))
            if hasattr(llm_call.output, "thinking_blocks") and llm_call.output.thinking_blocks:
                for block in llm_call.output.thinking_blocks:
                    logger.info(colored(f"Turn {turns} LLM Thinking Block: {block}", "cyan"))
        logger.info(colored(f"Turn {turns} Agent output: {agent_output}", "magenta"))
