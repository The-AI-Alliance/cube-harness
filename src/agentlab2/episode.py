import logging
import time
from pathlib import Path
from typing import Self

from opentelemetry.trace import Span
from termcolor import colored

from agentlab2.agent import AgentConfig
from agentlab2.base import TypedBaseModel
from agentlab2.core import AgentOutput, EnvironmentOutput, StepError, Trajectory, TrajectoryStep
from agentlab2.environment import EnvConfig
from agentlab2.metrics.tracer import get_tracer
from agentlab2.storage import FileStorage, Storage
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

    def save_config(self) -> None:
        """Save episode configuration to disk for later resumption."""
        episode_config = EpisodeConfig(
            id=self.id,
            task_id=self.task_id,
            agent_config=self.agent_config,
            tool_config=self.env_config.tool_config,
            exp_name=self.exp_name,
            output_dir=self.output_dir,
            max_steps=self.max_steps,
        )
        self.storage.save_episode_config(episode_config)

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
        from agentlab2.storage import FileStorage

        # Infer output_dir from config_path structure: {output_dir}/episode_configs/episode_*.json
        output_dir = config_path.parent.parent
        storage = FileStorage(output_dir)
        episode_config = storage.load_episode_config(config_path)

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

    def _record_step_attributes(
        self,
        span: Span,
        agent_output: AgentOutput,
        env_output: EnvironmentOutput,
    ) -> None:
        span.set_attribute("agent_output", agent_output.model_dump_json())
        span.set_attribute("env_output", env_output.model_dump_json())
        span.set_attribute("done", env_output.done)
        span.set_attribute("reward", env_output.reward)

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
                        try:
                            agent_output = agent.step(env_output.obs)
                        except Exception as e:
                            logger.exception(f"Error in agent.step() at turn {turns}: {e}")
                            agent_output = AgentOutput(error=StepError.from_exception(e))
                            agent_step = TrajectoryStep(output=agent_output, start_time=ts, end_time=time.time())
                            self.storage.save_step(agent_step, trajectory.id, len(trajectory.steps))
                            trajectory.steps.append(agent_step)
                            raise e

                        self.log_agent_output(turns, agent_output)
                        agent_step = TrajectoryStep(output=agent_output, start_time=ts, end_time=time.time())
                        self.storage.save_step(agent_step, trajectory.id, len(trajectory.steps))
                        trajectory.steps.append(agent_step)

                        # Environment step
                        env_ts = time.time()
                        try:
                            env_output = env.step(agent_output.actions)
                        except Exception as e:
                            logger.exception(f"Error in env.step() at turn {turns}: {e}")
                            # Use current env_output.obs since env.step() failed (env_output still has previous value)
                            env_output = EnvironmentOutput(obs=env_output.obs, error=StepError.from_exception(e))
                            env_step = TrajectoryStep(output=env_output, start_time=env_ts, end_time=time.time())
                            self.storage.save_step(env_step, trajectory.id, len(trajectory.steps))
                            trajectory.steps.append(env_step)
                            raise e

                        logger.info(colored(f"Turn {turns} Env output: {env_output}", "blue"))
                        env_step = TrajectoryStep(output=env_output, start_time=env_ts, end_time=time.time())
                        self.storage.save_step(env_step, trajectory.id, len(trajectory.steps))
                        trajectory.steps.append(env_step)
                        self._record_step_attributes(span, agent_output, env_output)
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
