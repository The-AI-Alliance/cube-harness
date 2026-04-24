import logging
import time
from pathlib import Path
from typing import Callable, Self

from cube.benchmark import Benchmark, RuntimeContext
from cube.container import ContainerBackend
from cube.core import EnvironmentOutput, StepError, TypedBaseModel
from cube.task import TaskConfig
from opentelemetry.trace import StatusCode
from termcolor import colored

from cube_harness.agent import AgentConfig
from cube_harness.core import AgentOutput, Trajectory, TrajectoryStep
from cube_harness.metrics.tracer import get_tracer
from cube_harness.storage import EPISODES_DIR, FileStorage, Storage
from cube_harness.summary import SummaryProcessor

logger = logging.getLogger(__name__)

MAX_STEPS = 1000  # System-wide upper limit on steps


class EpisodeConfig(TypedBaseModel):
    """Configuration for an episode that can be saved and reloaded."""

    id: int
    agent_config: AgentConfig
    exp_name: str
    output_dir: Path
    max_steps: int
    task_config: TaskConfig


class Episode:
    """Manages the execution of an agent on a specific task in an environment."""

    def __init__(
        self,
        id: int,
        output_dir: Path,
        agent_config: AgentConfig,
        task_config: TaskConfig,
        exp_name: str = "default",
        max_steps: int = MAX_STEPS,
        storage: Storage | None = None,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> None:
        self.config = EpisodeConfig(
            id=id,
            agent_config=agent_config,
            exp_name=exp_name,
            output_dir=output_dir,
            max_steps=max_steps,
            task_config=task_config,
        )
        self._runtime_context = runtime_context
        self._container_backend = container_backend
        self.storage = storage or FileStorage(output_dir)
        self.allow_overwrite = False

    @classmethod
    def load_episode_from_config(cls, config_path: Path, benchmark: Benchmark | None = None) -> Self:
        """
        Load episode configuration from disk and recreate the episode.

        The full TaskConfig is stored in EpisodeConfig and is self-contained
        (call task_config.make()). `benchmark` is optional — if provided, its
        runtime_context and container_backend are forwarded to the episode.

        Args:
            config_path: Path to the episode config JSON file
            benchmark: Benchmark instance (optional; used for runtime_context/container_backend)

        Returns:
            Episode instance ready to run
        """
        if config_path.name == "episode_config.json":
            output_dir = config_path.parent.parent.parent
            if config_path.parent.parent.name != EPISODES_DIR:
                raise ValueError(f"Expected episode_config.json inside {EPISODES_DIR}/, got {config_path}")
        else:
            output_dir = config_path.parent
            if output_dir.name == "episode_configs":
                output_dir = output_dir.parent
        storage = FileStorage(output_dir)
        episode_config = storage.load_episode_config(config_path)

        if benchmark is not None and not isinstance(benchmark, Benchmark):
            raise ValueError(f"benchmark must be a cube.benchmark.Benchmark instance or None, got {type(benchmark)}")
        runtime_context = benchmark._runtime_context if benchmark is not None else None
        container_backend = benchmark.container_backend if benchmark is not None else None
        return cls(
            id=episode_config.id,
            output_dir=episode_config.output_dir,
            agent_config=episode_config.agent_config,
            task_config=episode_config.task_config,
            exp_name=episode_config.exp_name,
            max_steps=episode_config.max_steps,
            storage=storage,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )

    def run(self) -> Trajectory:
        """Main loop to run the agent on a single specific task.

        Returns:
            Trajectory containing the full history of the run.
        """
        task = self.config.task_config.make(
            runtime_context=self._runtime_context, container_backend=self._container_backend
        )
        action_set = task.action_set
        step_fn = task.step
        close_fn = task.close

        def setup_fn() -> EnvironmentOutput:
            obs, info = task.reset()
            return EnvironmentOutput(obs=obs, info=info)

        agent = self.config.agent_config.make(action_set, task_id=self.config.task_config.task_id)
        return self._run_loop(setup_fn, step_fn, close_fn, agent)

    def _run_loop(
        self,
        setup_fn: Callable[[], EnvironmentOutput],
        step_fn: Callable,
        close_fn: Callable,
        agent,
    ) -> Trajectory:
        """Run loop for the agent on the task."""
        task_id = self.config.task_config.task_id
        tracer = get_tracer(self.config.exp_name)
        try:
            with tracer.episode(task_id, experiment=self.config.exp_name) as episode_span:
                start_time = time.time()
                env_output = setup_fn()
                agent_name = type(self.config.agent_config).__name__
                trajectory = Trajectory(
                    id=f"{task_id}_ep{self.config.id}",
                    steps=[TrajectoryStep(output=env_output, start_time=start_time, end_time=time.time())],
                    metadata={
                        "task_id": task_id,
                        "agent_name": agent_name,
                        **env_output.info,
                    },
                    start_time=start_time,
                )
                self.storage.save_trajectory(trajectory, allow_overwrite=self.allow_overwrite)
                ep_dir = self.storage._episode_dir(trajectory.id)
                (ep_dir / "episode_config.json").write_text(
                    self.config.model_dump_json(indent=2, serialize_as_any=True)
                )
                summary_proc = SummaryProcessor(ep_dir)
                summary_proc.on_step(0, trajectory.steps[0])
                logger.info(colored(f"Episode started — done={env_output.done} reward={env_output.reward}", "blue"))
                turns = 0
                while not env_output.done and turns < self.config.max_steps:
                    with tracer.step(f"turn_{turns}") as span:
                        ts = time.time()
                        try:
                            agent_output = agent.step(env_output.obs)
                        except Exception as e:
                            logger.exception(f"Error in agent.step() at turn {turns}: {e}")
                            agent_output = AgentOutput(error=StepError.from_exception(e))
                            agent_step = TrajectoryStep(output=agent_output, start_time=ts, end_time=time.time())
                            self.storage.save_step(agent_step, trajectory.id, len(trajectory.steps))
                            summary_proc.on_step(len(trajectory.steps), agent_step)
                            trajectory.steps.append(agent_step)
                            raise e

                        self.log_agent_output(turns, agent_output)
                        agent_step = TrajectoryStep(output=agent_output, start_time=ts, end_time=time.time())
                        self.storage.save_step(agent_step, trajectory.id, len(trajectory.steps))
                        summary_proc.on_step(len(trajectory.steps), agent_step)
                        trajectory.steps.append(agent_step)

                        if not agent_output.actions and not agent_output.error:
                            logger.info(colored("Agent returned no actions — stopping episode.", "yellow"))
                            break

                        env_ts = time.time()
                        try:
                            env_output = step_fn(agent_output.actions)
                        except Exception as e:
                            logger.exception(f"Error in step() at turn {turns}: {e}")
                            env_output = EnvironmentOutput(obs=env_output.obs, error=StepError.from_exception(e))
                            env_step = TrajectoryStep(output=env_output, start_time=env_ts, end_time=time.time())
                            self.storage.save_step(env_step, trajectory.id, len(trajectory.steps))
                            summary_proc.on_step(len(trajectory.steps), env_step)
                            trajectory.steps.append(env_step)
                            raise e

                        logger.info(
                            colored(
                                f"Turn {turns} Env output: done={env_output.done} reward={env_output.reward}", "blue"
                            )
                        )
                        env_step = TrajectoryStep(output=env_output, start_time=env_ts, end_time=time.time())
                        self.storage.save_step(env_step, trajectory.id, len(trajectory.steps))
                        summary_proc.on_step(len(trajectory.steps), env_step)
                        trajectory.steps.append(env_step)
                        span.set_attribute("done", env_output.done)
                        span.set_attribute("reward", env_output.reward)
                        turns += 1
                trajectory.end_time = time.time()
                trajectory.reward_info = {"reward": env_output.reward, "done": env_output.done, **env_output.info}
                trajectory.summary_stats = _compute_summary_stats(trajectory)
                self.storage.save_trajectory(trajectory)
                summary_proc.on_episode_complete(trajectory, self.storage)
                logger.info(colored(f"Episode completed in {turns} turns, reward: {env_output.reward}", "blue"))
                final_reward = trajectory.last_env_step().reward
                status = StatusCode.OK if final_reward > 0 else StatusCode.ERROR
                episode_span.set_status(status)
        except Exception as e:
            logger.exception(f"Error during agent run: {e}")
            raise e
        finally:
            close_fn()
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
        actions_summary = [a.name for a in agent_output.actions] if agent_output.actions else []
        logger.info(colored(f"Turn {turns} Agent output: actions={actions_summary}", "magenta"))


def _compute_summary_stats(traj: Trajectory) -> dict:
    n_env_steps = 0
    n_agent_steps = 0
    total_actions = 0
    total_llm_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    cache_creation_tokens = 0
    cost = 0.0

    for step in traj.steps:
        if isinstance(step.output, EnvironmentOutput):
            n_env_steps += 1
        elif isinstance(step.output, AgentOutput):
            n_agent_steps += 1
            total_actions += len(step.output.actions)
            total_llm_calls += len(step.output.llm_calls)
            for llm_call in step.output.llm_calls:
                if llm_call.usage:
                    prompt_tokens += llm_call.usage.prompt_tokens
                    completion_tokens += llm_call.usage.completion_tokens
                    cached_tokens += llm_call.usage.cached_tokens
                    cache_creation_tokens += llm_call.usage.cache_creation_tokens
                    cost += llm_call.usage.cost

    duration = None
    if traj.start_time is not None and traj.end_time is not None:
        duration = traj.end_time - traj.start_time

    final_reward = 0.0
    if traj.reward_info:
        final_reward = traj.reward_info.get("reward", 0.0)
    else:
        for step in reversed(traj.steps):
            if isinstance(step.output, EnvironmentOutput):
                final_reward = step.output.reward
                break

    return {
        "n_env_steps": n_env_steps,
        "n_agent_steps": n_agent_steps,
        "total_actions": total_actions,
        "total_llm_calls": total_llm_calls,
        "duration": duration,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cost": cost,
        "final_reward": final_reward,
    }
