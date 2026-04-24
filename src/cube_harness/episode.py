import logging
import time
import warnings
from pathlib import Path
from typing import Callable, Self

from cube.benchmark import Benchmark, RuntimeContext
from cube.container import ContainerBackend
from cube.core import EnvironmentOutput, StepError, TypedBaseModel
from cube.task import EvalInfo, TaskConfig
from cube.tool import ToolConfig
from opentelemetry.trace import StatusCode
from termcolor import colored

from cube_harness.agent import AgentConfig
from cube_harness.core import AgentOutput, Trajectory, TrajectoryStep, TerminationReason
from cube_harness.legacy import Benchmark as LegacyBenchmark
from cube_harness.legacy import EnvConfig
from cube_harness.metrics.tracer import get_tracer
from cube_harness.storage import EPISODES_DIR, FileStorage, Storage, NoopStorage
from cube_harness.summary import PersistentSummaryProcessor, NoopSummaryProcessor

logger = logging.getLogger(__name__)

MAX_STEPS = 1000  # System-wide upper limit on steps


class EpisodeConfig(TypedBaseModel):
    """Configuration for an episode that can be saved and reloaded."""

    id: int
    task_id: str
    agent_config: AgentConfig
    exp_name: str
    output_dir: Path
    max_steps: int
    persist_episode: bool = True
    # New cube path:
    task_config: TaskConfig | None = None
    # Deprecated legacy path:
    tool_config: ToolConfig | None = None


class Episode:
    """Manages the execution of an agent on a specific task in an environment."""

    def __init__(
        self,
        id: int,
        output_dir: Path,
        agent_config: AgentConfig,
        task_config: TaskConfig | None = None,  # new cube path: pass full TaskConfig
        env_config: EnvConfig | None = None,  # deprecated legacy path: pass EnvConfig
        exp_name: str = "default",
        max_steps: int = MAX_STEPS,
        storage: Storage | None = None,
        persist_episode: bool = True,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> None:
        if task_config is None and env_config is None:
            raise ValueError("Provide either task_config (new) or env_config (deprecated).")
        if task_config is not None and env_config is not None:
            raise ValueError("Provide only one of task_config (new) or env_config (deprecated).")

        if env_config is not None:
            warnings.warn(
                "env_config is deprecated. Pass task_config with a cube.Task instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            effective_task_id = env_config.task.id
        else:
            assert task_config is not None  # guaranteed by earlier check
            effective_task_id = task_config.task_id

        self.config = EpisodeConfig(
            id=id,
            task_id=effective_task_id,
            agent_config=agent_config,
            exp_name=exp_name,
            output_dir=output_dir,
            max_steps=max_steps,
            task_config=task_config,
            persist_episode=persist_episode,
            tool_config=env_config.tool_config if env_config is not None else None,
        )
        self._env_config = env_config  # kept for the legacy run path
        self._runtime_context = runtime_context  # passed to task_config.make()
        self._container_backend = container_backend  # passed to task_config.make()

        if persist_episode: self.storage = storage or FileStorage(output_dir)
        else: self.storage = NoopStorage()  # in-memory no-op storage for ephemeral episodes

        if persist_episode:
            self.summary_proc_cls = PersistentSummaryProcessor
        else:
            self.summary_proc_cls = NoopSummaryProcessor

        if persist_episode and isinstance(self.storage, NoopStorage):
            raise ValueError("persist_episode=True requires a real FileStorage, got NoopStorage")

        self.persist_episode = persist_episode
        self.allow_overwrite = False

    @classmethod
    def load_episode_from_config(cls, config_path: Path, benchmark: Benchmark | LegacyBenchmark | None = None) -> Self:
        """
        Load episode configuration from disk and recreate the episode.

        For the new cube path, `benchmark` is optional — the full TaskConfig is
        stored in EpisodeConfig and is self-contained (call task_config.make()).

        For the legacy path, `benchmark` is required to map task_id → Task instance.

        Args:
            config_path: Path to the episode config JSON file
            benchmark: Benchmark instance (required for legacy path only)

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

        if episode_config.task_config is not None:
            # New cube path — fully self-contained, benchmark is optional
            if not isinstance(benchmark, Benchmark) and benchmark is not None:
                raise ValueError(
                    f"benchmark must be a cube.benchmark.Benchmark instance or None, got {type(benchmark)}"
                )
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
        else:
            # Legacy path — needs benchmark to reconstruct Task
            if not isinstance(benchmark, LegacyBenchmark) or benchmark is None:
                raise ValueError(
                    f"benchmark must be a cube_harness.legacy.Benchmark instance for legacy episodes, got {type(benchmark)}"
                )

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
            assert episode_config.tool_config is not None, "Legacy EpisodeConfig must have tool_config"
            env_config = EnvConfig(task=task, tool_config=episode_config.tool_config)

            # Create and return Episode with config stored internally
            episode = cls(
                id=episode_config.id,
                output_dir=episode_config.output_dir,
                agent_config=episode_config.agent_config,
                env_config=env_config,
                exp_name=episode_config.exp_name,
                max_steps=episode_config.max_steps,
                storage=storage,  # Allow overwriting existing trajectory since this is a resumed episode
            )
            return episode

    def run(self) -> Trajectory:
        """Main loop to run the agent on a single specific task.

        Dispatches to the cube path (task_config) or the deprecated legacy path (env_config),
        then delegates to the shared _run_loop.

        Returns:
            Trajectory containing the full history of the run.
        """
        if self.config.task_config is not None:
            task = self.config.task_config.make(
                runtime_context=self._runtime_context, container_backend=self._container_backend
            )
            action_set = task.action_set
            step_fn = task.step
            close_fn = task.close
            evaluate_fn = task.evaluate

            def setup_fn() -> EnvironmentOutput:
                obs, info = task.reset()
                return EnvironmentOutput(obs=obs, info=info)
        else:
            warnings.warn(
                "Running via env_config is deprecated. Use task_config with a cube.Task.",
                DeprecationWarning,
                stacklevel=2,
            )
            assert self._env_config is not None  # guaranteed: task_config is None means env_config was provided
            env = self._env_config.make()
            action_set = env.action_set
            step_fn = env.step
            close_fn = env.close
            setup_fn = env.setup
            evaluate_fn = env.evaluate

        agent = self.config.agent_config.make(action_set)
        return self._run_loop(setup_fn, step_fn, close_fn, evaluate_fn, agent)

    def _run_loop(
        self,
        setup_fn: Callable[[], EnvironmentOutput],
        step_fn: Callable,
        close_fn: Callable,
        evaluate_fn: Callable,
        agent,
    ) -> Trajectory:
        """Shared run loop used by both the cube path and the legacy path."""
        task_id = self.config.task_id
        tracer = get_tracer(self.config.exp_name)
        termination_reason = None
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
                if self.persist_episode:
                    ep_dir = self.storage._episode_dir(trajectory.id)
                    (ep_dir / "episode_config.json").write_text(
                        self.config.model_dump_json(indent=2, serialize_as_any=True)
                    )
                else:
                    ep_dir = None  # no directory since we're not persisting
                summary_proc = self.summary_proc_cls(ep_dir)
                summary_proc.on_step(0, trajectory.steps[0])
                logger.info(colored(f"Start env output: {env_output}", "blue"))
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
                            termination_reason = TerminationReason.AGENT_ERROR
                            break

                        self.log_agent_output(turns, agent_output)
                        agent_step = TrajectoryStep(output=agent_output, start_time=ts, end_time=time.time())
                        self.storage.save_step(agent_step, trajectory.id, len(trajectory.steps))
                        summary_proc.on_step(len(trajectory.steps), agent_step)
                        trajectory.steps.append(agent_step)

                        if not agent_output.actions and not agent_output.error:
                            logger.info(colored("Agent returned no actions — stopping episode.", "yellow"))
                            termination_reason = TerminationReason.NO_ACTION
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
                            termination_reason = TerminationReason.ENV_ERROR
                            break

                        logger.info(colored(f"Turn {turns} Env output: {env_output}", "blue"))
                        env_step = TrajectoryStep(output=env_output, start_time=env_ts, end_time=time.time())
                        self.storage.save_step(env_step, trajectory.id, len(trajectory.steps))
                        summary_proc.on_step(len(trajectory.steps), env_step)
                        trajectory.steps.append(env_step)
                        span.set_attribute("done", env_output.done)
                        span.set_attribute("reward", env_output.reward)
                        turns += 1

                if env_output.done:
                    termination_reason = termination_reason or TerminationReason.ENV_DONE
                elif turns >= self.config.max_steps:
                    termination_reason = TerminationReason.MAX_STEPS
                else:
                    termination_reason = termination_reason or TerminationReason.STOPPED
                
                if termination_reason != TerminationReason.ENV_DONE:
                    reward, raw_info = evaluate_fn(env_output.obs)
                    if isinstance(raw_info, EvalInfo):
                        info = raw_info.model_dump(exclude_none=True)
                        extra_info = info.pop("extra", None)
                        if isinstance(extra_info, dict):
                            info.update(extra_info)
                    elif isinstance(raw_info, dict):
                        info = dict(raw_info)
                    else:
                        raise TypeError(
                            f"Task.evaluate() must return dict or EvalInfo info payload, got {type(raw_info).__name__}"
                        )
                    env_output = EnvironmentOutput(obs=env_output.obs, reward=reward, info=info)

                trajectory.end_time = time.time()
                trajectory.reward_info = {"reward": env_output.reward, "done": env_output.done, **env_output.info}
                trajectory.summary_stats = _compute_summary_stats(trajectory)
                trajectory.termination_reason = termination_reason
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
        logger.info(colored(f"Turn {turns} Agent output: {agent_output}", "magenta"))


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
