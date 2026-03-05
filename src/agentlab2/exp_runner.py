"""Run experiments with Ray or sequentially."""

import logging
from pathlib import Path
from uuid import uuid4

import ray

from agentlab2.core import Trajectory
from agentlab2.episode import Episode
from agentlab2.episode_logs import LOG_FORMAT, get_log_path, redirect_output_to_log, trajectory_log_id
from agentlab2.experiment import Experiment, ExpResult
from agentlab2.metrics.tracer import get_trace_env_vars, get_tracer

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _extract_model(exp: Experiment) -> str | None:
    llm_config = getattr(exp.agent_config, "llm_config", None)
    return llm_config.model_name if llm_config else None


def run_with_ray(
    exp: Experiment,
    n_cpus: int = 4,
    ray_poll_timeout: float = 2.0,
    trace_output: str | Path | None = None,
    otlp_endpoint: str | None = None,
    model: str | None = None,
    agent_name: str | None = None,
) -> ExpResult:
    model = model or _extract_model(exp)
    tracer = get_tracer(
        exp.name,
        output_dir=trace_output,
        otlp_endpoint=otlp_endpoint,
        model=model,
        agent_name=agent_name,
    )

    try:
        with tracer.benchmark(exp.name):
            return _run_with_ray_impl(exp, n_cpus, ray_poll_timeout)
    finally:
        tracer.shutdown()


def _run_with_ray_impl(exp: Experiment, n_cpus: int, ray_poll_timeout: float) -> ExpResult:
    exp.save_config()
    output_dir = exp.output_dir

    @ray.remote
    def run_episode(episode: Episode) -> Trajectory:
        trajectory_id = trajectory_log_id(episode.config.task_id, episode.config.id)
        log_file = get_log_path(output_dir, trajectory_id)
        with redirect_output_to_log(log_file, append=True, tee=False, log_format=LOG_FORMAT):
            return episode.run()

    if not ray.is_initialized():
        ray.init(
            num_cpus=n_cpus,
            dashboard_host="0.0.0.0",
            include_dashboard=True,
            log_to_driver=True,
            runtime_env={"working_dir": None, "env_vars": get_trace_env_vars()},
        )  # TODO: Ray breaks signal handling, we cannot react to Ctrl+C here, still cannot find a workaround

    exp.benchmark.setup()
    try:
        episodes = exp.get_episodes_to_run()
        ref_to_id = {run_episode.remote(episode): episode.config.task_id for episode in episodes}
        logger.info(f"Start {len(episodes)} episodes in parallel using Ray with {n_cpus} workers")
        results = _poll_ray(exp, ref_to_id, ray_poll_timeout)
        exp.print_stats(results)
        return results
    finally:
        ray.shutdown()
        exp.benchmark.close()


def _poll_ray(exp: Experiment, ref_to_id: dict[ray.ObjectRef, str], ray_poll_timeout: float) -> ExpResult:
    results = ExpResult(tasks_num=len(ref_to_id), config=exp.config, exp_id=f"{exp.name}_{uuid4().hex}")
    completed = 0
    episodes_in_progress = list(ref_to_id.keys())
    while len(episodes_in_progress) > 0:
        done, episodes_in_progress = ray.wait(
            episodes_in_progress,
            num_returns=len(episodes_in_progress),
            timeout=ray_poll_timeout,
        )
        completed += len(done)
        if len(done) > 0:
            logger.info(f"{completed} episodes completed, {len(episodes_in_progress)} in progress")
        for task_ref in done:
            task_id = ref_to_id[task_ref]
            try:
                traj: Trajectory = ray.get(task_ref)
                logger.info(f"Completed trajectory for task {task_id} with {len(traj.steps)} steps")
                results.trajectories[task_id] = traj
            except Exception as e:
                logger.exception(f"Run failed with exception: {e}")
                results.failures[task_id] = str(e)
    return results


def run_sequentially(
    exp: Experiment,
    debug_limit: int | None = None,
    trace_output: str | Path | None = None,
    otlp_endpoint: str | None = None,
    model: str | None = None,
    agent_name: str | None = None,
) -> ExpResult:
    model = model or _extract_model(exp)
    tracer = get_tracer(
        exp.name,
        output_dir=trace_output,
        otlp_endpoint=otlp_endpoint,
        model=model,
        agent_name=agent_name,
    )

    try:
        with tracer.benchmark(exp.name):
            return _run_sequentially_impl(exp, debug_limit)
    finally:
        tracer.shutdown()


def _run_sequentially_impl(exp: Experiment, debug_limit: int | None) -> ExpResult:
    exp.save_config()
    exp.benchmark.setup()
    try:
        episodes = exp.get_episodes_to_run()
        if debug_limit is not None:
            logger.info(f"Running only first {debug_limit} episodes for debugging")
            episodes = episodes[:debug_limit]
        trajectories = []
        for episode in episodes:
            trajectory_id = trajectory_log_id(episode.config.task_id, episode.config.id)
            log_file = get_log_path(exp.output_dir, trajectory_id)
            with redirect_output_to_log(log_file, append=False, tee=True, log_format=LOG_FORMAT):
                trajectories.append(episode.run())
        results = ExpResult(
            tasks_num=len(episodes),
            trajectories={traj.metadata["task_id"]: traj for traj in trajectories},
            config=exp.config,
            exp_id=f"{exp.name}_{uuid4().hex}",
        )
        exp.print_stats(results)
        return results
    finally:
        exp.benchmark.close()
