"""Run experiments with Ray or sequentially."""

import logging
import time
from pathlib import Path
from uuid import uuid4

import ray
from ray.util.state.api import list_tasks

from cube_harness.core import Trajectory
from cube_harness.episode import Episode
from cube_harness.episode_logs import LOG_FORMAT, get_log_path, redirect_output_to_log, trajectory_log_id
from cube_harness.experiment import Experiment, ExpResult
from cube_harness.metrics.tracer import get_trace_env_vars, get_tracer

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

_CANCEL_GRACE_PERIOD_S = 60


def _extract_model(exp: Experiment) -> str | None:
    llm_config = getattr(exp.agent_config, "llm_config", None)
    return llm_config.model_name if llm_config else None


def run_with_ray(
    exp: Experiment,
    n_cpus: int = 4,
    ray_poll_timeout: float = 2.0,
    episode_timeout: float | None = 3600.0,
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
            return _run_with_ray_impl(exp, n_cpus, ray_poll_timeout, episode_timeout)
    finally:
        tracer.shutdown()


def _run_with_ray_impl(
    exp: Experiment, n_cpus: int, ray_poll_timeout: float, episode_timeout: float | None
) -> ExpResult:
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
        results = _poll_ray(exp, ref_to_id, ray_poll_timeout, episode_timeout)
        exp.print_stats(results)
        return results
    finally:
        ray.shutdown()
        exp.benchmark.close()


def _get_running_elapsed_s(refs: list[ray.ObjectRef]) -> dict[ray.ObjectRef, float]:
    """Return elapsed running time (seconds) for each ref that is actively executing.

    Queries the Ray dashboard in bulk. Refs that are still queued (no start_time_ms)
    are excluded. Returns an empty dict if the dashboard is unreachable.
    """
    try:
        running = {
            t.task_id: t.start_time_ms
            for t in list_tasks(filters=[("state", "=", "RUNNING")])
            if t.start_time_ms is not None
        }
    except Exception:
        logger.warning("Could not reach Ray dashboard to check episode timeouts — skipping this cycle")
        return {}
    now_ms = time.time() * 1000
    return {ref: (now_ms - running[ref.task_id().hex()]) / 1000 for ref in refs if ref.task_id().hex() in running}


def _poll_ray(
    exp: Experiment,
    ref_to_id: dict[ray.ObjectRef, str],
    ray_poll_timeout: float,
    episode_timeout: float | None,
) -> ExpResult:
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
        if episode_timeout is not None:
            for ref, elapsed in _get_running_elapsed_s(episodes_in_progress).items():
                if elapsed > episode_timeout:
                    task_id = ref_to_id[ref]
                    if elapsed < episode_timeout + _CANCEL_GRACE_PERIOD_S:
                        logger.warning(f"Episode {task_id} timed out after {elapsed:.0f}s — cancelling gracefully")
                        ray.cancel(ref, force=False)
                    else:
                        logger.error(
                            f"Episode {task_id} timed out after {elapsed:.0f}s — force killing after grace period"
                        )
                        ray.cancel(ref, force=True)
                        results.failures[task_id] = f"Episode timed out after {elapsed:.0f}s"
                        episodes_in_progress.remove(ref)
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
