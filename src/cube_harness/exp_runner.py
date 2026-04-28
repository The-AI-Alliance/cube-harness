"""Run experiments with Ray or sequentially."""

import logging
import time
import traceback
from uuid import uuid4

import ray

from cube_harness.core import Trajectory
from cube_harness.episode import Episode
from cube_harness.episode_logs import LOG_FORMAT, get_log_path, redirect_output_to_log, trajectory_log_id
from cube_harness.episode_status import TERMINAL_STATUSES, EpisodeStatus, next_retry_count
from cube_harness.experiment import Experiment, ExpResult, sweep_stale_statuses
from cube_harness.metrics.tracer import get_trace_env_vars, get_tracer
from cube_harness.storage import FileStorage

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _extract_model(exp: Experiment) -> str | None:
    llm_config = getattr(exp.agent_config, "llm_config", None)
    return llm_config.model_name if llm_config else None


def _trajectory_id(episode: Episode) -> str:
    return trajectory_log_id(episode.config.task_config.task_id, episode.config.id)


def _pre_claim(storage: FileStorage, episode: Episode) -> None:
    """Write `RUNNING` for an episode before submitting it to Ray.

    If the previous attempt left a terminal status, archive its directory first
    so the per-attempt history (including the terminal `status.json`) is preserved.
    """
    traj_id = _trajectory_id(episode)
    prior = storage.read_episode_status(traj_id)
    if prior is not None and prior.status in TERMINAL_STATUSES:
        ep_dir = storage._episode_dir(traj_id)
        if ep_dir.exists():
            storage._archive_episode(ep_dir)
    storage.write_episode_status(
        traj_id,
        EpisodeStatus(
            status="RUNNING",
            task_id=episode.config.task_config.task_id,
            episode_id=episode.config.id,
            started_at=time.time(),
            last_heartbeat_at=None,
            current_step=0,
            retry_count=next_retry_count(prior),
        ),
    )


def run_with_ray(
    exp: Experiment,
    *,
    n_cpus: int = 4,
    ray_poll_timeout: float = 2.0,
    step_timeout_s: float = 1800.0,
    cancel_grace_s: float = 120.0,
    orphan_threshold_s: float = 3600.0,
    max_retry_rounds: int = 3,
    otlp_endpoint: str | None = None,
    model: str | None = None,
    agent_name: str | None = None,
) -> ExpResult:
    model = model or _extract_model(exp)
    tracer = get_tracer(
        exp.name,
        otlp_endpoint=otlp_endpoint,
        model=model,
        agent_name=agent_name,
    )

    try:
        with tracer.benchmark(exp.name):
            return _run_with_retries(
                exp,
                n_cpus=n_cpus,
                ray_poll_timeout=ray_poll_timeout,
                step_timeout_s=step_timeout_s,
                cancel_grace_s=cancel_grace_s,
                orphan_threshold_s=orphan_threshold_s,
                max_retry_rounds=max_retry_rounds,
            )
    finally:
        tracer.shutdown()


def _run_with_retries(
    exp: Experiment,
    *,
    n_cpus: int,
    ray_poll_timeout: float,
    step_timeout_s: float,
    cancel_grace_s: float,
    orphan_threshold_s: float,
    max_retry_rounds: int,
) -> ExpResult:
    """Run the experiment, then re-run any retriable failures up to `max_retry_rounds` times."""
    aggregated = ExpResult(tasks_num=0, config=exp.config, exp_id=f"{exp.name}_{uuid4().hex}")
    original_resume = exp.resume
    original_retry_failed = exp.retry_failed
    try:
        round_num = 0
        while True:
            round_result = _run_with_ray_impl(
                exp,
                n_cpus=n_cpus,
                ray_poll_timeout=ray_poll_timeout,
                step_timeout_s=step_timeout_s,
                cancel_grace_s=cancel_grace_s,
                orphan_threshold_s=orphan_threshold_s,
            )
            aggregated.tasks_num = max(aggregated.tasks_num, round_result.tasks_num)
            aggregated.trajectories.update(round_result.trajectories)
            for k, v in round_result.failures.items():
                aggregated.failures[k] = v
            for k in round_result.trajectories:
                aggregated.failures.pop(k, None)

            if round_num >= max_retry_rounds:
                break

            # Decide whether to run another retry round by checking if anything is still retriable.
            storage = FileStorage(exp.output_dir)
            statuses = storage.list_episode_statuses()
            has_retriable = any(
                s.status in ("FAILED", "CANCELLED", "STALE") and s.retry_count < exp.max_retries
                for s in statuses.values()
            )
            if not has_retriable:
                break

            round_num += 1
            logger.info(f"Auto-retry round {round_num}/{max_retry_rounds}: re-running retriable episodes")
            exp.resume = False
            exp.retry_failed = True
        return aggregated
    finally:
        exp.resume = original_resume
        exp.retry_failed = original_retry_failed


def _run_with_ray_impl(
    exp: Experiment,
    *,
    n_cpus: int,
    ray_poll_timeout: float,
    step_timeout_s: float,
    cancel_grace_s: float,
    orphan_threshold_s: float,
) -> ExpResult:
    exp.save_config()
    output_dir = exp.output_dir
    storage = FileStorage(output_dir)

    @ray.remote
    def run_episode(episode: Episode) -> Trajectory:
        traj_id = _trajectory_id(episode)
        log_file = get_log_path(output_dir, traj_id)
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
        episodes = exp.get_episodes_to_run(
            step_timeout_s=step_timeout_s,
            cancel_grace_s=cancel_grace_s,
            orphan_threshold_s=orphan_threshold_s,
        )

        # Pre-claim every episode before any Ray submission so a concurrent runner
        # opening the same output_dir sees them as RUNNING (queued).
        for episode in episodes:
            _pre_claim(storage, episode)

        ref_to_traj_id: dict[ray.ObjectRef, str] = {}
        for episode in episodes:
            ref = run_episode.remote(episode)
            ref_to_traj_id[ref] = _trajectory_id(episode)
        logger.info(f"Start {len(episodes)} episodes in parallel using Ray with {n_cpus} workers")
        results = _poll_ray(
            exp,
            ref_to_traj_id,
            storage,
            ray_poll_timeout=ray_poll_timeout,
            step_timeout_s=step_timeout_s,
            cancel_grace_s=cancel_grace_s,
        )
        exp.print_stats(results)
        return results
    finally:
        ray.shutdown()
        # End-of-run STALE sweep: any RUNNING entries whose worker just got killed
        # by ray.shutdown() get marked STALE so the next round can retry them.
        sweep_stale_statuses(
            storage,
            step_timeout_s=step_timeout_s,
            cancel_grace_s=cancel_grace_s,
            orphan_threshold_s=orphan_threshold_s,
        )
        exp.benchmark.close()


def _poll_ray(
    exp: Experiment,
    ref_to_traj_id: dict[ray.ObjectRef, str],
    storage: FileStorage,
    *,
    ray_poll_timeout: float,
    step_timeout_s: float,
    cancel_grace_s: float,
) -> ExpResult:
    results = ExpResult(tasks_num=len(ref_to_traj_id), config=exp.config, exp_id=f"{exp.name}_{uuid4().hex}")
    completed = 0
    episodes_in_progress = list(ref_to_traj_id.keys())
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
            traj_id = ref_to_traj_id[task_ref]
            try:
                traj: Trajectory = ray.get(task_ref)
                logger.info(
                    "Completed trajectory %s with %d steps (%d agent steps, %d environment steps)",
                    traj_id,
                    len(traj.steps),
                    traj.n_agent_steps,
                    traj.n_env_steps,
                )
                results.trajectories[traj_id] = traj
            except Exception as e:
                logger.exception(f"Run failed with exception: {e}")
                results.failures[traj_id] = str(e)

        # Driver-side step timeout via filesystem read (replaces ray-dashboard list_tasks).
        _kill_stale_workers(
            episodes_in_progress,
            ref_to_traj_id,
            storage,
            results,
            step_timeout_s=step_timeout_s,
            cancel_grace_s=cancel_grace_s,
        )
    return results


def _kill_stale_workers(
    episodes_in_progress: list[ray.ObjectRef],
    ref_to_traj_id: dict[ray.ObjectRef, str],
    storage: FileStorage,
    results: ExpResult,
    *,
    step_timeout_s: float,
    cancel_grace_s: float,
) -> None:
    """Read each active episode's status.json; force-kill workers with stale heartbeats.

    A worker that has not advanced its heartbeat in `step_timeout_s + cancel_grace_s`
    is presumed stuck. The driver force-cancels it, writes a `CANCELLED` status with
    `error_type="StepTimeout"`, and removes the ref from the in-progress list.
    """
    now = time.time()
    to_remove: list[ray.ObjectRef] = []
    for ref in list(episodes_in_progress):
        traj_id = ref_to_traj_id[ref]
        status = storage.read_episode_status(traj_id)
        if status is None or status.last_heartbeat_at is None:
            # Pre-claim only — episode is queued in Ray, not yet picked up.
            continue
        age = now - status.last_heartbeat_at
        if age <= step_timeout_s + cancel_grace_s:
            continue
        logger.error(
            f"Episode {traj_id} step {status.current_step} stalled for {age:.0f}s "
            f"(>{step_timeout_s + cancel_grace_s:.0f}s) — force killing"
        )
        try:
            ray.cancel(ref, force=True)
        except Exception:
            logger.exception(f"ray.cancel failed for {traj_id}")
        status.status = "CANCELLED"
        status.ended_at = now
        status.error_type = "StepTimeout"
        status.error_message = f"Step {status.current_step} exceeded {step_timeout_s:.0f}s"
        storage.write_episode_status(traj_id, status)
        results.failures[traj_id] = status.error_message
        to_remove.append(ref)
    for ref in to_remove:
        episodes_in_progress.remove(ref)


def run_sequentially(
    exp: Experiment,
    debug_limit: int | None = None,
    *,
    max_retry_rounds: int = 3,
    step_timeout_s: float = 1800.0,
    cancel_grace_s: float = 120.0,
    orphan_threshold_s: float = 3600.0,
    otlp_endpoint: str | None = None,
    model: str | None = None,
    agent_name: str | None = None,
) -> ExpResult:
    model = model or _extract_model(exp)
    tracer = get_tracer(
        exp.name,
        otlp_endpoint=otlp_endpoint,
        model=model,
        agent_name=agent_name,
    )

    try:
        with tracer.benchmark(exp.name):
            return _run_sequentially_with_retries(
                exp,
                debug_limit=debug_limit,
                max_retry_rounds=max_retry_rounds,
                step_timeout_s=step_timeout_s,
                cancel_grace_s=cancel_grace_s,
                orphan_threshold_s=orphan_threshold_s,
            )
    finally:
        tracer.shutdown()


def _run_sequentially_with_retries(
    exp: Experiment,
    *,
    debug_limit: int | None,
    max_retry_rounds: int,
    step_timeout_s: float,
    cancel_grace_s: float,
    orphan_threshold_s: float,
) -> ExpResult:
    aggregated = ExpResult(tasks_num=0, config=exp.config, exp_id=f"{exp.name}_{uuid4().hex}")
    original_resume = exp.resume
    original_retry_failed = exp.retry_failed
    try:
        round_num = 0
        while True:
            round_result = _run_sequentially_impl(
                exp,
                debug_limit=debug_limit,
                step_timeout_s=step_timeout_s,
                cancel_grace_s=cancel_grace_s,
                orphan_threshold_s=orphan_threshold_s,
            )
            aggregated.tasks_num = max(aggregated.tasks_num, round_result.tasks_num)
            aggregated.trajectories.update(round_result.trajectories)
            for k, v in round_result.failures.items():
                aggregated.failures[k] = v
            for k in round_result.trajectories:
                aggregated.failures.pop(k, None)

            if round_num >= max_retry_rounds:
                break

            storage = FileStorage(exp.output_dir)
            statuses = storage.list_episode_statuses()
            has_retriable = any(
                s.status in ("FAILED", "CANCELLED", "STALE") and s.retry_count < exp.max_retries
                for s in statuses.values()
            )
            if not has_retriable:
                break

            round_num += 1
            logger.info(f"Auto-retry round {round_num}/{max_retry_rounds}: re-running retriable episodes")
            exp.resume = False
            exp.retry_failed = True
        return aggregated
    finally:
        exp.resume = original_resume
        exp.retry_failed = original_retry_failed


def _run_sequentially_impl(
    exp: Experiment,
    *,
    debug_limit: int | None,
    step_timeout_s: float,
    cancel_grace_s: float,
    orphan_threshold_s: float,
) -> ExpResult:
    exp.save_config()
    exp.benchmark.setup()
    try:
        episodes = exp.get_episodes_to_run(
            step_timeout_s=step_timeout_s,
            cancel_grace_s=cancel_grace_s,
            orphan_threshold_s=orphan_threshold_s,
        )
        if debug_limit is not None:
            logger.info(f"Running only first {debug_limit} episodes for debugging")
            episodes = episodes[:debug_limit]
        results = ExpResult(
            tasks_num=len(episodes),
            config=exp.config,
            exp_id=f"{exp.name}_{uuid4().hex}",
        )
        for episode in episodes:
            traj_id = _trajectory_id(episode)
            log_file = get_log_path(exp.output_dir, traj_id)
            with redirect_output_to_log(log_file, append=False, tee=True, log_format=LOG_FORMAT):
                try:
                    trajectory = episode.run()
                    results.trajectories[traj_id] = trajectory
                except Exception as e:
                    logger.exception(f"Episode {traj_id} failed")
                    results.failures[traj_id] = str(e)
        exp.print_stats(results)
        return results
    finally:
        exp.benchmark.close()
