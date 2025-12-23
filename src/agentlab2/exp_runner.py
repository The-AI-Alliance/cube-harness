"""Run experiments with Ray or sequentially."""

import logging
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TextIO
from uuid import uuid4

import ray


class TeeWriter:
    def __init__(self, *writers: TextIO) -> None:
        self.writers = writers

    def write(self, data: str) -> int:
        for writer in self.writers:
            writer.write(data)
        return len(data)

    def flush(self) -> None:
        for writer in self.writers:
            writer.flush()


@contextmanager
def tee_to_file(log_file: Path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_file.open("w") as f:
        tee_out = TeeWriter(original_stdout, f)
        tee_err = TeeWriter(original_stderr, f)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):  # type: ignore[type-var]
            yield

from agentlab2.core import Trajectory
from agentlab2.episode import Episode
from agentlab2.experiment import Experiment, ExpResult
from agentlab2.metrics.tracer import get_trace_env_vars, get_tracer

LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d %(funcName)s() - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def run_with_ray(
    exp: Experiment,
    n_cpus: int = 4,
    ray_poll_timeout: float = 2.0,
    trace_output: str | Path | None = None,
    otlp_endpoint: str | None = None,
) -> ExpResult:
    tracer = get_tracer(exp.name, output_dir=trace_output, otlp_endpoint=otlp_endpoint)

    try:
        with tracer.benchmark(exp.name):
            return _run_with_ray_impl(exp, n_cpus, ray_poll_timeout)
    finally:
        tracer.shutdown()


def _run_with_ray_impl(exp: Experiment, n_cpus: int, ray_poll_timeout: float) -> ExpResult:
    exp.save_config()
    log_dir = Path(exp.output_dir) / "logs"

    @ray.remote
    def run_episode(episode: Episode) -> Trajectory:
        trajectory_id = f"{episode.task_id}_ep{episode.id}"
        log_file = log_dir / f"{trajectory_id}.log"
        sys.stdout = sys.stderr = log_file.open("a", buffering=1)
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout, force=True)
        return episode.run()

    if not ray.is_initialized():
        log_dir.mkdir(parents=True, exist_ok=True)
        ray.init(
            num_cpus=n_cpus,
            dashboard_host="0.0.0.0",
            include_dashboard=True,
            log_to_driver=True,
            runtime_env={"working_dir": None, "env_vars": get_trace_env_vars()},
        )  # TODO: Ray breaks signal handling, we cannot react to Ctrl+C here, still cannot find a workaround

    exp.benchmark.setup()
    try:
        episodes = exp.create_episodes()
        ref_to_id = {run_episode.remote(episode): episode.task_id for episode in episodes}
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
) -> ExpResult:
    tracer = get_tracer(exp.name, output_dir=trace_output, otlp_endpoint=otlp_endpoint)

    try:
        with tracer.benchmark(exp.name):
            return _run_sequentially_impl(exp, debug_limit)
    finally:
        tracer.shutdown()


def _run_sequentially_impl(exp: Experiment, debug_limit: int | None) -> ExpResult:
    exp.save_config()
    log_dir = Path(exp.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    exp.benchmark.setup()
    try:
        episodes = exp.create_episodes()
        if debug_limit is not None:
            logger.info(f"Running only first {debug_limit} episodes for debugging")
            episodes = episodes[:debug_limit]

        trajectories = []
        for episode in episodes:
            trajectory_id = f"{episode.task_id}_ep{episode.id}"
            log_file = log_dir / f"{trajectory_id}.log"
            with tee_to_file(log_file):
                trajectory = episode.run()
                trajectories.append(trajectory)
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
