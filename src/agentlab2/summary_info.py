"""Accumulate and save experiment summary info (Issue #148).

Saves cumulative statistics at every step to summary_info.json so the viewer
and users get a quick overview without loading full trajectories. Saving is
non-blocking so the main loop is not slowed by I/O.
"""

import json
import logging
import threading
from pathlib import Path
from queue import Empty, Queue

from agentlab2.core import AgentOutput, EnvironmentOutput, Trajectory, TrajectoryStep
from agentlab2.llm import Usage

logger = logging.getLogger(__name__)

SUMMARY_FILENAME = "summary_info.json"


def _safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def _safe_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def extract_step_stats(step: TrajectoryStep) -> dict[str, float | int | None]:
    """Extract numeric and error stats from a single step for aggregation.

    Returns a flat dict of stat keys to values. Agent steps contribute
    usage stats (prompt_tokens, completion_tokens, total_tokens, cost);
    env steps contribute reward, done. Errors contribute err_msg and stack_trace
    as non-numeric fields (handled separately in summary).
    """
    stats: dict[str, float | int | None] = {}
    out = step.output

    if isinstance(out, EnvironmentOutput):
        stats["reward"] = out.reward
        stats["done"] = 1.0 if out.done else 0.0
        if out.error:
            stats["error_type"] = out.error.error_type  # store as sentinel; aggregate can count
    elif isinstance(out, AgentOutput):
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cost = 0.0
        for llm_call in out.llm_calls:
            u: Usage = getattr(llm_call, "usage", None) or Usage()
            prompt_tokens += _safe_int(getattr(u, "prompt_tokens", 0))
            completion_tokens += _safe_int(getattr(u, "completion_tokens", 0))
            total_tokens += _safe_int(getattr(u, "total_tokens", 0))
            cost += _safe_float(getattr(u, "cost", 0.0))
        stats["prompt_tokens"] = prompt_tokens
        stats["completion_tokens"] = completion_tokens
        stats["total_tokens"] = total_tokens
        stats["cost"] = cost
        if out.error:
            stats["error_type"] = out.error.error_type

    return stats


def aggregate_stats(step_stats_list: list[dict[str, float | int | None]]) -> dict[str, float | int | None]:
    """Aggregate step stats: compute sum and max per key (stats-agnostic).

    For each numeric key we produce cum_<key> (sum) and max_<key> (max).
    None/NaN are skipped in sum/max. Returns a dict suitable for summary_info.
    """
    if not step_stats_list:
        return {"cum_steps": 0}

    # Collect values per key (skip non-numeric and error_type)
    key_to_values: dict[str, list[float]] = {}
    for d in step_stats_list:
        for key, val in d.items():
            if key == "error_type":
                continue
            if val is None:
                continue
            try:
                f = float(val) if not isinstance(val, (int, float)) else float(val)
            except (TypeError, ValueError):
                continue
            if key not in key_to_values:
                key_to_values[key] = []
            key_to_values[key].append(f)

    aggregated: dict[str, float | int | None] = {
        "cum_steps": len(step_stats_list),
    }

    for key, values in key_to_values.items():
        if not values:
            continue
        # Use standard library; avoid numpy dependency
        valid = [v for v in values if v == v]  # drop nan
        if valid:
            aggregated[f"cum_{key}"] = sum(valid)
            aggregated[f"max_{key}"] = max(valid)

    return aggregated


def _last_error_from_trajectory(trajectory: Trajectory) -> tuple[str | None, str | None]:
    """Get err_msg and stack_trace from the last step that has an error."""
    err_msg: str | None = None
    stack_trace: str | None = None
    for step in reversed(trajectory.steps):
        out = step.output
        err = getattr(out, "error", None)
        if err is not None:
            err_msg = getattr(err, "exception_str", None) or str(err)
            stack_trace = getattr(err, "stack_trace", None)
            break
    return err_msg, stack_trace


def build_summary_info(
    trajectories: list[Trajectory],
    err_msg: str | None = None,
    stack_trace: str | None = None,
) -> dict:
    """Build the summary_info dict from a list of trajectories (cumulative so far)."""
    all_step_stats: list[dict[str, float | int | None]] = []
    n_steps = 0
    cum_reward = 0.0
    last_done = False
    last_truncated = False

    for traj in trajectories:
        for step in traj.steps:
            st = extract_step_stats(step)
            all_step_stats.append(st)
            if "reward" in st:
                cum_reward += _safe_float(st["reward"]) if st["reward"] is not None else 0.0
        n_steps += len(traj.steps)
        if traj.steps:
            last_out = traj.steps[-1].output
            if isinstance(last_out, EnvironmentOutput):
                last_done = last_out.done
                last_truncated = last_out.info.get("truncated", False) if last_out.info else False
        if err_msg is None and stack_trace is None:
            err_msg, stack_trace = _last_error_from_trajectory(traj)

    summary: dict = {
        "n_steps": n_steps,
        "n_trajectories": len(trajectories),
        "cum_reward": cum_reward,
        "err_msg": err_msg,
        "stack_trace": stack_trace,
        "terminated": last_done,
        "truncated": last_truncated,
    }

    agg = aggregate_stats(all_step_stats)
    for key, val in agg.items():
        if val is not None and (val == val):  # not nan
            summary[f"stats.{key}"] = val

    return summary


def save_summary_info_sync(output_dir: Path, summary: dict) -> None:
    """Write summary_info.json to output_dir (blocking)."""
    path = Path(output_dir) / SUMMARY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=4)


class SummaryAccumulator:
    """Maintains cumulative stats per output_dir and writes summary_info.json non-blocking."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trajectories_by_dir: dict[Path, list[Trajectory]] = {}
        self._write_queue: Queue[tuple[Path, list[Trajectory]]] = Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def update(self, output_dir: Path, trajectory: Trajectory) -> None:
        """Record the current state of this trajectory and queue a non-blocking write.

        Call this after each save_step or save_trajectory. We merge this trajectory
        into the in-memory list for this output_dir (by id), then enqueue a write.
        """
        output_dir = Path(output_dir).resolve()
        with self._lock:
            if output_dir not in self._trajectories_by_dir:
                self._trajectories_by_dir[output_dir] = []
            trajs = self._trajectories_by_dir[output_dir]
            # Replace or append by trajectory id
            existing_ids = {t.id for t in trajs}
            if trajectory.id in existing_ids:
                trajs = [t if t.id != trajectory.id else trajectory for t in trajs]
            else:
                trajs = list(trajs) + [trajectory]
            self._trajectories_by_dir[output_dir] = trajs
            self._write_queue.put((output_dir, list(trajs)))

    def _worker_loop(self) -> None:
        while True:
            try:
                output_dir, trajectories = self._write_queue.get(timeout=1.0)
            except Empty:
                continue
            try:
                summary = build_summary_info(trajectories)
                save_summary_info_sync(output_dir, summary)
                logger.debug(f"Wrote {SUMMARY_FILENAME} to {output_dir}")
            except Exception as e:
                logger.warning(f"Failed to write summary_info: {e}")

    def flush(self, output_dir: Path | None = None) -> None:
        """Drain the write queue for the given dir (or all). Blocking."""
        to_flush: list[tuple[Path, list[Trajectory]]] = []
        while True:
            try:
                d, t = self._write_queue.get_nowait()
                if output_dir is None or Path(d).resolve() == Path(output_dir).resolve():
                    to_flush.append((d, t))
            except Empty:
                break
        for d, t in to_flush:
            try:
                save_summary_info_sync(d, build_summary_info(t))
            except Exception as e:
                logger.warning(f"Failed to write summary_info: {e}")


# Module-level accumulator for sequential runs; workers can use their own
_accumulator: SummaryAccumulator | None = None
_accumulator_lock = threading.Lock()


def get_global_accumulator() -> SummaryAccumulator:
    with _accumulator_lock:
        global _accumulator
        if _accumulator is None:
            _accumulator = SummaryAccumulator()
        return _accumulator
