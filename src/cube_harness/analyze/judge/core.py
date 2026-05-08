"""Public judge API: judge_episode, judge_experiment, and supporting helpers."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cube_harness.analyze.judge.context import _load_experiment_view, collect_source_paths
from cube_harness.analyze.judge.prompt import JUDGE_SYSTEM_PROMPT, build_user_prompt
from cube_harness.analyze.judge.sdk import TraceMode, _extract_json_block, _run_claude_code
from cube_harness.analyze.judge.selection import EpisodeRef, _load_episode_record, discover_episodes, select_episodes
from cube_harness.analyze.judge.transcript import extract_transcript
from cube_harness.core import Trajectory
from cube_harness.eval_log import (
    JUDGE_SCHEMA_VERSION,
    JudgeMetadata,
    JudgeOutput,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_SAMPLE_FRACTION = 0.10
EXPERIMENT_JUDGE_SUMMARY_FILENAME = "experiment_judge_summary.json"
EXPERIMENT_JUDGE_REPORT_FILENAME = "experiment_judge_report.csv"


def _load_trajectory_meta(path: Path) -> Trajectory | None:
    """Load episode.metadata.json as a Trajectory. The `steps` field will be empty
    (steps live in `steps/*.msgpack.zst`); reward_info/metadata/summary_stats are populated."""
    if not path.exists():
        return None
    try:
        return Trajectory.model_validate_json(path.read_text())
    except (ValidationError, json.JSONDecodeError) as e:
        logger.warning("Could not parse %s as Trajectory: %s", path, e)
        return None


def _validate_invariants(obj: JudgeOutput) -> None:
    """Enforce the V1 invariants from the spec (post-parse, pre-write)."""
    if obj.primary_blame.value != "none" and not obj.evidence:
        raise ValueError("evidence must be non-empty when primary_blame != 'none'")
    if obj.primary_blame in obj.other_blames:
        raise ValueError("other_blames must not repeat primary_blame")
    if obj.outcome.value in ("success", "success_lucky") and obj.primary_blame.value != "none":
        # Soft-correct: tighten to spec rather than fail.
        logger.warning(
            "Judge returned outcome=%s with primary_blame=%s; spec requires 'none'. Coercing.",
            obj.outcome.value,
            obj.primary_blame.value,
        )
        obj.primary_blame = obj.primary_blame.__class__("none")


async def _judge_episode_impl(
    episode_dir: Path,
    experiment_dir: Path,
    model: str,
    verbose: bool,
    trace_mode: TraceMode = "actions",
) -> tuple[JudgeOutput, JudgeMetadata, list[dict[str, Any]]]:
    """Async core shared by judge_episode (single) and judge_experiment (parallel)."""
    transcript_dir = episode_dir / "_judge_transcript"
    extract_transcript(episode_dir, transcript_dir)

    metadata_path = episode_dir / "episode.metadata.json"
    config_path = episode_dir / "episode_config.json"
    experiment_config_path = experiment_dir / "experiment_config.json"

    trajectory = _load_trajectory_meta(metadata_path)
    view = _load_experiment_view(experiment_config_path)

    if trajectory is not None:
        task_id = trajectory.metadata.get("task_id") or trajectory.id
        reward = trajectory.reward_info.get("reward")
        total_steps = (trajectory.summary_stats or {}).get("n_agent_steps")
        task_description = trajectory.metadata.get("task_description", "")
    else:
        task_id, reward, total_steps, task_description = "unknown", None, None, ""

    source_paths = collect_source_paths(view)
    user_prompt = build_user_prompt(
        trajectory_id=episode_dir.name,
        task_id=task_id,
        reward=reward,
        total_steps=total_steps,
        agent_name=view.agent_dotted,
        benchmark_name=view.benchmark_dotted,
        transcript_dir=transcript_dir,
        episode_metadata_path=metadata_path,
        episode_config_path=config_path,
        task_description=task_description,
        source_paths=source_paths,
    )

    logger.info("Judging %s (reward=%s, steps=%s) with %s", episode_dir.name, reward, total_steps, model)

    result = await _run_claude_code(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        cwd=episode_dir,
        additional_dirs=list(source_paths.values()) + [transcript_dir],
        model=model,
        verbose=verbose,
        trace_mode=trace_mode,
    )

    obj = _extract_json_block(result.output_text)
    judge_output = JudgeOutput.model_validate(obj)
    _validate_invariants(judge_output)

    judge_metadata = JudgeMetadata(
        model=model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        duration_s=result.duration_s,
        timestamp=time.time(),
        judge_schema_version=JUDGE_SCHEMA_VERSION,
    )
    return judge_output, judge_metadata, result.actions


def judge_episode(
    episode_dir: Path,
    *,
    experiment_dir: Path | None = None,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    trace_mode: TraceMode = "actions",
) -> tuple[JudgeOutput, JudgeMetadata]:
    """Run a post-hoc judge on a single episode trajectory directory.

    `experiment_dir` is needed only to locate `experiment_config.json`; if omitted,
    we look one level up from `episode_dir`.
    """
    episode_dir = Path(episode_dir).resolve()
    if experiment_dir is None:
        experiment_dir = episode_dir.parent.parent
    judge_output, judge_metadata, _ = asyncio.run(
        _judge_episode_impl(episode_dir, Path(experiment_dir).resolve(), model, verbose, trace_mode)
    )
    return judge_output, judge_metadata


def _persist_judgment(
    ref: EpisodeRef,
    judge_output: JudgeOutput,
    judge_metadata: JudgeMetadata,
    actions: list[dict[str, Any]] | None = None,
    trace_mode: TraceMode = "actions",
) -> None:
    """Write `judge_output` and `judge_metadata` into the episode_record.json.

    If the record file does not exist yet (older runs without atlas-eval-log enabled),
    write a sidecar `judge_output.json` so the result is not lost.

    When `actions` is non-empty, writes a `judge_trace.json` sidecar alongside the
    episode record with the judge's tool-call sequence.
    """
    if ref.record is not None:
        updated = ref.record.model_copy(update={"judge_output": judge_output, "judge_metadata": judge_metadata})
        ref.record_path.write_text(updated.model_dump_json(indent=2))
    else:
        sidecar = ref.episode_dir / "judge_output.json"
        sidecar.write_text(
            json.dumps(
                {
                    "judge_output": judge_output.model_dump(mode="json"),
                    "judge_metadata": judge_metadata.model_dump(mode="json"),
                },
                indent=2,
            )
        )
    if actions:
        (ref.episode_dir / "judge_trace.json").write_text(
            json.dumps({"trace_mode": trace_mode, "actions": actions}, indent=2)
        )


def judge_experiment(
    experiment_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    ids: list[str] | None = None,
    sample: float | None = None,
    n: int | None = None,
    failures_only: bool = False,
    overwrite: bool = False,
    seed: int | None = None,
    verbose: bool = False,
    n_parallel: int = 1,
    trace_mode: TraceMode = "actions",
) -> dict[str, tuple[JudgeOutput, JudgeMetadata]]:
    """Batch judge selected episodes in an experiment output directory.

    Selection (default): all episodes that don't already have a judge_output.
    With `--sample 0.1`: 10% of those, randomly. With `--ids`: exactly those.
    With `n_parallel > 1`: run that many judge sub-processes concurrently.

    Writes per-episode results into `episode_record.json` (or a sidecar if missing)
    and aggregate stats into `experiment_judge_summary.json`.
    """
    experiment_dir = Path(experiment_dir).resolve()
    refs = discover_episodes(experiment_dir)
    selected = select_episodes(
        refs,
        ids=ids,
        sample=sample,
        n=n,
        failures_only=failures_only,
        overwrite=overwrite,
        seed=seed,
    )

    if not selected:
        logger.info("No episodes selected to judge in %s", experiment_dir)
        return {}

    logger.info(
        "Judging %d / %d episodes in %s (n_parallel=%d)",
        len(selected),
        len(refs),
        experiment_dir.name,
        n_parallel,
    )

    results: dict[str, tuple[JudgeOutput, JudgeMetadata]] = {}
    if n_parallel > 1:
        results = asyncio.run(
            _judge_experiment_parallel(selected, experiment_dir, model, verbose, n_parallel, trace_mode)
        )
    else:
        for ref in selected:
            try:
                judge_output, judge_metadata, actions = asyncio.run(
                    _judge_episode_impl(ref.episode_dir, experiment_dir, model, verbose, trace_mode)
                )
            except Exception as e:
                logger.exception("Judge failed on %s: %s", ref.trajectory_id, e)
                continue
            ref.record = _load_episode_record(ref.record_path)
            _persist_judgment(ref, judge_output, judge_metadata, actions, trace_mode)
            results[ref.trajectory_id] = (judge_output, judge_metadata)

    _write_summary(experiment_dir, selected, results, model=model)
    return results


async def _judge_experiment_parallel(
    selected: list[EpisodeRef],
    experiment_dir: Path,
    model: str,
    verbose: bool,
    n_parallel: int,
    trace_mode: TraceMode = "actions",
) -> dict[str, tuple[JudgeOutput, JudgeMetadata]]:
    semaphore = asyncio.Semaphore(n_parallel)
    results: dict[str, tuple[JudgeOutput, JudgeMetadata]] = {}

    async def _one(ref: EpisodeRef) -> None:
        async with semaphore:
            try:
                judge_output, judge_metadata, actions = await _judge_episode_impl(
                    ref.episode_dir, experiment_dir, model, verbose, trace_mode
                )
            except Exception as e:
                logger.exception("Judge failed on %s: %s", ref.trajectory_id, e)
                return
            ref.record = _load_episode_record(ref.record_path)
            _persist_judgment(ref, judge_output, judge_metadata, actions, trace_mode)
            results[ref.trajectory_id] = (judge_output, judge_metadata)

    await asyncio.gather(*[_one(ref) for ref in selected])
    return results


def _write_summary(
    experiment_dir: Path,
    selected: list[EpisodeRef],
    results: dict[str, tuple[JudgeOutput, JudgeMetadata]],
    *,
    model: str,
) -> None:
    if not results:
        return
    outcomes = Counter(o.outcome.value for o, _ in results.values())
    blames = Counter(o.primary_blame.value for o, _ in results.values())
    total_cost = sum(m.cost_usd for _, m in results.values())
    total_prompt = sum(m.prompt_tokens for _, m in results.values())
    total_completion = sum(m.completion_tokens for _, m in results.values())
    n = len(results)

    judged_episodes = [
        {
            "trajectory_id": ref.trajectory_id,
            "episode_record": str(ref.record_path.relative_to(experiment_dir)),
        }
        for ref in selected
        if ref.trajectory_id in results
    ]

    summary = {
        "n_judged": n,
        "model": model,
        "judge_schema_version": JUDGE_SCHEMA_VERSION,
        "timestamp": time.time(),
        "total_judge_cost_usd": round(total_cost, 4),
        "avg_judge_cost_usd": round(total_cost / n, 4) if n else 0.0,
        "total_judge_prompt_tokens": total_prompt,
        "total_judge_completion_tokens": total_completion,
        "outcomes": dict(outcomes),
        "primary_blame": dict(blames),
        "report_csv": EXPERIMENT_JUDGE_REPORT_FILENAME,
        "judged_episodes": judged_episodes,
    }
    (experiment_dir / EXPERIMENT_JUDGE_SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))
    _write_csv_report(experiment_dir, selected, results)


def _write_csv_report(
    experiment_dir: Path,
    selected: list[EpisodeRef],
    results: dict[str, tuple[JudgeOutput, JudgeMetadata]],
) -> None:
    """Write one row per judged episode for spreadsheet-friendly inspection.

    Excludes `analysis` and `evidence` — they're too verbose for LLM consumption.
    Read them from the per-episode `episode_record.json` when needed.
    """
    fields = [
        "trajectory_id",
        "episode_record",
        "reward",
        "n_steps",
        "outcome",
        "primary_blame",
        "primary_blame_confidence",
        "other_blames",
        "hypothesis_confidence",
        "summary",
        "hypothesis",
        "cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "duration_s",
    ]
    path = experiment_dir / EXPERIMENT_JUDGE_REPORT_FILENAME
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ref in selected:
            if ref.trajectory_id not in results:
                continue
            o, m = results[ref.trajectory_id]
            reward = ref.record.score if ref.record is not None else None
            n_steps = ref.record.n_agent_steps if ref.record is not None else None
            w.writerow(
                {
                    "trajectory_id": ref.trajectory_id,
                    "episode_record": str(ref.record_path.relative_to(experiment_dir)),
                    "reward": reward,
                    "n_steps": n_steps,
                    "outcome": o.outcome.value,
                    "primary_blame": o.primary_blame.value,
                    "primary_blame_confidence": o.primary_blame_confidence,
                    "other_blames": ";".join(b.value for b in o.other_blames),
                    "hypothesis_confidence": o.hypothesis_confidence,
                    "summary": o.summary,
                    "hypothesis": o.hypothesis,
                    "cost_usd": round(m.cost_usd, 4),
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "duration_s": round(m.duration_s, 2),
                }
            )
