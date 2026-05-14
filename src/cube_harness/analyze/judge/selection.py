"""Episode discovery and selection.

Two layers:

1. `discover_episodes` / `select_episodes` — scan `<exp>/episodes/` and filter
   the result by ids / sample / failures-only. Used by `judge_experiment`'s
   default path. Unchanged from PR #366.

2. `Selector` Protocol with three concrete selectors —
   `SameTaskDifferentAgent`, `SameAgentPreviousIteration`, `TopKBySimilarityStub`.
   These are *related-trajectory* selectors: given one main episode, return up to
   `k` other episode paths the judge should also read for context. Plugged in via
   the `selector=` kwarg on `judge_episode` / `judge_experiment`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from pydantic import ValidationError

from cube_harness.eval_log import EPISODE_RECORD_FILENAME, EpisodeRecord

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRef:
    trajectory_id: str
    episode_dir: Path
    record_path: Path
    record: EpisodeRecord | None  # None if record file missing


def load_episode_record(record_path: Path) -> EpisodeRecord | None:
    if not record_path.exists():
        return None
    try:
        return EpisodeRecord.model_validate_json(record_path.read_text())
    except (ValidationError, json.JSONDecodeError) as e:
        logger.warning("Could not parse %s: %s", record_path, e)
        return None


def discover_episodes(experiment_dir: Path) -> list[EpisodeRef]:
    """Return all episode directories under `<experiment_dir>/episodes/`."""
    episodes_dir = experiment_dir / "episodes"
    if not episodes_dir.exists():
        # Maybe the user passed a single episode directory.
        if (experiment_dir / "steps").exists():
            return [
                EpisodeRef(
                    trajectory_id=experiment_dir.name,
                    episode_dir=experiment_dir,
                    record_path=experiment_dir / EPISODE_RECORD_FILENAME,
                    record=load_episode_record(experiment_dir / EPISODE_RECORD_FILENAME),
                )
            ]
        raise FileNotFoundError(f"No 'episodes/' under {experiment_dir} and no 'steps/' inside it")

    refs: list[EpisodeRef] = []
    for ep_dir in sorted(episodes_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        record_path = ep_dir / EPISODE_RECORD_FILENAME
        refs.append(
            EpisodeRef(
                trajectory_id=ep_dir.name,
                episode_dir=ep_dir,
                record_path=record_path,
                record=load_episode_record(record_path),
            )
        )
    return refs


def select_episodes(
    refs: list[EpisodeRef],
    *,
    ids: list[str] | None = None,
    sample: float | None = None,
    n: int | None = None,
    failures_only: bool = False,
    overwrite: bool = False,
    seed: int | None = None,
) -> list[EpisodeRef]:
    """Filter and sample episode refs.

    `ids` is an explicit override: when set, the named episodes are returned
    verbatim regardless of `failures_only` / already-judged / sampling — the user
    typed exactly what they want.

    Otherwise: failures-only → already-judged (unless `overwrite`) → sample/n.
    """
    if ids:
        wanted = set(ids)
        return [r for r in refs if r.trajectory_id in wanted or r.trajectory_id.split("_ep")[0] in wanted]

    pool = refs
    if failures_only:
        pool = [r for r in pool if (r.record is not None and not r.record.is_correct)]

    if not overwrite:
        pool = [r for r in pool if not (r.record is not None and r.record.judge_output is not None)]

    rng = random.Random(seed)
    if n is not None:
        if n >= len(pool):
            return pool
        return rng.sample(pool, n)
    if sample is not None:
        if sample >= 1.0:
            return pool
        k = max(1, int(round(len(pool) * sample))) if pool else 0
        return rng.sample(pool, k)
    return pool


# ---------------------------------------------------------------------------
# Related-trajectory selectors
# ---------------------------------------------------------------------------


@runtime_checkable
class Selector(Protocol):
    """Pluggable rule for picking *related* episodes the judge should also read.

    `select` is sync: selectors are expected to be cheap (look at filenames or a
    few JSON files), not run extra LLMs. Heavy similarity scorers should
    pre-compute embeddings off-line and load them here.
    """

    name: str
    k: int

    def select(
        self,
        *,
        main_episode: EpisodeRef,
        experiment_dir: Path,
        all_refs: Sequence[EpisodeRef],
    ) -> list[Path]: ...


def _filter_main_out(refs: Sequence[EpisodeRef], main: EpisodeRef) -> list[EpisodeRef]:
    """Drop the main episode from a candidate list."""
    return [r for r in refs if r.episode_dir != main.episode_dir]


@dataclass
class SameTaskDifferentAgent:
    """Pick episodes with the same `sample_id` but a different `evaluation_id`.

    Useful for "did other agents fail this task the same way?" Reads each
    candidate's `episode_record.json` to compare; cheap because the records
    are already on disk.
    """

    k: int = 3
    name: str = "same_task_different_agent"

    def select(
        self,
        *,
        main_episode: EpisodeRef,
        experiment_dir: Path,
        all_refs: Sequence[EpisodeRef],
    ) -> list[Path]:
        if main_episode.record is None:
            return []
        main_sample = main_episode.record.sample_id
        main_eval = main_episode.record.evaluation_id
        out: list[Path] = []
        for ref in _filter_main_out(all_refs, main_episode):
            if ref.record is None:
                continue
            if ref.record.sample_id == main_sample and ref.record.evaluation_id != main_eval:
                out.append(ref.episode_dir)
                if len(out) >= self.k:
                    break
        return out


@dataclass
class SameAgentPreviousIteration:
    """Pick episodes from the same agent on the same task at earlier timestamps.

    Useful for "did the agent already try this and learn anything?" Compares
    `evaluation_id` (same agent run) and `sample_id` (same task), keeps the K
    earliest by `timestamp`."""

    k: int = 2
    name: str = "same_agent_previous_iteration"

    def select(
        self,
        *,
        main_episode: EpisodeRef,
        experiment_dir: Path,
        all_refs: Sequence[EpisodeRef],
    ) -> list[Path]:
        if main_episode.record is None:
            return []
        main_eval = main_episode.record.evaluation_id
        main_sample = main_episode.record.sample_id
        main_ts = main_episode.record.timestamp
        candidates: list[tuple[float, Path]] = []
        for ref in _filter_main_out(all_refs, main_episode):
            if ref.record is None:
                continue
            if ref.record.evaluation_id != main_eval or ref.record.sample_id != main_sample:
                continue
            if ref.record.timestamp >= main_ts:
                continue
            candidates.append((ref.record.timestamp, ref.episode_dir))
        # Earliest first; bound by k.
        candidates.sort(key=lambda t: t[0])
        return [p for _, p in candidates[: self.k]]


@dataclass
class TopKBySimilarityStub:
    """Placeholder for a real similarity scorer.

    Today: deterministic hash of trajectory_id, sorted ascending — it returns a
    stable list per main episode but the choice has no semantic meaning. The
    concrete scorer (e.g. via cosine similarity over task descriptions) is a
    follow-up; this stub keeps the call-site contract working."""

    k: int = 3
    name: str = "top_k_by_similarity_stub"

    def select(
        self,
        *,
        main_episode: EpisodeRef,
        experiment_dir: Path,
        all_refs: Sequence[EpisodeRef],
    ) -> list[Path]:
        candidates = _filter_main_out(all_refs, main_episode)
        # Stable ranking by SHA-256(trajectory_id) — placeholder for real cosine
        # similarity over task-description embeddings.
        ranked = sorted(
            candidates,
            key=lambda r: hashlib.sha256(r.trajectory_id.encode()).hexdigest(),
        )
        return [r.episode_dir for r in ranked[: self.k]]


__all__ = [
    "EpisodeRef",
    "discover_episodes",
    "select_episodes",
    "load_episode_record",
    "Selector",
    "SameTaskDifferentAgent",
    "SameAgentPreviousIteration",
    "TopKBySimilarityStub",
]
