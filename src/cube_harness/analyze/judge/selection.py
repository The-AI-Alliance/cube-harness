"""Episode discovery and selection."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from cube_harness.eval_log import EPISODE_RECORD_FILENAME, EpisodeRecord

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRef:
    trajectory_id: str
    episode_dir: Path
    record_path: Path
    record: EpisodeRecord | None  # None if record file missing


def _load_episode_record(record_path: Path) -> EpisodeRecord | None:
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
                    record=_load_episode_record(experiment_dir / EPISODE_RECORD_FILENAME),
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
                record=_load_episode_record(record_path),
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
