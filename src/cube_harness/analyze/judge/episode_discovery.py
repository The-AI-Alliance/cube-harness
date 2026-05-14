"""Scan and filter the episodes inside an experiment directory.

`discover_episodes(<experiment_dir>)` walks `<experiment_dir>/episodes/` and
returns one `EpisodeRef` per subdirectory; if the path itself looks like a
single episode (has `steps/`), it is returned as the only ref.

`select_episodes(refs, *, ids=..., failures_only=..., overwrite=...)` filters
the pool. Selection is intentionally explicit — random sampling was removed
because experiments now embed their own task selection upfront; the
meta-agent passes `ids` lists when it wants a subset.

Related-trajectory selection (Selector Protocol + the three concrete
selectors) lives in `selectors.py`.
"""

from __future__ import annotations

import json
import logging
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
    failures_only: bool = False,
    overwrite: bool = False,
) -> list[EpisodeRef]:
    """Filter the episode pool.

    `ids` is an explicit override: when set, the named episodes are returned
    verbatim regardless of `failures_only` / already-judged — the caller
    typed exactly what they want.

    Otherwise: pool = refs, then optional `failures_only` filter, then drop
    already-judged unless `overwrite=True`. The result is the full eligible
    set — random sampling lives elsewhere if a caller really needs it.
    """
    if ids:
        wanted = set(ids)
        return [r for r in refs if r.trajectory_id in wanted or r.trajectory_id.split("_ep")[0] in wanted]

    pool = refs
    if failures_only:
        pool = [r for r in pool if (r.record is not None and not r.record.is_correct)]
    if not overwrite:
        pool = [r for r in pool if not (r.record is not None and r.record.judge_output is not None)]
    return pool


__all__ = ["EpisodeRef", "discover_episodes", "select_episodes", "load_episode_record"]
