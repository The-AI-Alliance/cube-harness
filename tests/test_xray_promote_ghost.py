"""Unit tests for _promote_ghost_episodes (xray_utils).

Covers the fix in PR #372: QUEUED episodes must never be promoted to STALE
by the viewer's refresh path — only sweep_stale_statuses() (runner-owned)
may touch them.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from cube_harness.analyze.xray_utils import _promote_ghost_episodes
from cube_harness.episode_status import STATUS_FILENAME, EpisodeStatus


def _write_status(ep_dir: Path, status: str, age_seconds: float) -> None:
    ep_dir.mkdir(parents=True, exist_ok=True)
    s = EpisodeStatus(
        status=status,
        task_id="test-task",
        episode_id=0,
        started_at=time.time() - age_seconds,
        last_heartbeat_at=time.time() - age_seconds,
    )
    s.write(ep_dir / STATUS_FILENAME)


def test_queued_episode_never_promoted(tmp_path: Path) -> None:
    """QUEUED episodes are not touched regardless of age."""
    ep = tmp_path / "episodes" / "ep_000"
    _write_status(ep, "QUEUED", age_seconds=99_999)

    _promote_ghost_episodes(tmp_path)

    after = EpisodeStatus.read(ep / STATUS_FILENAME)
    assert after is not None
    assert after.status == "QUEUED"


def test_old_running_episode_promoted_to_stale(tmp_path: Path) -> None:
    """RUNNING episode whose heartbeat is beyond GHOST_TIMEOUT becomes STALE."""
    ep = tmp_path / "episodes" / "ep_001"
    _write_status(ep, "RUNNING", age_seconds=99_999)

    _promote_ghost_episodes(tmp_path)

    after = EpisodeStatus.read(ep / STATUS_FILENAME)
    assert after is not None
    assert after.status == "STALE"


def test_fresh_running_episode_not_promoted(tmp_path: Path) -> None:
    """RUNNING episode with a recent heartbeat is left alone."""
    ep = tmp_path / "episodes" / "ep_002"
    _write_status(ep, "RUNNING", age_seconds=30)

    _promote_ghost_episodes(tmp_path)

    after = EpisodeStatus.read(ep / STATUS_FILENAME)
    assert after is not None
    assert after.status == "RUNNING"


@pytest.mark.parametrize("terminal_status", ["COMPLETED", "FAILED", "CANCELLED", "STALE"])
def test_terminal_episodes_not_touched(tmp_path: Path, terminal_status: str) -> None:
    """Already-terminal episodes are never re-promoted."""
    ep = tmp_path / "episodes" / "ep_003"
    _write_status(ep, terminal_status, age_seconds=99_999)

    _promote_ghost_episodes(tmp_path)

    after = EpisodeStatus.read(ep / STATUS_FILENAME)
    assert after is not None
    assert after.status == terminal_status
