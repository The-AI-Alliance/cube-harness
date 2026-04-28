"""Per-episode status file used to drive resume/retry decisions.

`status.json` is the control-plane sibling of the trajectory data. It is read
and written without deserialising trajectory steps, so the driver can make
retry decisions in O(N) file reads rather than O(N×steps) deserialisations.

See `openspec/changes/episode-status/proposal.md` for the full design.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Status = Literal["RUNNING", "COMPLETED", "FAILED", "CANCELLED", "STALE"]

TERMINAL_STATUSES: frozenset[Status] = frozenset({"COMPLETED", "FAILED", "CANCELLED", "STALE"})
RETRIABLE_STATUSES: frozenset[Status] = frozenset({"FAILED", "CANCELLED", "STALE"})

STATUS_FILENAME = "status.json"


@dataclass
class EpisodeStatus:
    """Lifecycle snapshot of an episode.

    Written by both the worker (real progress) and the driver (pre-claim,
    cancellation). `last_heartbeat_at` is `None` until the worker enters
    `_run_loop`; this is the "queued in Ray" signal for the driver poll.
    """

    status: Status
    task_id: str
    episode_id: int
    started_at: float
    ended_at: float | None = None
    last_heartbeat_at: float | None = None
    current_step: int = 0
    reward: float | None = None
    had_step_errors: bool = False
    error_type: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "EpisodeStatus":
        return cls(**json.loads(raw))

    @classmethod
    def read(cls, path: Path) -> "EpisodeStatus | None":
        if not path.exists():
            return None
        try:
            return cls.from_json(path.read_text())
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def write(self, path: Path) -> None:
        """Atomic write: tmp sibling + os.replace() so partial files are never observed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json())
        os.replace(tmp, path)


def next_retry_count(prior: "EpisodeStatus | None") -> int:
    """Return the retry_count for a new attempt, given the prior status (if any).

    - No prior: 0 (original attempt).
    - Prior was RUNNING: same retry_count (e.g., re-pre-claim within a round, idempotent).
    - Prior was terminal (FAILED/CANCELLED/STALE/COMPLETED): prior + 1.
    """
    if prior is None:
        return 0
    if prior.status == "RUNNING":
        return prior.retry_count
    return prior.retry_count + 1
