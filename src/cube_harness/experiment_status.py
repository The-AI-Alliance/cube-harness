"""Experiment-level status file, heartbeated by the runner driver process.

`experiment_status.json` sits at the root of `output_dir` alongside
`experiment_config.json`. It is written by the runner (not by Ray workers)
so the XRay viewer can determine whether the experiment driver is still alive
— and, if it is not, promote orphaned QUEUED episodes to STALE.

See also: `episode_status.py` for the per-episode equivalent.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

EXPERIMENT_STATUS_FILENAME = "experiment_status.json"


@dataclass
class ExperimentStatus:
    """Lifecycle snapshot of an entire experiment run, written by the driver.

    Fields
    ------
    status
        Terminal statuses are written once at shutdown; RUNNING is heartbeated.
    mode
        "ray"        — driver runs a Ray cluster; heartbeat comes from the
                       poll loop in _poll_ray (every ~30 s).
        "sequential" — driver IS the worker; heartbeat is updated before each
                       episode.run() call, so it may lag by one episode's
                       wall-clock time. XRay must fall back to episode-level
                       heartbeats before declaring the driver dead.
    pid / host
        Used for informational display and to detect cross-machine restarts.
    ray_dashboard_url
        Populated in "ray" mode when the dashboard URL is available;
        None otherwise.
    completed / failed / stale
        Running counters, updated alongside the heartbeat.
    """

    status: Literal["RUNNING", "COMPLETED", "INTERRUPTED"]
    mode: Literal["ray", "sequential"]
    pid: int
    host: str
    started_at: float
    last_heartbeat_at: float
    total_episodes: int
    completed: int = 0
    failed: int = 0
    stale: int = 0
    ray_dashboard_url: str | None = None
    ended_at: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "ExperimentStatus":
        """Parse, dropping unknown keys for forward compatibility."""
        data = json.loads(raw)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def read(cls, path: Path) -> "ExperimentStatus | None":
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
