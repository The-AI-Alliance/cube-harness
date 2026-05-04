"""Shared helpers for xray tests and screenshot tools."""

from __future__ import annotations

import socket
import time
import urllib.request
from pathlib import Path

from cube_harness.core import Trajectory
from cube_harness.episode_status import EpisodeStatus
from cube_harness.storage import FileStorage

# ---------------------------------------------------------------------------
# Synthetic scenarios used by both the e2e tests and the screenshot script.
# ---------------------------------------------------------------------------

# Base scenarios — stable, used by pytest assertions.
SCENARIOS: list[dict] = [
    {
        "traj_id": "task_1_ep0",
        "task_id": "task_1",
        "agent_name": "agent_a",
        "status": "COMPLETED",
        "reward": 1.0,
        "retry_count": 0,
    },
    {
        "traj_id": "task_1_ep1",
        "task_id": "task_1",
        "agent_name": "agent_a",
        "status": "COMPLETED",
        "reward": 0.0,
        "retry_count": 0,
    },
    {
        "traj_id": "task_2_ep0",
        "task_id": "task_2",
        "agent_name": "agent_a",
        "status": "MAX_STEPS_REACHED",
        "reward": 0.0,
        "retry_count": 1,
    },
    {
        "traj_id": "task_3_ep0",
        "task_id": "task_3",
        "agent_name": "agent_a",
        "status": "FAILED",
        "reward": 0.0,
        "retry_count": 0,
        "error_type": "RuntimeError",
        "error_message": "Environment crashed unexpectedly",
    },
    {
        "traj_id": "task_4_ep0",
        "task_id": "task_4",
        "agent_name": "agent_a",
        "status": "STALE",
        "reward": 0.0,
        "retry_count": 0,
    },
]

# Extended scenarios — richer error info, extra in-flight tasks; for visual review.
EXTENDED_SCENARIOS: list[dict] = SCENARIOS + [
    {
        "traj_id": "task_5_ep0",
        "task_id": "task_5",
        "agent_name": "agent_a",
        "status": "RUNNING",
        "reward": 0.0,
        "retry_count": 0,
    },
    {
        "traj_id": "task_6_ep0",
        "task_id": "task_6",
        "agent_name": "agent_a",
        "status": "QUEUED",
        "reward": 0.0,
        "retry_count": 0,
    },
    {
        "traj_id": "task_7_ep0",
        "task_id": "task_7",
        "agent_name": "agent_a",
        "status": "CANCELLED",
        "reward": 0.0,
        "retry_count": 0,
    },
]


def build_experiment(exp_dir: Path, scenarios: list[dict] | None = None) -> None:
    """Write a minimal on-disk experiment from scenario dicts.

    Each scenario may contain:
      traj_id, task_id, agent_name, status, reward, retry_count,
      error_type (optional), error_message (optional)
    """
    if scenarios is None:
        scenarios = SCENARIOS
    storage = FileStorage(exp_dir)
    for s in scenarios:
        traj = Trajectory(
            id=s["traj_id"],
            metadata={"task_id": s["task_id"], "agent_name": s["agent_name"]},
            start_time=1000.0,
            end_time=1010.0 if s["status"] in ("COMPLETED", "FAILED") else None,
            reward_info={"reward": s["reward"]} if s["reward"] is not None else None,
        )
        storage.save_trajectory(traj)
        ep_status = EpisodeStatus(
            status=s["status"],
            task_id=s["task_id"],
            episode_id=0,
            started_at=1000.0,
            retry_count=s.get("retry_count", 0),
            error_type=s.get("error_type"),
            error_message=s.get("error_message"),
        )
        storage.write_episode_status(s["traj_id"], ep_status)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def wait_for_server(url: str, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"XRay server did not start at {url} within {timeout}s")
