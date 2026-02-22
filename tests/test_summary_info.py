"""Tests for agentlab2.summary_info (Issue #148)."""

from pathlib import Path

import pytest

from agentlab2.core import EnvironmentOutput, Observation, Trajectory, TrajectoryStep
from agentlab2.summary_info import (
    SUMMARY_FILENAME,
    aggregate_stats,
    build_summary_info,
    extract_step_stats,
    save_summary_info_sync,
)


def test_extract_step_stats_env_step() -> None:
    obs = Observation.from_text("test")
    env_out = EnvironmentOutput(obs=obs, reward=0.5, done=False)
    step = TrajectoryStep(output=env_out)
    stats = extract_step_stats(step)
    assert stats["reward"] == 0.5
    assert stats["done"] == 0.0


def test_extract_step_stats_env_step_done() -> None:
    obs = Observation.from_text("test")
    env_out = EnvironmentOutput(obs=obs, reward=1.0, done=True)
    step = TrajectoryStep(output=env_out)
    stats = extract_step_stats(step)
    assert stats["reward"] == 1.0
    assert stats["done"] == 1.0


def test_aggregate_stats_empty() -> None:
    assert aggregate_stats([]) == {"cum_steps": 0}


def test_aggregate_stats_single_step() -> None:
    stats_list = [{"reward": 0.5, "done": 0.0}]
    agg = aggregate_stats(stats_list)
    assert agg["cum_steps"] == 1
    assert agg["cum_reward"] == 0.5
    assert agg["max_reward"] == 0.5


def test_aggregate_stats_multiple_steps() -> None:
    stats_list = [
        {"reward": 0.0, "prompt_tokens": 10, "completion_tokens": 5},
        {"reward": 0.5, "prompt_tokens": 20, "completion_tokens": 10},
    ]
    agg = aggregate_stats(stats_list)
    assert agg["cum_steps"] == 2
    assert agg["cum_reward"] == 0.5
    assert agg["cum_prompt_tokens"] == 30
    assert agg["cum_completion_tokens"] == 15
    assert agg["max_prompt_tokens"] == 20
    assert agg["max_completion_tokens"] == 10


def test_build_summary_info_single_trajectory() -> None:
    obs = Observation.from_text("x")
    env_out = EnvironmentOutput(obs=obs, reward=0.5, done=False)
    traj = Trajectory(
        id="t1",
        steps=[TrajectoryStep(output=env_out)],
        metadata={"task_id": "task1"},
    )
    summary = build_summary_info([traj])
    assert summary["n_steps"] == 1
    assert summary["n_trajectories"] == 1
    assert summary["cum_reward"] == 0.5
    assert "stats.cum_steps" in summary
    assert summary["terminated"] is False


def test_save_summary_info_sync(tmp_path: Path) -> None:
    summary = {"n_steps": 1, "cum_reward": 0.5}
    save_summary_info_sync(tmp_path, summary)
    path = tmp_path / SUMMARY_FILENAME
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert data["n_steps"] == 1
    assert data["cum_reward"] == 0.5
