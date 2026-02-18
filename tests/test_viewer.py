"""Tests for agentlab2.viewer module — live mode, incremental loading, and trajectory status."""

import os
import time
from pathlib import Path

from agentlab2.core import (
    AgentOutput,
    Content,
    EnvironmentOutput,
    Observation,
    StepError,
    Trajectory,
    TrajectoryStep,
)
from agentlab2.storage import FileStorage
from agentlab2.viewer import ViewerState, _trajectory_status


def _make_env_output(reward: float = 0.0, done: bool = False) -> EnvironmentOutput:
    return EnvironmentOutput(
        obs=Observation(contents=[Content(data="obs")]),
        reward=reward,
        done=done,
        info={},
    )


def _make_agent_output() -> AgentOutput:
    return AgentOutput(actions=[])


def _save_trajectory(
    storage: FileStorage,
    traj_id: str,
    task_id: str = "task_1",
    start_time: float = 100.0,
    end_time: float | None = 200.0,
    reward: float = 1.0,
    done: bool = True,
    with_error: bool = False,
) -> Trajectory:
    steps = [
        TrajectoryStep(output=_make_env_output(), start_time=start_time, end_time=start_time + 1),
        TrajectoryStep(output=_make_agent_output(), start_time=start_time + 1, end_time=start_time + 2),
        TrajectoryStep(
            output=_make_env_output(reward=reward, done=done),
            start_time=start_time + 2,
            end_time=start_time + 3,
        ),
    ]
    if with_error:
        error_output = AgentOutput(actions=[], error=StepError(error_type="RuntimeError", exception_str="boom", stack_trace=""))
        steps.insert(1, TrajectoryStep(output=error_output, start_time=start_time + 0.5, end_time=start_time + 0.6))

    reward_info = {"reward": reward, "done": done} if end_time is not None else {}
    traj = Trajectory(
        id=traj_id,
        metadata={"task_id": task_id},
        steps=steps,
        start_time=start_time,
        end_time=end_time,
        reward_info=reward_info,
    )
    storage.save_trajectory(traj)
    return traj


class TestTrajectoryStatus:
    def test_running(self) -> None:
        traj = Trajectory(id="t1", metadata={}, start_time=1.0, end_time=None)
        assert _trajectory_status(traj) == "running"

    def test_success(self) -> None:
        traj = Trajectory(id="t1", metadata={}, start_time=1.0, end_time=2.0, reward_info={"reward": 1.0})
        assert _trajectory_status(traj) == "success"

    def test_completed_zero_reward(self) -> None:
        traj = Trajectory(id="t1", metadata={}, start_time=1.0, end_time=2.0, reward_info={"reward": 0.0})
        assert _trajectory_status(traj) == "completed"

    def test_error(self) -> None:
        error_step = TrajectoryStep(
            output=AgentOutput(actions=[], error=StepError(error_type="RuntimeError", exception_str="boom", stack_trace=""))
        )
        traj = Trajectory(
            id="t1", metadata={}, steps=[error_step], start_time=1.0, end_time=2.0, reward_info={"reward": 0.0}
        )
        assert _trajectory_status(traj) == "error"

    def test_no_reward_info(self) -> None:
        traj = Trajectory(id="t1", metadata={}, start_time=1.0, end_time=2.0)
        assert _trajectory_status(traj) == "completed"


class TestViewerStateLoadExperiment:
    def test_load_populates_trajectories(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        _save_trajectory(storage, "traj_1")
        _save_trajectory(storage, "traj_2")

        state = ViewerState(results_dir=tmp_dir.parent)
        result = state.load_experiment(tmp_dir)

        assert result == {"loaded": 2}
        assert len(state.trajectories) == 2
        assert "traj_1" in state.trajectories
        assert "traj_2" in state.trajectories

    def test_load_sets_completed_ids(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        _save_trajectory(storage, "done_traj", end_time=200.0)
        _save_trajectory(storage, "running_traj", end_time=None)

        state = ViewerState(results_dir=tmp_dir.parent)
        state.load_experiment(tmp_dir)

        assert "done_traj" in state._completed_ids
        assert "running_traj" not in state._completed_ids

    def test_load_sets_current_exp_dir(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        _save_trajectory(storage, "t1")

        state = ViewerState(results_dir=tmp_dir.parent)
        state.load_experiment(tmp_dir)

        assert state._current_exp_dir == tmp_dir

    def test_load_detects_expected_total(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        _save_trajectory(storage, "t1")

        config_dir = tmp_dir / "episode_configs"
        config_dir.mkdir()
        (config_dir / "episode_0_task_task_1.json").write_text("{}")
        (config_dir / "episode_1_task_task_2.json").write_text("{}")
        (config_dir / "episode_2_task_task_3.json").write_text("{}")

        state = ViewerState(results_dir=tmp_dir.parent)
        state.load_experiment(tmp_dir)

        assert state._expected_total == 3

    def test_load_no_trajectories_dir(self, tmp_dir: Path) -> None:
        state = ViewerState(results_dir=tmp_dir.parent)
        result = state.load_experiment(tmp_dir)

        assert "error" in result


class TestViewerStateRefresh:
    def test_refresh_detects_new_trajectory(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        _save_trajectory(storage, "traj_1")

        state = ViewerState(results_dir=tmp_dir.parent)
        state.load_experiment(tmp_dir)
        assert len(state.trajectories) == 1

        storage2 = FileStorage(tmp_dir)
        _save_trajectory(storage2, "traj_2", start_time=300.0, end_time=400.0)

        changed = state.refresh_experiment()
        assert changed is True
        assert len(state.trajectories) == 2
        assert "traj_2" in state.trajectories

    def test_refresh_skips_completed(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        _save_trajectory(storage, "done_traj", end_time=200.0)

        state = ViewerState(results_dir=tmp_dir.parent)
        state.load_experiment(tmp_dir)
        assert "done_traj" in state._completed_ids

        changed = state.refresh_experiment()
        assert changed is False

    def test_refresh_reloads_in_progress(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        _save_trajectory(storage, "running_traj", end_time=None)

        state = ViewerState(results_dir=tmp_dir.parent)
        state.load_experiment(tmp_dir)
        assert state.trajectories["running_traj"].end_time is None
        original_steps = len(state.trajectories["running_traj"].steps)

        traj_dir = tmp_dir / "trajectories"
        with open(traj_dir / "running_traj.jsonl", "a") as f:
            new_step = TrajectoryStep(output=_make_agent_output(), start_time=105.0, end_time=106.0)
            f.write(new_step.model_dump_json(serialize_as_any=True) + "\n")

        jsonl_path = traj_dir / "running_traj.jsonl"
        os.utime(jsonl_path, (time.time() + 10, time.time() + 10))

        changed = state.refresh_experiment()
        assert changed is True
        assert len(state.trajectories["running_traj"].steps) == original_steps + 1

    def test_refresh_no_exp_dir(self) -> None:
        state = ViewerState(results_dir=Path("/nonexistent"))
        assert state.refresh_experiment() is False

    def test_is_experiment_complete(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        _save_trajectory(storage, "t1", end_time=200.0)
        _save_trajectory(storage, "t2", end_time=None)

        state = ViewerState(results_dir=tmp_dir.parent)
        state.load_experiment(tmp_dir)
        assert state.is_experiment_complete() is False

        state.trajectories["t2"] = Trajectory(
            id="t2", metadata={}, start_time=1.0, end_time=2.0, reward_info={"reward": 0.0}
        )
        state._completed_ids.add("t2")
        assert state.is_experiment_complete() is True


class TestFileStorageListing:
    def test_list_trajectory_ids(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        _save_trajectory(storage, "a")
        _save_trajectory(storage, "b")

        ids = storage.list_trajectory_ids()
        assert sorted(ids) == ["a", "b"]

    def test_list_trajectory_ids_excludes_archived(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        _save_trajectory(storage, "a")

        traj_dir = tmp_dir / "trajectories"
        (traj_dir / "old.archived_123.metadata.json").write_text("{}")

        ids = storage.list_trajectory_ids()
        assert ids == ["a"]

    def test_list_trajectory_ids_empty(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        assert storage.list_trajectory_ids() == []

    def test_list_trajectory_ids_with_mtime(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        _save_trajectory(storage, "a")
        _save_trajectory(storage, "b")

        result = storage.list_trajectory_ids_with_mtime()
        assert set(result.keys()) == {"a", "b"}
        assert all(isinstance(v, float) for v in result.values())

    def test_list_trajectory_ids_with_mtime_empty(self, tmp_dir: Path) -> None:
        storage = FileStorage(tmp_dir)
        assert storage.list_trajectory_ids_with_mtime() == {}
