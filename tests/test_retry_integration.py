"""End-to-end integration test for the episode-status retry mechanism.

A single Ray-based run over a 4-task benchmark covers ~80% of the retry machinery:
pre-claim, step-boundary heartbeat, driver-side stale-heartbeat kill, CANCELLED
status, auto-retry rounds, retry_count cap, and per-attempt archives.

Each task's "scenario" is a list of behaviours indexed by attempt number
(0 = original, 1.. = retries):

| Episode    | Scenarios                | Expected final | retry_count | archives |
|------------|--------------------------|----------------|-------------|----------|
| task_succeed | ["succeed"]            | COMPLETED      | 0           | 0        |
| task_flaky   | ["fail","fail","ok"]   | COMPLETED      | 2           | 2        |
| task_dead    | ["fail"]*4             | FAILED (cap)   | 3           | 3        |
| task_hang    | ["hang","succeed"]     | COMPLETED      | 1           | 1        |
"""

from __future__ import annotations

import fcntl
import time
from pathlib import Path

import pytest
import ray
from cube.benchmark import Benchmark as CubeBenchmark
from cube.benchmark import BenchmarkMetadata
from cube.core import Action, Observation
from cube.task import Task as CubeTask
from cube.task import TaskConfig as CubeTaskConfig
from cube.task import TaskMetadata

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput
from cube_harness.episode_status import STATUS_FILENAME
from cube_harness.exp_runner import run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.storage import ARCHIVED_MARKER, EPISODES_DIR, FileStorage
from tests.conftest import MockToolConfig

pytestmark = pytest.mark.slow


SCENARIOS = {
    "task_succeed": ["succeed"],
    "task_flaky": ["fail", "fail", "succeed"],
    "task_dead": ["fail", "fail", "fail", "fail"],
    "task_hang": ["hang", "succeed"],
}


# --- Debug task / benchmark ---


class DebugCubeTask(CubeTask):
    """Cube task that exposes its own task_id in the initial observation.

    The DebugAgent reads this to look up the per-task scripted scenario.
    """

    def reset(self) -> tuple[Observation, dict]:
        return Observation.from_text(f"task_id={self.metadata.id}"), {"task_id": self.metadata.id}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        return 1.0, {"success": True}


class DebugCubeTaskConfig(CubeTaskConfig):
    def make(self, runtime_context=None, container_backend=None) -> DebugCubeTask:
        return DebugCubeTask(
            metadata=TaskMetadata(id=self.task_id),
            tool_config=self.tool_config or MockToolConfig(),
        )


class DebugBenchmark(CubeBenchmark):
    benchmark_metadata = BenchmarkMetadata(
        name="debug-retry",
        version="0.1.0",
        description="Scripted scenarios for retry-mechanism integration test",
    )
    task_metadata = {tid: TaskMetadata(id=tid) for tid in SCENARIOS}
    task_config_class = DebugCubeTaskConfig

    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


# --- Debug agent ---


def _next_attempt(counter_dir: str, task_id: str) -> int:
    """Atomically read+increment a per-task attempt counter on disk.

    Uses fcntl to serialise concurrent writers (no contention expected within a
    single round, but cheap insurance against future parallelism bugs).
    """
    Path(counter_dir).mkdir(parents=True, exist_ok=True)
    path = Path(counter_dir) / f"{task_id}.txt"
    with open(path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read().strip()
            current = int(raw) if raw else 0
            f.seek(0)
            f.truncate()
            f.write(str(current + 1))
            return current
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class DebugAgentConfig(AgentConfig):
    """Agent config carrying scenarios + counter dir.

    Both fields are picklable plain data, so Ray can ship the config to workers.
    """

    counter_dir: str
    hang_seconds: float = 60.0
    name: str = "debug_agent"

    def make(self, action_set=None, **kwargs) -> "DebugAgent":
        return DebugAgent(config=self)


class DebugAgent(Agent):
    name = "DebugAgent"
    description = "Scripted agent that reads scenarios per-task per-attempt"
    input_content_types = ["text"]
    output_content_types = ["action"]

    def __init__(self, config: DebugAgentConfig):
        super().__init__(config)
        self.config = config

    def step(self, obs: Observation) -> AgentOutput:
        # Initial observation embeds task_id; pull it out.
        text = obs.contents[0].data if obs.contents else ""
        task_id = text.replace("task_id=", "").strip()
        scenarios = SCENARIOS[task_id]
        attempt = _next_attempt(self.config.counter_dir, task_id)
        # Saturate at the last scenario if attempts exceed the script length.
        behavior = scenarios[min(attempt, len(scenarios) - 1)]

        if behavior == "succeed":
            return AgentOutput(actions=[Action(name="final_step", arguments={})])
        if behavior == "fail":
            raise RuntimeError(f"scripted failure: {task_id} attempt {attempt}")
        if behavior == "hang":
            time.sleep(self.config.hang_seconds)
            return AgentOutput(actions=[Action(name="final_step", arguments={})])
        raise ValueError(f"Unknown behavior: {behavior}")


# --- The test ---


@pytest.fixture(autouse=True)
def _ray_shutdown_between_tests():
    yield
    if ray.is_initialized():
        ray.shutdown()


def test_retry_machinery_end_to_end(tmp_dir: Path) -> None:
    """4-task Ray run exercises the full retry mechanism."""
    counter_dir = str(tmp_dir / "_attempt_counters")
    agent_config = DebugAgentConfig(counter_dir=counter_dir, hang_seconds=10.0)
    benchmark = DebugBenchmark()

    exp = Experiment(
        name="retry_integration",
        output_dir=tmp_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_retries=3,
    )

    result = run_with_ray(
        exp,
        n_cpus=2,
        ray_poll_timeout=0.5,
        step_timeout_s=1.5,
        cancel_grace_s=0.5,
        orphan_threshold_s=30.0,
        max_retry_rounds=3,
    )

    storage = FileStorage(tmp_dir)
    statuses = storage.list_episode_statuses()
    # All four episode dirs exist.
    assert set(statuses.keys()) == {f"{tid}_ep{i}" for i, tid in enumerate(SCENARIOS)}

    # task_succeed: COMPLETED on first try, retry_count 0, no archives.
    s0 = statuses["task_succeed_ep0"]
    assert s0.status == "COMPLETED", s0
    assert s0.retry_count == 0
    assert _archive_count(tmp_dir, "task_succeed_ep0") == 0

    # task_flaky: 2 retries, finally COMPLETED. retry_count = 2 on the live attempt.
    # 2 archived dirs (the two failed attempts).
    s1 = statuses["task_flaky_ep1"]
    assert s1.status == "COMPLETED", s1
    assert s1.retry_count == 2
    assert _archive_count(tmp_dir, "task_flaky_ep1") == 2

    # task_dead: 4 attempts, all failed. retry_count = 3 (capped). 3 archives.
    s2 = statuses["task_dead_ep2"]
    assert s2.status == "FAILED", s2
    assert s2.retry_count == 3
    assert s2.error_type == "RuntimeError"
    assert _archive_count(tmp_dir, "task_dead_ep2") == 3

    # task_hang: first attempt CANCELLED via step-timeout, second succeeds.
    # 1 archive (the cancelled attempt).
    s3 = statuses["task_hang_ep3"]
    assert s3.status == "COMPLETED", s3
    assert s3.retry_count == 1
    assert _archive_count(tmp_dir, "task_hang_ep3") == 1

    # The CANCELLED attempt is preserved in the archive — verify error_type.
    archived_status = _read_archived_status(tmp_dir, "task_hang_ep3")
    assert archived_status is not None
    assert archived_status["status"] == "CANCELLED", archived_status
    assert archived_status["error_type"] == "StepTimeout"

    # Aggregated result: 3 successful trajectories, 1 failure (task_dead).
    assert "task_dead_ep2" in result.failures
    assert {"task_succeed_ep0", "task_flaky_ep1", "task_hang_ep3"}.issubset(result.trajectories.keys())


def _archive_count(output_dir: Path, traj_id: str) -> int:
    base = output_dir / EPISODES_DIR
    return sum(1 for p in base.iterdir() if p.name.startswith(f"{traj_id}{ARCHIVED_MARKER}"))


def _read_archived_status(output_dir: Path, traj_id: str) -> dict | None:
    """Read status.json from any archived dir for `traj_id` (returns the first found)."""
    import json

    base = output_dir / EPISODES_DIR
    for p in base.iterdir():
        if p.name.startswith(f"{traj_id}{ARCHIVED_MARKER}"):
            status_path = p / STATUS_FILENAME
            if status_path.exists():
                return json.loads(status_path.read_text())
    return None
