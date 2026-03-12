"""Deterministic debug agent for testing terminalbench-cube end-to-end without an LLM.

Public API
----------
get_debug_benchmark()         → TerminalBenchBenchmark
make_debug_agent(task_id)     → DebugAgent
"""

from __future__ import annotations

import logging
import os

from cube.benchmark import Benchmark
from cube.core import Action, ActionSchema, Observation
from cube.backends.daytona import DaytonaContainerBackend
from terminalbench_cube.benchmark import TerminalBenchBenchmark

logger = logging.getLogger(__name__)

# Each debug task replays a fixed action sequence.
# These actions explore the environment and then stop — they don't solve
# the task (that would require task-specific knowledge), but they exercise
# the full pipeline: container launch → reset → step → evaluate → close.
_TASK_ACTIONS: dict[str, list[Action]] = {
    "fix-git": [
        Action(name="bash", arguments={"command": "ls -la /app"}),
        Action(name="bash", arguments={"command": "cd /app/personal-site && git branch -a"}),
        Action(name="final_step", arguments={}),
    ],
    "overfull-hbox": [
        Action(name="bash", arguments={"command": "ls -la /app"}),
        Action(name="read_file", arguments={"path": "/app/main.tex"}),
        Action(name="final_step", arguments={}),
    ],
}


class DebugAgent:
    """Deterministic agent that replays a fixed action sequence."""

    def __init__(self, task_id: str) -> None:
        if task_id not in _TASK_ACTIONS:
            raise ValueError(f"No debug actions for {task_id!r}. Known: {list(_TASK_ACTIONS)}")
        self._task_id = task_id
        self._step = 0
        self._actions = list(_TASK_ACTIONS[task_id])

    def get_action(self, obs: Observation) -> Action:
        if self._step >= len(self._actions):
            raise StopIteration(f"All actions exhausted for task {self._task_id!r}")
        action = self._actions[self._step]
        self._step += 1
        return action

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        return self.get_action(obs)


def get_debug_benchmark() -> "Benchmark":
    """Return a TerminalBenchBenchmark scoped to the debug tasks.

    The harness will call benchmark.install() and benchmark.setup() on the
    returned instance, iterate benchmark.get_task_configs() to discover tasks,
    and call benchmark.close() at the end to free resources.
    """
    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        raise RuntimeError("DAYTONA_API_KEY environment variable is required for cube test terminalbench-cube")

    return TerminalBenchBenchmark(
        container_backend=DaytonaContainerBackend(api_key=api_key),
    ).subset_from_list(list(_TASK_ACTIONS), benchmark_name_suffix="debug")


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


if __name__ == "__main__":
    import sys

    import terminalbench_cube.debug as _this_module
    from cube.testing import run_debug_suite

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    results = run_debug_suite("terminalbench-cube", _this_module)
    failed = [r for r in results if r["error"]]
    sys.exit(1 if failed else 0)
