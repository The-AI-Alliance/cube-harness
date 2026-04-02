"""Deterministic debug agent for testing WAATask end-to-end with a live VM.

Each debug task in debug_tasks.json uses an infeasible evaluator: the agent
calls ``fail()`` and the evaluator returns reward=1.0. This validates the full
CUBE task loop — VM startup, snapshot restore, screenshot capture, and
evaluation — without requiring the agent to perform any GUI actions.

Public API
----------
make_debug_agent(task_id)    → DebugAgent
get_debug_benchmark()        → WAABenchmark

Usage::

    # Run all debug tasks and print a JSON report
    python -m waa_cube.debug
"""

from __future__ import annotations

import logging
from pathlib import Path

from cube.core import Action, ActionSchema, Observation
from cube.vm import VMBackend

from waa_cube.benchmark import WAABenchmark
from waa_cube.computer import ComputerConfig
from waa_cube.vm_backend.backend import WAADockerVMBackend

logger = logging.getLogger(__name__)

_TASKS_FILE = Path(__file__).parent / "debug_tasks.json"

# ---------------------------------------------------------------------------
# Hardcoded action sequences per task ID
# ---------------------------------------------------------------------------

_TASK_ACTIONS: dict[str, list[Action]] = {
    "waa-debug-infeasible": [
        # Signal that the task is infeasible — triggers reward=1.0 from the evaluator
        Action(name="fail", arguments={}),
    ],
}


# ---------------------------------------------------------------------------
# DebugAgent
# ---------------------------------------------------------------------------


class DebugAgent:
    """Deterministic debug agent that replays a fixed action sequence for a given task.

    Interface matches the stress-test spec:
        agent = make_debug_agent(task_id)
        action = agent.get_action(obs)

    Args:
        task_id: ID of the debug task to run. Must match a key in _TASK_ACTIONS.

    Raises:
        ValueError: If task_id has no registered action sequence.
    """

    def __init__(self, task_id: str) -> None:
        if task_id not in _TASK_ACTIONS:
            raise ValueError(
                f"No debug actions registered for task {task_id!r}. Known tasks: {list(_TASK_ACTIONS)}"
            )
        self._task_id = task_id
        self._step = 0
        self._actions = list(_TASK_ACTIONS[task_id])
        logger.debug(
            "[DebugAgent] Initialised for task=%r with %d actions",
            task_id,
            len(self._actions),
        )

    def get_action(self, obs: Observation) -> Action:
        """Return the next predetermined action."""
        if self._step >= len(self._actions):
            raise StopIteration(
                f"[DebugAgent] task={self._task_id!r}: all {len(self._actions)} actions exhausted"
            )
        action = self._actions[self._step]
        logger.info(
            "[DebugAgent] task=%r  step=%d/%d  action=%s  args=%s",
            self._task_id,
            self._step + 1,
            len(self._actions),
            action.name,
            action.arguments or "",
        )
        self._step += 1
        return action

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        """Callable shorthand — delegates to get_action() for task-loop compatibility."""
        return self.get_action(obs)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_debug_benchmark(vm_backend: VMBackend | None = None) -> WAABenchmark:
    """Return a WAABenchmark scoped to the debug tasks.

    Uses debug_tasks.json as the task source — no evaluation_examples_windows/
    directory required. The caller (cube.testing) is responsible for calling
    install() and setup().

    Args:
        vm_backend: Backend to use. Defaults to WAADockerVMBackend (requires
                    Docker and a pre-built Windows QEMU image).
    """
    return WAABenchmark(
        tasks_file=str(_TASKS_FILE),
        default_tool_config=ComputerConfig(),
        vm_backend=vm_backend or WAADockerVMBackend(),
    )


def make_debug_agent(task_id: str) -> DebugAgent:
    """Return a fresh DebugAgent for the given task_id."""
    return DebugAgent(task_id)


# ---------------------------------------------------------------------------
# __main__ — run all debug tasks, print JSON report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import waa_cube.debug as _mod
    from cube.testing import run_debug_suite

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    results = run_debug_suite("waa-cube", _mod)

    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] <= 0]
    sys.exit(1 if failed else 0)
