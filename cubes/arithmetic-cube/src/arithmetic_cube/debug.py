"""Deterministic debug agent for testing ArithmeticBenchmark end-to-end without an LLM.

Public API
----------
get_debug_benchmark()         → ArithmeticBenchmark
make_debug_agent(task_id)     → DebugAgent
"""

from __future__ import annotations

import logging

from cube.core import Action, ActionSchema, Observation
from arithmetic_cube.benchmark import ArithmeticBenchmark

logger = logging.getLogger(__name__)

_TASK_ACTIONS: dict[str, list[Action]] = {
    "add-3-4": [Action(name="submit_answer", arguments={"answer": 7})],
    "sub-10-3": [Action(name="submit_answer", arguments={"answer": 7})],
    "mul-6-7": [Action(name="submit_answer", arguments={"answer": 42})],
    "add-100-1": [Action(name="submit_answer", arguments={"answer": 101})],
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


def get_debug_benchmark() -> ArithmeticBenchmark:
    return ArithmeticBenchmark()


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


if __name__ == "__main__":
    import sys
    import arithmetic_cube.debug as _mod
    from cube.testing import run_debug_suite

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    results = run_debug_suite("arithmetic-cube", _mod)
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] <= 0]
    sys.exit(1 if failed else 0)
