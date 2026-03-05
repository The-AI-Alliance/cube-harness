"""Smoke-test script for miniwob-cube — validates infrastructure without an LLM.

Verifies that the MiniWob HTTP server starts, BrowserGym connects, the task
page loads and JS initialises, and page observations are returned without errors.
The DebugAgent immediately stops (reward=0 is expected), so only errors are
treated as failures.

Public API (cube.testing protocol)
-----------------------------------
get_debug_task_configs()           -> list[MiniWobTaskConfig]
make_debug_agent(task_id: str)     -> DebugAgent

Usage:
    uv run python -m miniwob_cube.debug
"""

from __future__ import annotations

import logging
import sys

from cube.core import Action, ActionSchema, Observation
from cube.container import Container
from cube.tool import ToolConfig, AbstractTool
from cube.testing import run_debug_suite

from miniwob_cube.benchmark import MiniWobBenchmark
from miniwob_cube.task import MiniWobTaskConfig


logger = logging.getLogger(__name__)

# A small set of representative tasks that cover the JS setup / observation path.
_DEBUG_TASK_IDS = ["click-button", "click-checkboxes"]


class MockToolConfig(ToolConfig):
    def make(self, container: Container | None = None) -> AbstractTool:
        return MockTool()


class MockTool(AbstractTool):
    def execute_action(self, action: Action) -> Observation:
        return Observation(contents=[])

    def goto(self, url: str) -> None:
        _ = url

    def evaluate_js(self, js: str) -> str:
        _ = js
        return "0.0,0.0,reason,true,1,true"

    def page_obs(self) -> Observation:
        return Observation(contents=[])

    @property
    def action_set(self) -> list[ActionSchema]:
        return []


class DebugAgent:
    """Minimal agent that stops immediately — validates infrastructure, not task solving."""

    def __init__(self, task_id: str) -> None:
        self._task_id = task_id

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        return Action(name="final_step", arguments={})


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


def get_debug_task_configs(base_url: str = "http://localhost:8000/miniwob") -> list[MiniWobTaskConfig]:
    return [
        MiniWobTaskConfig(
            task_id=tid,
            task_metadata=MiniWobBenchmark.task_metadata[tid],
            base_url=base_url,
            tool_config=MockToolConfig(),
        )
        for tid in _DEBUG_TASK_IDS
    ]


if __name__ == "__main__":
    import miniwob_cube.debug as _this_module

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    benchmark = MiniWobBenchmark()
    try:
        benchmark.setup()
        results = run_debug_suite("miniwob-cube", _this_module)

        # Smoke test: fail only on errors, not on reward (agent stops immediately).
        failed = [r for r in results if r["error"]]
    finally:
        benchmark.close()

    sys.exit(1 if failed else 0)
