"""Smoke-test script for workarena-cube — validates infrastructure without an LLM.

Verifies that WorkArena task configs can be enumerated, tasks can be instantiated,
and the tool + WorkArena episode lifecycle run without errors.

Requires ServiceNow credentials (SNOW_INSTANCE_URL, SNOW_INSTANCE_UNAME,
SNOW_INSTANCE_PWD) or HUGGING_FACE_HUB_TOKEN for the hosted instance pool.

Public API (cube.testing protocol)
-----------------------------------
get_debug_task_configs()           -> list[WorkArenaTaskConfig]
make_debug_agent(task_id: str)     -> DebugAgent

Usage:
    uv run python -m workarena_cube.debug
"""

from __future__ import annotations

import logging
import sys

from browsergym.workarena import get_all_tasks_agents
from cube.core import Action, ActionSchema, Observation
from cube.task import TaskMetadata
from cube.testing import run_debug_suite
from cube_browser_tool import PlaywrightConfig

from workarena_cube.benchmark import WorkArenaBenchmark
from workarena_cube.task import WorkArenaTaskConfig

logger = logging.getLogger(__name__)

_DEBUG_N_TASKS = 2


class DebugAgent:
    """Minimal agent that stops immediately — validates infrastructure, not task solving."""

    def __init__(self, task_id: str) -> None:
        self._task_id = task_id

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        return Action(name="final_step", arguments={})


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


def get_debug_task_configs() -> list[WorkArenaTaskConfig]:
    task_tuples = get_all_tasks_agents(filter="l1", meta_seed=42, n_seed_l1=1, is_agent_curriculum=False)[
        :_DEBUG_N_TASKS
    ]
    return [
        WorkArenaTaskConfig(
            task_id=task_class.get_task_id(),
            seed=seed,
            tool_config=PlaywrightConfig(),
            task_metadata=TaskMetadata(
                id=task_class.get_task_id(),
                extra_info={
                    "task_class_path": f"{task_class.__module__}.{task_class.__qualname__}",
                    "level": "l1",
                },
            ),
        )
        for task_class, seed in task_tuples
    ]


if __name__ == "__main__":
    import workarena_cube.debug as _this_module

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    benchmark = WorkArenaBenchmark()
    try:
        benchmark.setup()
        results = run_debug_suite("workarena-cube", _this_module)
        failed = [r for r in results if r["error"]]
    finally:
        benchmark.close()

    sys.exit(1 if failed else 0)
