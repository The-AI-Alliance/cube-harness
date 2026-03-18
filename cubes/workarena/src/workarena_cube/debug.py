"""Smoke-test script for workarena-cube — validates infrastructure without an LLM.

Verifies that WorkArena task configs can be enumerated, tasks can be instantiated,
and the tool + WorkArena episode lifecycle run without errors.

Requires ServiceNow credentials (SNOW_INSTANCE_URL, SNOW_INSTANCE_UNAME,
SNOW_INSTANCE_PWD) or HUGGING_FACE_HUB_TOKEN for the hosted instance pool.

Public API (cube.testing protocol)
-----------------------------------
get_debug_benchmark()              -> WorkArenaBenchmark
make_debug_agent(task_id: str)     -> CheatAgent

Usage:
    uv run cube test workarena-cube
"""

from __future__ import annotations

import logging
import sys
from typing import ClassVar, Generator

from cube.benchmark import BenchmarkMetadata
from cube.core import Action, ActionSchema, Observation
from cube.task import TaskConfig, TaskMetadata
from cube.testing import run_debug_suite

from workarena_cube.benchmark import WorkArenaBenchmark
from workarena_cube.task import WorkArenaCheatToolConfig, WorkArenaTaskConfig

logger = logging.getLogger(__name__)

_DEBUG_N_TASKS = 2


class CheatAgent:
    """Agent that calls WorkArena's cheat action to solve the task, then stops."""

    def __init__(self, task_id: str) -> None:
        self._task_id = task_id
        self._cheated: bool = False

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        if not self._cheated:
            self._cheated = True
            return Action(name="workarena_cheat", arguments={})
        return Action(name="final_step", arguments={})


class DebugBenchmark(WorkArenaBenchmark):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = WorkArenaBenchmark.benchmark_metadata
    task_metadata: ClassVar[dict[str, TaskMetadata]] = WorkArenaBenchmark.task_metadata
    task_config_class: ClassVar[type[TaskConfig]] = WorkArenaTaskConfig

    def get_task_configs(self) -> Generator[WorkArenaTaskConfig, None, None]:
        yield from list(super().get_task_configs())[:_DEBUG_N_TASKS]


def make_debug_agent(task_id: str) -> CheatAgent:
    return CheatAgent(task_id)


def get_debug_benchmark() -> WorkArenaBenchmark:
    return DebugBenchmark(
        n_seeds_l1=1,
        default_tool_config=WorkArenaCheatToolConfig(),
    )


if __name__ == "__main__":
    import workarena_cube.debug as _this_module

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    results = run_debug_suite("workarena-cube", _this_module)
    failed = [r for r in results if r["error"]]

    sys.exit(1 if failed else 0)
