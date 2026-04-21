"""Reusable debug-suite harness for per-task-container cubes.

Parameterises the ``(benchmark class, InfraConfig)`` pair so one test per
(cube, infra) combination is a ~5-line wiring step.

Expected cube module shape
---------------------------
The cube's ``debug`` module must expose:

- ``get_debug_benchmark(infra: InfraConfig) -> Benchmark`` — returns a ready-to-run
  benchmark already scoped to the debug subset and already ``.setup()``-ed.
- ``make_debug_agent(task_id: str) -> Callable[[Observation, list[ActionSchema]], Action]``
- ``_TASK_ACTIONS: dict[str, list[Action]]`` — drives which tasks run.

Run one task through the full ``reset → step*→ evaluate`` loop, assert reward==1.0.
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Any

from cube.resource import InfraConfig
from cube.testing import run_debug_episode

logger = logging.getLogger(__name__)


def run_debug_on(cube_debug_module: ModuleType, infra: InfraConfig, *, max_steps: int = 20) -> list[dict[str, Any]]:
    """Run every debug task in ``cube_debug_module`` against the given infra.

    Asserts per-task: no error, ``done=True``, ``reward == 1.0``. Returns the
    list of per-episode reports (mirrors ``run_debug_suite`` schema).
    """
    bench = cube_debug_module.get_debug_benchmark(infra=infra)
    # subset_from_list resets PrivateAttrs per spec — re-run setup() on the subset
    # so _runtime_context is repopulated (including the "infra" key).
    bench.setup()
    try:
        results: list[dict[str, Any]] = []
        for tc in bench.get_task_configs():
            logger.info("Running debug task %r on %s", tc.task_id, infra.fingerprint())
            task = tc.make(runtime_context=bench._runtime_context)
            try:
                result = run_debug_episode(
                    task,
                    cube_debug_module.make_debug_agent(tc.task_id),
                    max_steps=max_steps,
                )
            finally:
                task.close()
            results.append(result)
            assert not result["error"], f"Task {tc.task_id!r} errored: {result['error']}"
            assert result["done"], f"Task {tc.task_id!r} did not complete (reward={result['reward']})"
            assert result["reward"] == 1.0, f"Task {tc.task_id!r} reward={result['reward']} (expected 1.0)"
        return results
    finally:
        bench.close()
