"""Reusable debug-suite harness for per-task-container cubes.

Expected cube module shape
--------------------------
- ``get_debug_benchmark() -> BenchmarkConfig``
- ``make_debug_agent(task_id) -> agent callable``
- ``_TASK_ACTIONS: dict[str, list[Action]]``

The harness owns ``config.install()`` and ``config.make(infra)`` — the cube's
``get_debug_benchmark`` is a pure factory (no infra argument) since the rc7
``BenchmarkConfig`` migration.
"""

from __future__ import annotations

import logging
import time
from types import ModuleType
from typing import Any

from cube.resource import InfraConfig
from cube.testing import run_debug_episode

logger = logging.getLogger(__name__)

# Per-task wall-clock timeout in seconds.  Raised as a pytest failure (not a
# hang) so the matrix keeps running.  Local Docker tasks take ~60-90s;
# cloud tasks (Toolkit/Daytona) take up to ~10 min including job startup,
# exec relay bootstrap, and any nonroot python3/uv pre-install.
_TASK_TIMEOUT_SECONDS = 900


def run_debug_task(
    cube_debug_module: ModuleType,
    task_id: str,
    infra: InfraConfig,
    *,
    max_steps: int = 20,
    timeout: int = _TASK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run a single debug task end-to-end and return the episode result dict.

    The result dict has keys: task_id, reward, done, error (mirrors
    run_debug_suite schema).  Raises TimeoutError if the task exceeds
    ``timeout`` seconds — this surfaces as a pytest FAILED rather than a hang.
    """
    import threading

    config = cube_debug_module.get_debug_benchmark()
    config.install()
    benchmark = config.make(infra)

    result: dict[str, Any] = {}
    exc_holder: list[BaseException] = []

    def _run() -> None:
        try:
            task_configs = [tc for tc in config.get_task_configs() if tc.task_id == task_id]
            if not task_configs:
                raise ValueError(f"Task {task_id!r} not found in benchmark debug subset")
            tc = task_configs[0]
            task = benchmark.spawn(tc)
            try:
                logger.info("START  task=%r  infra=%s", task_id, infra.fingerprint())
                t0 = time.monotonic()
                result.update(
                    run_debug_episode(
                        task,
                        cube_debug_module.make_debug_agent(task_id),
                        max_steps=max_steps,
                    )
                )
                elapsed = time.monotonic() - t0
                logger.info(
                    "FINISH task=%r  reward=%s  done=%s  elapsed=%.1fs",
                    task_id,
                    result.get("reward"),
                    result.get("done"),
                    elapsed,
                )
            finally:
                task.close()
        except BaseException as e:
            exc_holder.append(e)

    thread = threading.Thread(target=_run, daemon=True)
    try:
        thread.start()
        thread.join(timeout=timeout)
    finally:
        benchmark.close()

    if thread.is_alive():
        raise TimeoutError(
            f"Task {task_id!r} on {infra.fingerprint()} exceeded {timeout}s timeout — "
            "still running in background thread (daemon, will be killed on process exit)"
        )
    if exc_holder:
        raise exc_holder[0]
    return result
