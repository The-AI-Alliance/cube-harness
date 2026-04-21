"""Integration test: terminalbench-cube end-to-end on LocalInfraConfig.

What this exercises
-------------------
For every entry in ``terminalbench_cube.debug._TASK_ACTIONS``:

1. ``LocalInfraConfig.provision`` — docker pull the per-task image (idempotent).
2. ``LocalInfraConfig.launch``    — ``docker run -d sleep infinity`` as the per-task container.
3. ``TerminalBenchTask.reset``    — upload the task archive + solution.
4. ``DebugAgent``                 — replay the canned solve.sh actions.
5. ``TerminalBenchTask.evaluate`` — run the pytest harness, expect reward == 1.0.
6. ``ResourceHandle.close``       — ``docker stop`` + ``docker rm`` the container.

Prerequisites
-------------
- Local Docker daemon running (``docker info`` green).
- ~2 GB free disk for the pulled images.

Run
---
    cd cube-harness
    uv run pytest integration-tests/test_terminalbench_local.py -v -s
"""

from __future__ import annotations

import logging

import pytest

from cube.infra_local import LocalInfraConfig
from cube_integration_tests.debug_harness import run_debug_on

import terminalbench_cube.debug as _terminalbench_debug

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")


@pytest.mark.integration
def test_terminalbench_local() -> None:
    results = run_debug_on(_terminalbench_debug, LocalInfraConfig())
    assert len(results) >= 1
