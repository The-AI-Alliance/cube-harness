"""Integration test: swebench-verified-cube end-to-end on DaytonaInfraConfig.

Exercises the full resource lifecycle for a per-task Docker container on a
cloud-sandbox backend (no local Docker daemon required):

1. ``DaytonaInfraConfig.provision`` — idempotent record for the debug image.
2. ``DaytonaInfraConfig.launch``    — create a Daytona sandbox with the SWE-bench image.
3. ``SWEBenchVerifiedTask.reset``   — write gold patch for oracle_mode.
4. ``DebugAgent``                   — replay apply-patch + final_step actions.
5. ``SWEBenchVerifiedTask.evaluate`` — apply the test patch, run unit tests, expect reward == 1.0.
6. ``DaytonaResourceHandle.close``  — delete the Daytona sandbox.

Prerequisites
-------------
- ``DAYTONA_API_KEY`` (and optionally ``DAYTONA_TARGET``) in the environment.
- Network access to the configured Daytona API URL.

SWE-bench images are ~1-3 GB each; Daytona pulls them on-demand.  Expect the
first run to be slow (~5-10 minutes per task) while Daytona caches the image.

Run
---
    cd cube-harness
    uv run --group daytona pytest integration-tests/test_swebench_verified_daytona.py -v -s
"""

from __future__ import annotations

import logging
import os

import pytest

from cube_infra_daytona import DaytonaInfraConfig
from cube_integration_tests.debug_harness import run_debug_on

import swebench_verified_cube.debug as _swebench_debug

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("DAYTONA_API_KEY"), reason="DAYTONA_API_KEY not set")
def test_swebench_verified_daytona() -> None:
    results = run_debug_on(_swebench_debug, DaytonaInfraConfig())
    assert len(results) >= 1
