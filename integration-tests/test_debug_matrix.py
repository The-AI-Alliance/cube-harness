"""Per-task-container cube × infra integration matrix.

Runs every (cube, infra) combination end-to-end through the cube's debug suite
(oracle-mode agent replays gold actions, benchmark-defined evaluator must return
reward == 1.0). Covers:

    cubes: terminalbench, swebench-verified, swebench-live
    infras: LocalInfraConfig, DaytonaInfraConfig, ToolkitInfraConfig

Each parametrised case is independently skippable based on prerequisites
(Docker daemon for local, ``DAYTONA_API_KEY`` for Daytona, ``eai`` CLI +
``EAI_PROFILE`` for Toolkit), so the matrix degrades gracefully in CI.

Run
---
    cd cube-harness
    uv run --group local --group daytona --group toolkit \\
        pytest integration-tests/test_debug_matrix.py -v -s

    # Subset examples
    uv run --group local pytest integration-tests -v -s -k local
    uv run --group daytona pytest integration-tests -v -s -k "swebench_verified and daytona"
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from types import ModuleType

import pytest

from cube.resource import InfraConfig
from cube_integration_tests.debug_harness import run_debug_on

import swebench_live_cube.debug as _swebench_live
import swebench_verified_cube.debug as _swebench_verified
import terminalbench_cube.debug as _terminalbench

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")


# ── cubes ────────────────────────────────────────────────────────────────────

_CUBES: list[tuple[str, ModuleType]] = [
    ("terminalbench", _terminalbench),
    ("swebench_verified", _swebench_verified),
    ("swebench_live", _swebench_live),
]


# ── infras (lazy factories: imports only when the factory is called) ─────────


def _local_infra() -> InfraConfig:
    from cube.infra_local import LocalInfraConfig

    return LocalInfraConfig()


def _daytona_infra() -> InfraConfig:
    from cube_infra_daytona import DaytonaInfraConfig

    return DaytonaInfraConfig()


def _toolkit_infra() -> InfraConfig:
    from cube_infra_toolkit import ToolkitInfraConfig

    return ToolkitInfraConfig()


# ── prerequisite checks (run at collection time to decide skip/run) ──────────


def _has_docker() -> bool:
    """Docker daemon reachable — either DOCKER_HOST is set or the default socket exists."""
    return bool(os.environ.get("DOCKER_HOST")) or os.path.exists("/var/run/docker.sock")


def _has_daytona() -> bool:
    return bool(os.environ.get("DAYTONA_API_KEY"))


def _has_toolkit() -> bool:
    """eai CLI on PATH and an EAI_PROFILE in the environment."""
    return shutil.which("eai") is not None and bool(os.environ.get("EAI_PROFILE"))


_INFRAS: list[tuple[str, Callable[[], InfraConfig], Callable[[], bool], str]] = [
    ("local", _local_infra, _has_docker, "Docker daemon not reachable"),
    ("daytona", _daytona_infra, _has_daytona, "DAYTONA_API_KEY not set"),
    ("toolkit", _toolkit_infra, _has_toolkit, "eai CLI or EAI_PROFILE not set"),
]


def _build_matrix() -> list[pytest.param]:
    """Yield (cube, infra) pytest.param entries with skips for missing prerequisites."""
    params = []
    for cube_name, cube_mod in _CUBES:
        for infra_name, factory, ready_check, skip_reason in _INFRAS:
            test_id = f"{cube_name}--{infra_name}"
            marks = []
            if not ready_check():
                marks.append(pytest.mark.skip(reason=skip_reason))
            params.append(pytest.param(cube_mod, factory, id=test_id, marks=marks))
    return params


# ── the matrix ───────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.parametrize("cube_module,infra_factory", _build_matrix())
def test_debug_suite(cube_module: ModuleType, infra_factory: Callable[[], InfraConfig]) -> None:
    results = run_debug_on(cube_module, infra_factory())
    assert len(results) >= 1
