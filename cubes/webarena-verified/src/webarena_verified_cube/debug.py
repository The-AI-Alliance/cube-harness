"""Smoke-test script for webarena-verified-cube — validates infrastructure without an LLM.

Verifies that tasks load, the real Playwright tool initialises, the task resets correctly,
and the DebugAgent can submit the expected action from the task definition.

The evaluation may return reward=0 because task.evaluate() passes a
FinalAgentResponse object directly to wav.evaluate_task(), which expects str/dict.
Only errors (Python exceptions) are treated as failures.

The debug suite requires a running shopping-admin server. It is started automatically
when this script is invoked directly. A local Docker daemon must be available.

Public API (cube.testing protocol)
-----------------------------------
get_debug_benchmark()              -> Benchmark
make_debug_agent(task_id: str)     -> DebugAgent

Usage:
    uv run python -m webarena_verified_cube.debug
"""

from __future__ import annotations

import logging
import subprocess
import sys

from cube.benchmark import Benchmark
from cube.core import Action, ActionSchema, Observation
from cube.testing import run_debug_suite
from webarena_verified.api.webarena_verified import WebArenaVerified
from webarena_verified.types.agent_response import FinalAgentResponse
from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
from webarena_verified.types.task import WebArenaSite

from cube.tool import ToolboxConfig
from webarena_verified_cube.benchmark import WebArenaVerifiedBenchmark
from webarena_verified_cube.tool import HarPlaywrightConfig, SubmitResponseConfig

logger = logging.getLogger(__name__)

# Task IDs 0 and 1 are both shopping_admin RETRIEVE tasks with only
# AgentResponseEvaluator — no live servers or network events required.
_DEBUG_TASK_IDS = ["0", "1"]

# Dummy environment config: render_url() raises when environments is None,
# even with strict=False. A placeholder URL lets reset() proceed without error.
_DEBUG_WAV_CONFIG = WebArenaVerifiedConfig(
    environments={
        WebArenaSite.SHOPPING_ADMIN: EnvironmentConfig(urls=["http://localhost:7780"]),
    }
)


class DebugAgent:
    """Agent that submits the expected action from the task definition."""

    def __init__(self, expected_response: FinalAgentResponse) -> None:
        self._expected_response = expected_response

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        resp = self._expected_response
        args: dict = {
            "task_type": str(resp.task_type),
            "status": str(resp.status),
            "error_details": resp.error_details,
            "retrieved_data": resp.retrieved_data,
        }
        return Action(name="submit_response", arguments=args)


def make_debug_agent(task_id: str) -> DebugAgent:
    wav = WebArenaVerified()
    wav_task = wav.get_task(int(task_id))
    return DebugAgent(expected_response=wav_task.expected_agent_response)


def get_debug_benchmark() -> Benchmark:
    return WebArenaVerifiedBenchmark(
        wav_config=_DEBUG_WAV_CONFIG,
        task_ids_filter=[int(tid) for tid in _DEBUG_TASK_IDS],
        default_tool_config=ToolboxConfig(tool_configs=[HarPlaywrightConfig(), SubmitResponseConfig()]),
    )


def _start_debug_server() -> None:
    """Start the shopping-admin server required by the debug suite.

    Requires a local Docker daemon. Prints a clear error and exits if the
    command fails (Docker not running, webarena-verified not installed, etc.).
    """
    cmd = ["webarena-verified", "env", "start", "--site"]
    sites = [env.value for env in _DEBUG_WAV_CONFIG.environments]  # type: ignore[attr-defined]
    cmd += sites
    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print(
            f"\nERROR: 'webarena-verified' command not found.\n"
            f"Make sure the package is installed, then run:\n"
            f"  {' '.join(cmd)}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if result.returncode != 0:
        print(
            f"\nERROR: Failed to start the {', '.join(sites)} server(s) (exit code {result.returncode}).\n"
            f"Make sure Docker is running and try manually:\n"
            f"  {' '.join(cmd)}\n",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    import webarena_verified_cube.debug as _this_module

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    _start_debug_server()
    results = run_debug_suite("webarena-verified-cube", _this_module)

    failed = [r for r in results if r["error"] or r["reward"] != 1.0]
    sys.exit(1 if failed else 0)
