"""Smoke-test script for webarena-verified-cube — validates infrastructure without an LLM.

Verifies that tasks load, the real Playwright tool initialises, the task resets correctly,
and the DebugAgent can submit the expected action from the task definition.

The evaluation may return reward=0 because task.evaluate() passes a
FinalAgentResponse object directly to wav.evaluate_task(), which expects str/dict.
Only errors (Python exceptions) are treated as failures.

Debug tasks are restricted to AgentResponseEvaluator-only tasks on a single site
so no live servers are required.

Public API (cube.testing protocol)
-----------------------------------
get_debug_task_configs()           -> list[WebArenaVerifiedTaskConfig]
make_debug_agent(task_id: str)     -> DebugAgent

Usage:
    uv run python -m webarena_verified_cube.debug
"""

from __future__ import annotations

import logging
import sys

from cube.benchmark import Benchmark
from cube.core import Action, ActionSchema, Observation
from cube.testing import run_debug_suite
from webarena_verified.api.webarena_verified import WebArenaVerified
from webarena_verified.types.agent_response import FinalAgentResponse
from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
from webarena_verified.types.task import WebArenaSite

from webarena_verified_cube.benchmark import WebArenaVerifiedBenchmark
from webarena_verified_cube.tool import WebArenaToolConfig

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
        default_tool_config=WebArenaToolConfig(headless=False),
    )


if __name__ == "__main__":
    import webarena_verified_cube.debug as _this_module

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    results = run_debug_suite("webarena-verified-cube", _this_module)

    # Smoke test: fail only on errors, not on reward.
    # Reward may be 0 because evaluate() passes FinalAgentResponse to wav.evaluate_task()
    # which expects str/dict — no live server is needed to verify infrastructure.
    failed = [r for r in results if r["error"]]
    sys.exit(1 if failed else 0)
