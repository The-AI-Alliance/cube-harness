"""Deterministic debug agent for testing terminalbench-cube end-to-end without an LLM.

Public API
----------
get_debug_benchmark()         → TerminalBenchBenchmark
make_debug_agent(task_id)     → DebugAgent
"""

from __future__ import annotations

import logging

from cube.benchmark import Benchmark
from cube.core import Action, ActionSchema, Observation
from cube.infra_local import LocalInfraConfig
from cube.resource import InfraConfig
from terminalbench_cube.benchmark import TerminalBenchBenchmark

logger = logging.getLogger(__name__)

# Each debug task runs in oracle_mode: the ground-truth solution is uploaded
# to /solution in the container during reset(). The debug agent applies it
# and calls final_step, which triggers evaluate() → pytest → reward == 1.0.
_FINAL = Action(name="final_step", arguments={})

_TASK_ACTIONS: dict[str, list[Action]] = {
    # Relative `personal-site` works whether working_dir is /app (default) or
    # /tmp/app (after _maybe_relocate_app on non-root backends).
    # /tmp/solution is where TerminalBenchTask.reset() uploads the solution
    # for any backend — /tmp is universally writable.
    "fix-git": [
        Action(
            name="bash",
            arguments={"command": "cd personal-site && bash /tmp/solution/solve.sh 2>&1", "timeout": 600},
        ),
        _FINAL,
    ],
    "overfull-hbox": [
        # Prepend the apt-extracted python3 to PATH so solve.sh's `python3` call
        # works on minimal images (e.g. bare LaTeX) that ship without python3.
        # On images that already have python3 in PATH the prepended directory
        # either doesn't exist (harmless) or provides a compatible python3.12.
        Action(
            name="bash",
            arguments={
                "command": (
                    "export PATH=/tmp/python3_pkg/usr/bin:$PATH && "
                    "export LD_LIBRARY_PATH=/tmp/python3_pkg/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH && "
                    "bash /tmp/solution/solve.sh 2>&1"
                ),
                "timeout": 600,
            },
        ),
        _FINAL,
    ],
}


class DebugAgent:
    """Deterministic agent that replays a fixed action sequence."""

    def __init__(self, task_id: str) -> None:
        if task_id not in _TASK_ACTIONS:
            raise ValueError(f"No debug actions for {task_id!r}. Known: {list(_TASK_ACTIONS)}")
        self._task_id = task_id
        self._step = 0
        self._actions = list(_TASK_ACTIONS[task_id])

    def get_action(self, obs: Observation) -> Action:
        if self._step >= len(self._actions):
            raise StopIteration(f"All actions exhausted for task {self._task_id!r}")
        action = self._actions[self._step]
        self._step += 1
        return action

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        return self.get_action(obs)


def get_debug_benchmark(infra: InfraConfig | None = None) -> "Benchmark":
    """Return a TerminalBenchBenchmark scoped to the debug tasks.

    Args:
        infra: InfraConfig that provisions and launches per-task containers. Defaults
               to ``LocalInfraConfig()`` — suitable for local development / CI with
               a running Docker daemon. Integration tests override this to target
               Toolkit, Daytona, etc.
    """
    bench = TerminalBenchBenchmark(
        infra=infra or LocalInfraConfig(),
        oracle_mode=True,
    )
    bench.install()
    bench.setup()
    return bench.subset_from_list(list(_TASK_ACTIONS), benchmark_name_suffix="debug")


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


if __name__ == "__main__":
    import sys

    import terminalbench_cube.debug as _this_module
    from cube.testing import run_debug_suite

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    results = run_debug_suite("terminalbench-cube", _this_module)
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] < 1.0]
    sys.exit(1 if failed else 0)
