"""
Deterministic debug agent for testing OSWorldTask end-to-end without an LLM.

Each debug task in debug_tasks.json has a hardcoded action sequence that
completes it successfully. Used to validate the CUBE task loop in CI or
local development without requiring an LLM.

Public API
----------
make_debug_agent(task_id)       → DebugAgent
get_debug_task_configs()        → list[OSWorldTaskConfig]

Usage::

    # Run all debug tasks and print a JSON report
    python -m osworld_cube.debug
"""

from __future__ import annotations

import logging
from pathlib import Path

from cube.core import Action, ActionSchema, Observation
from osworld_cube.benchmark import OSWorldBenchmark, OSWorldTaskConfig
from osworld_cube.computer import ComputerConfig

logger = logging.getLogger(__name__)

_TASKS_FILE = Path(__file__).parent / "debug_tasks.json"

# ---------------------------------------------------------------------------
# Hardcoded action sequences per task ID
# ---------------------------------------------------------------------------

_TASK_ACTIONS: dict[str, list[Action]] = {
    "simple-create-file": [
        # Open a terminal
        Action(name="hotkey", arguments={"keys": ["ctrl", "alt", "t"]}),
        # Wait for the terminal window to appear
        Action(name="wait", arguments={}),
        # Type the shell command to create the file
        Action(name="typing", arguments={"text": "echo 'Hello World' > ~/Desktop/hello.txt"}),
        # Execute the command
        Action(name="press", arguments={"key": "enter"}),
        # Wait for the command to finish
        Action(name="wait", arguments={}),
        # Signal task completion (triggers OSWorldTask.evaluate())
        Action(name="done", arguments={}),
    ],
    "simple-make-directory": [
        Action(name="hotkey", arguments={"keys": ["ctrl", "alt", "t"]}),
        Action(name="wait", arguments={}),
        Action(name="typing", arguments={"text": "mkdir ~/Desktop/my_folder"}),
        Action(name="press", arguments={"key": "enter"}),
        Action(name="wait", arguments={}),
        Action(name="done", arguments={}),
    ],
}


# ---------------------------------------------------------------------------
# DebugAgent
# ---------------------------------------------------------------------------


class DebugAgent:
    """
    Deterministic debug agent that replays a fixed action sequence for a given task.

    Interface matches the stress-test spec (stress_test_specs.md §1.2):
        agent = make_debug_agent(task_id)
        action = agent.get_action(obs)

    The __call__ shorthand is also supported for use in the standard task loop:
        action = agent(obs, action_set)

    Args:
        task_id: ID of the debug task to run. Must match a key in _TASK_ACTIONS.

    Raises:
        ValueError: If task_id has no registered action sequence.
    """

    def __init__(self, task_id: str) -> None:
        if task_id not in _TASK_ACTIONS:
            raise ValueError(f"No debug actions registered for task {task_id!r}. Known tasks: {list(_TASK_ACTIONS)}")
        self._task_id = task_id
        self._step = 0
        self._actions = list(_TASK_ACTIONS[task_id])
        logger.debug(
            "[DebugAgent] Initialised for task=%r with %d actions",
            task_id,
            len(self._actions),
        )

    def get_action(self, obs: Observation) -> Action:
        """Return the next predetermined action (stress-test spec interface)."""
        if self._step >= len(self._actions):
            raise StopIteration(f"[DebugAgent] task={self._task_id!r}: all {len(self._actions)} actions exhausted")
        action = self._actions[self._step]
        logger.info(
            "[DebugAgent] task=%r  step=%d/%d  action=%s  args=%s",
            self._task_id,
            self._step + 1,
            len(self._actions),
            action.name,
            action.arguments or "",
        )
        self._step += 1
        return action

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        """Callable shorthand — delegates to get_action() for task-loop compatibility."""
        return self.get_action(obs)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def make_debug_agent(task_id: str) -> DebugAgent:
    """Return a fresh DebugAgent for the given task_id."""
    return DebugAgent(task_id)


def get_debug_task_configs() -> list[OSWorldTaskConfig]:
    """
    Load debug task definitions from debug_tasks.json and return as OSWorldTaskConfig list.

    Creates an OSWorldBenchmark backed by debug_tasks.json and calls setup() so
    that OSWorldBenchmark.task_metadata is populated.  make() on each returned
    config will find the metadata via the class-level registry without needing it
    embedded in the config.
    """
    bench = OSWorldBenchmark(tasks_file=str(_TASKS_FILE), default_tool_config=ComputerConfig())
    bench.setup()
    configs = [cfg for cfg in bench.get_task_configs() if isinstance(cfg, OSWorldTaskConfig)]
    logger.debug("[get_debug_task_configs] Loaded %d configs from %s", len(configs), _TASKS_FILE)
    return configs


# ---------------------------------------------------------------------------
# __main__ — run all debug tasks, print JSON report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import osworld_cube.debug as _mod
    from cube.testing import run_debug_suite

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    results = run_debug_suite("osworld-cube", _mod)

    # Exit non-zero if any episode failed or got reward 0
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] <= 0]
    sys.exit(1 if failed else 0)
