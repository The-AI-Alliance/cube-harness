"""OSWorld debug task and scripted agent for use with cube.testing.

Implements the module protocol expected by ``cube.testing.assert_debug_tasks_reward_one``:

    get_debug_task_configs() -> list[OSWorldDebugTaskConfig]
    make_debug_agent(task_id: str) -> Callable[[Observation, list[ActionSchema]], Action]

The single debug task is ``"simple-create-file"``: create ``~/Desktop/hello.txt``
containing ``"Hello World"``.  The scripted agent opens a terminal with
``Ctrl+Alt+T``, types the ``echo`` command, and calls ``done()``.

Usage (integration test — requires Docker + KVM + OSWorld qcow2)::

    from cube.testing import assert_debug_tasks_reward_one
    import osworld_cube.debug_agent as mod
    assert_debug_tasks_reward_one(mod, max_steps=20)

Or as a standalone script::

    uv run python -c "
    from cube.testing import run_debug_suite
    import osworld_cube.debug_agent as mod
    run_debug_suite('osworld-cube', mod)
    "
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional

from pydantic import Field

from cube.core import Action, ActionSchema, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import ToolConfig
from osworld_cube.vm_utils import OSWORLD_DOCKER_IMAGE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Debug task data
# ---------------------------------------------------------------------------

_SIMPLE_CREATE_FILE: dict = {
    "id": "simple-create-file",
    "instruction": "Create a file called 'hello.txt' on the Desktop with the content 'Hello World'.",
    "snapshot": "init_state",
    "config": [],
    "evaluator": {
        "func": "check_include_exclude",
        "result": {
            "type": "vm_command_line",
            "command": "cat ~/Desktop/hello.txt",
        },
        "expected": {
            "type": "rule",
            "rules": {"include": ["Hello World"], "exclude": []},
        },
    },
}


# ---------------------------------------------------------------------------
# cube.tool.ToolConfig bridge
# ---------------------------------------------------------------------------


class OSWorldCubeToolConfig(ToolConfig):
    """``cube.tool.ToolConfig`` adapter for OSWorld.

    Wraps ``OSWorldComputerConfig`` so it can be used with ``cube.task.Task``
    (which expects a ``cube.tool.ToolConfig``).
    """

    vm_image_path: Optional[str] = None
    docker_image: str = OSWORLD_DOCKER_IMAGE
    headless: bool = True
    require_a11y_tree: bool = False
    observe_after_action: bool = True

    def make(self, container: Any = None) -> Any:  # returns Computer
        from osworld_cube.config import OSWorldComputerConfig

        return OSWorldComputerConfig(
            vm_image_path=self.vm_image_path,
            docker_image=self.docker_image,
            headless=self.headless,
            require_a11y_tree=self.require_a11y_tree,
            observe_after_action=self.observe_after_action,
        ).make()


# ---------------------------------------------------------------------------
# cube.task.Task subclass
# ---------------------------------------------------------------------------


class OSWorldCubeTask(Task):
    """``cube.task.Task`` subclass for OSWorld desktop tasks.

    Bridges ``cube_computer_tool.Computer`` (the tool) with the OSWorld task
    lifecycle (setup, evaluate) so that ``cube.testing.run_debug_episode``
    can drive the episode without knowledge of OSWorld internals.
    """

    osworld_task_data: dict = Field(default_factory=dict)

    def reset(self) -> tuple[Observation, dict]:
        """Restore VM snapshot, run setup commands, return initial observation."""
        obs = self.tool.setup_task(self.osworld_task_data)  # type: ignore[attr-defined]
        instruction = self.osworld_task_data.get("instruction", "")
        goal_obs = Observation.from_text(f"Task: {instruction}")
        return goal_obs + obs, {"task_id": self.id}

    def evaluate(self, obs: Observation) -> tuple[float, dict]:
        """Run the task evaluator and return ``(reward, info)``."""
        evaluator = self.osworld_task_data.get("evaluator", {})
        result_spec = evaluator.get("result", {})
        expected = evaluator.get("expected", {})

        if result_spec.get("type") == "vm_command_line":
            command = result_spec["command"]
            output = self.tool.run_shell_command(command)  # type: ignore[attr-defined]
            logger.info("Evaluator command %r → output: %r", command, output[:200])

            rules = expected.get("rules", {})
            reward = 1.0
            for s in rules.get("include", []):
                if s not in output:
                    reward = 0.0
                    break
            for s in rules.get("exclude", []):
                if s in output:
                    reward = 0.0
                    break
            return reward, {"done": reward > 0.0, "output": output[:200]}

        logger.warning("Unsupported evaluator type: %r", result_spec.get("type"))
        return 0.0, {"done": False, "error": "unsupported evaluator"}

    def finished(self, obs: Observation) -> bool:
        return getattr(self.tool, "_is_done", False)


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------


class OSWorldDebugTaskConfig(TaskConfig):
    """Serialisable config for a single OSWorld debug task."""

    osworld_task_data: dict = Field(default_factory=dict)
    tool_config: OSWorldCubeToolConfig = Field(
        default_factory=OSWorldCubeToolConfig
    )

    def make(self, runtime_context: Any = None, container_backend: Any = None) -> OSWorldCubeTask:
        return OSWorldCubeTask(
            metadata=TaskMetadata(
                id=self.task_id,
                abstract_description=self.osworld_task_data.get("instruction", ""),
                recommended_max_steps=20,
            ),
            tool_config=self.tool_config,
            osworld_task_data=self.osworld_task_data,
        )


# ---------------------------------------------------------------------------
# cube.testing module protocol
# ---------------------------------------------------------------------------


def get_debug_task_configs() -> list[OSWorldDebugTaskConfig]:
    """Return one config per OSWorld debug task."""
    return [
        OSWorldDebugTaskConfig(
            task_id="simple-create-file",
            osworld_task_data=_SIMPLE_CREATE_FILE,
        ),
    ]


def make_debug_agent(task_id: str) -> Callable[[Observation, list[ActionSchema]], Action]:
    """Return a deterministic scripted agent for the given debug task.

    The agent "knows where to click": it executes a hardcoded action sequence
    without looking at the observation.

    ``simple-create-file`` sequence:
        1. ``Ctrl+Alt+T``  — open terminal
        2–4. ``wait`` × 3  — let terminal window appear (~1 s per wait step)
        5. ``typing``       — type the echo command
        6. ``press enter``  — execute it
        7. ``done``         — signal success
    """
    _SCRIPTS: dict[str, list[Action]] = {
        "simple-create-file": [
            Action(name="hotkey", arguments={"keys": ["ctrl", "alt", "t"]}),
            Action(name="wait", arguments={}),
            Action(name="wait", arguments={}),
            Action(name="wait", arguments={}),
            Action(
                name="typing",
                arguments={"text": "echo 'Hello World' > ~/Desktop/hello.txt"},
            ),
            Action(name="press", arguments={"key": "enter"}),
            Action(name="done", arguments={}),
        ],
    }

    if task_id not in _SCRIPTS:
        raise ValueError(f"No debug script for task_id={task_id!r}")

    script = list(_SCRIPTS[task_id])
    idx = [0]

    def agent(obs: Observation, action_set: list[ActionSchema]) -> Action:
        if idx[0] < len(script):
            action = script[idx[0]]
            idx[0] += 1
            return action
        return Action(name="done", arguments={})

    return agent
