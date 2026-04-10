"""
Deterministic debug agent for testing OSWorldTask end-to-end without an LLM.

Each debug task in debug_task_metadata.json has a hardcoded action sequence that
completes it successfully. Used to validate the CUBE task loop in CI or local
development without requiring an LLM.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import ClassVar

import cube
from cube import LocalInfraConfig
from cube.core import Action, ActionSchema, Observation
from cube.container import ContainerBackend
from cube.testing import run_debug_suite
from cube.task import TaskConfig
from cube.resource import InfraConfig
from cube.benchmark import Benchmark

from osworld_cube.benchmark import OSWorldBenchmark, OSWorldTaskConfig
from osworld_cube.computer import ComputerConfig
from osworld_cube.infra_loader import load_runtime_infra_from_config_file
from osworld_cube.task import OSWorldTask, OSWorldTaskMetadata

logger = logging.getLogger(__name__)

_DEBUG_TASK_METADATA_JSON = Path(__file__).with_name("debug_task_metadata.json")
_DEBUG_TASK_METADATA = Benchmark.task_metadata_from_json(_DEBUG_TASK_METADATA_JSON)


class DebugOSWorldTaskConfig(OSWorldTaskConfig):
    """Task config for the embedded debug benchmark."""

    def make(
        self,
        runtime_context: dict | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> OSWorldTask:
        if self.tool_config is None:
            raise ValueError(f"DebugOSWorldTaskConfig for task '{self.task_id}' has no tool_config.")
        return OSWorldTask(
            metadata=self.metadata,
            tool_config=self.tool_config,
            infra=self.infra,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )


class DebugOSWorldBenchmark(OSWorldBenchmark):
    """OSWorld benchmark scoped to the two hardcoded debug tasks."""

    benchmark_metadata: ClassVar = OSWorldBenchmark.benchmark_metadata.model_copy(
        update={"name": "osworld-debug-cube", "num_tasks": len(_DEBUG_TASK_METADATA), "named_subsets": {}}
    )
    task_metadata: ClassVar[dict[str, OSWorldTaskMetadata]] = _DEBUG_TASK_METADATA
    task_config_class: ClassVar[type[TaskConfig]] = DebugOSWorldTaskConfig

    @classmethod
    def cache_dir(cls) -> Path:
        return cube.get_cache_dir("osworld-debug-cube")

    @classmethod
    def install(cls) -> None:
        logger.info("DebugOSWorldBenchmark.install() - nothing to do")

    @classmethod
    def uninstall(cls) -> None:
        logger.info("DebugOSWorldBenchmark.uninstall() - nothing to do")


_TASK_ACTIONS: dict[str, list[Action]] = {
    "simple-create-file": [
        Action(name="hotkey", arguments={"keys": ["ctrl", "alt", "t"]}),
        Action(name="wait", arguments={}),
        Action(name="typing", arguments={"text": "echo 'Hello World' > ~/Desktop/hello.txt"}),
        Action(name="press", arguments={"key": "enter"}),
        Action(name="wait", arguments={}),
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


class DebugAgent:
    """Deterministic debug agent that replays a fixed action sequence for a task."""

    def __init__(self, task_id: str) -> None:
        if task_id not in _TASK_ACTIONS:
            raise ValueError(f"No debug actions registered for task {task_id!r}. Known tasks: {list(_TASK_ACTIONS)}")
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


def _get_default_infra() -> InfraConfig:
    return load_runtime_infra_from_config_file() or LocalInfraConfig()


def get_debug_benchmark(infra: InfraConfig | None = None) -> DebugOSWorldBenchmark:
    """Return a benchmark scoped to the embedded debug tasks."""
    resolved_infra = infra or _get_default_infra()
    return DebugOSWorldBenchmark(default_tool_config=ComputerConfig(), infra=resolved_infra)


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


if __name__ == "__main__":
    import osworld_cube.debug as _mod

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    results = run_debug_suite("osworld-cube", _mod)
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] <= 0]
    sys.exit(1 if failed else 0)
