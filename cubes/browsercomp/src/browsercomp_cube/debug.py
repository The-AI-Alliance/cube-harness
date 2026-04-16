"""Deterministic debug agent for testing BrowseCompBenchmark end-to-end without an LLM.

Public API
----------
get_debug_benchmark()         → DebugBrowseCompBenchmark
make_debug_agent(task_id)     → DebugAgent
"""

from __future__ import annotations

import csv
import io
import logging
from importlib.resources import files
from typing import ClassVar, Generator

from cube.benchmark import Benchmark, BenchmarkMetadata, RuntimeContext
from cube.container import ContainerBackend
from cube.core import Action, ActionSchema, Observation
from cube.task import TaskConfig, TaskMetadata
from cube.tool import ToolboxConfig

from browsercomp_cube.benchmark import BrowseCompBenchmark
from browsercomp_cube.crypto import decrypt
from browsercomp_cube.task import BrowseCompTask, BrowseCompTaskConfig
from browsercomp_cube.tool import SubmitAnswerToolConfig

logger = logging.getLogger(__name__)

_DEBUG_CSV_PATH = files("browsercomp_cube.data") / "browse_comp_debug_set.csv"


def _load_debug_task_metadata() -> dict[str, TaskMetadata]:
    metadata: dict[str, TaskMetadata] = {}
    text = _DEBUG_CSV_PATH.read_text(encoding="utf-8")
    for idx, row in enumerate(csv.DictReader(io.StringIO(text))):
        canary = row["canary"]
        problem = decrypt(row["problem"], canary)
        answer = decrypt(row["answer"], canary)
        topic = row.get("problem_topic", "")
        task_id = f"browsecomp-debug-{idx:04d}"
        metadata[task_id] = TaskMetadata(
            id=task_id,
            abstract_description=topic,
            recommended_max_steps=5,
            extra_info={"problem": problem, "answer": answer, "topic": topic},
        )
    return metadata


_TASK_ACTIONS: dict[str, list[Action]] = {
    "browsecomp-debug-0000": [
        Action(
            name="submit_answer",
            arguments={"answer": "Explanation: debug\nExact Answer: 4\nConfidence: 100"},
        )
    ],
    "browsecomp-debug-0001": [
        Action(
            name="submit_answer",
            arguments={"answer": "Explanation: debug\nExact Answer: Paris\nConfidence: 100"},
        )
    ],
}


class DebugBrowseCompTask(BrowseCompTask):
    """BrowseCompTask with a deterministic grader — no LLM calls."""

    def _call_grader(self, prompt: str, scorer_model: str) -> bool:
        submitted = self._submit_tool().last_answer or ""
        correct_answer = self.metadata.extra_info["answer"]
        return correct_answer.lower() in submitted.lower()


class DebugBrowseCompTaskConfig(BrowseCompTaskConfig):
    """TaskConfig that produces DebugBrowseCompTask instances."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> DebugBrowseCompTask:
        task_metadata = DebugBrowseCompBenchmark.task_metadata[self.task_id]
        tool_cfg = self.tool_config or ToolboxConfig(tool_configs=[SubmitAnswerToolConfig()])
        return DebugBrowseCompTask(
            metadata=task_metadata,
            tool_config=tool_cfg,
            scorer_model=self.scorer_model,
            container_backend=container_backend,
        )


class DebugBrowseCompBenchmark(BrowseCompBenchmark):
    """Lightweight debug benchmark — 2 tasks, no network calls."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BrowseCompBenchmark.benchmark_metadata
    task_metadata: ClassVar[dict[str, TaskMetadata]] = _load_debug_task_metadata()
    task_config_class: ClassVar[type[TaskConfig]] = DebugBrowseCompTaskConfig

    def get_task_configs(self) -> Generator[DebugBrowseCompTaskConfig, None, None]:
        for tm in self.task_metadata.values():
            yield DebugBrowseCompTaskConfig(
                task_id=tm.id,
                tool_config=ToolboxConfig(tool_configs=[SubmitAnswerToolConfig()]),
                scorer_model=self.scorer_model,
            )


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


def get_debug_benchmark() -> Benchmark:
    return DebugBrowseCompBenchmark()


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


if __name__ == "__main__":
    import sys

    import browsercomp_cube.debug as _mod
    from cube.testing import run_debug_suite

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    results = run_debug_suite("browsercomp-cube", _mod)
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] < 1.0]
    sys.exit(1 if failed else 0)
