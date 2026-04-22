from typing import ClassVar

import logging
from datasets import load_dataset

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata
from math_tool_use.task import MathToolUseTaskConfig

logger = logging.getLogger(__name__)

_DATASET_NAME: str = (
    "https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_57k_collected.json"
)


def _process_open_reasoner(dataset, dataset_name):
    for item in dataset:
        # Note: Open Reasoner tasks sometimes have preamble, e.g.
        # - Example 31 (2004 College Entrance Examination Hunan Paper)
        # - 8.
        # - 4. (7 points)
        # We are currently ignoring the preamble
        task = item["0"]["value"]
        answer = "\\boxed{" + item["1"]["ground_truth"]["value"] + "}"
        yield {"dataset": dataset_name, "task": task, "answer": answer}

def load_task_metadata() -> dict[str, TaskMetadata]:
    """Load task metadata from the Open Reasoner Zero dataset."""
    dataset = load_dataset(
        "json",
        data_files=_DATASET_NAME,
        split="train",
        trust_remote_code=True,
    )
    tasks_data = [s for s in _process_open_reasoner(dataset, "open_reasoner_zero_57k") if s is not None]

    metadata: dict[str, TaskMetadata] = {}
    for i, t in enumerate(tasks_data):
        instance_id = f"q_{i}"
        metadata[instance_id] = TaskMetadata(
            id=instance_id,
            abstract_description="Solve math by tool-using with Python and submit final LaTeX answer via MathAnswer",
            recommended_max_steps=3,
            split="train",
            extra_info={
                "question": t["task"],
                "expected": t["answer"],
                "dataset": t["dataset"],
            },
        )

    logger.info(f"Loading Open Reasoner Zero dataset: {len(metadata)} tasks")
    return metadata

class MathToolUseBenchmark(Benchmark):
    """Arithmetic tasks requiring deterministic Python tool use before final LaTeX answer submission."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="math-tool-use",
        version="0.1.0",
        description="Solve arithmetic by planning, using one deterministic Python call, then submitting a LaTeX answer",
        num_tasks=4,
        tags=["example", "arithmetic", "tool-use", "python"],
    )

    task_metadata: ClassVar[dict[str, TaskMetadata]] = load_task_metadata()
    task_config_class: ClassVar[type[TaskConfig]] = MathToolUseTaskConfig

    @classmethod
    def install(cls) -> None:
        cls.benchmark_metadata = cls.benchmark_metadata.model_copy(update={"num_tasks": len(cls.task_metadata)})

    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass
