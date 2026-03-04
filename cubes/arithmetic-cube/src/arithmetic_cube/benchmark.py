from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata
from arithmetic_cube.task import ArithmeticTaskConfig


class ArithmeticBenchmark(Benchmark):
    """Registry of arithmetic tasks — no shared infrastructure needed."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="arithmetic-cube",
        version="0.1.0",
        description="Simple arithmetic tasks — submit the correct integer answer",
        num_tasks=4,
        tags=["example", "arithmetic"],
    )

    task_metadata: ClassVar[dict[str, TaskMetadata]] = {
        "add-3-4": TaskMetadata(
            id="add-3-4",
            abstract_description="Compute 3 + 4 and submit the answer",
            recommended_max_steps=2,
            extra_info={"a": 3, "b": 4, "op": "+", "expected": 7},
        ),
        "sub-10-3": TaskMetadata(
            id="sub-10-3",
            abstract_description="Compute 10 - 3 and submit the answer",
            recommended_max_steps=2,
            extra_info={"a": 10, "b": 3, "op": "-", "expected": 7},
        ),
        "mul-6-7": TaskMetadata(
            id="mul-6-7",
            abstract_description="Compute 6 × 7 and submit the answer",
            recommended_max_steps=2,
            extra_info={"a": 6, "b": 7, "op": "*", "expected": 42},
        ),
        "add-100-1": TaskMetadata(
            id="add-100-1",
            abstract_description="Compute 100 + 1 and submit the answer",
            recommended_max_steps=2,
            extra_info={"a": 100, "b": 1, "op": "+", "expected": 101},
        ),
    }

    task_config_class: ClassVar[type[TaskConfig]] = ArithmeticTaskConfig

    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass
