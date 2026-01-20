"""LiveCodeBench benchmark implementation.

LiveCodeBench is a temporally updating benchmark for code generation.
Homepage: https://livecodebench.github.io/
Dataset: https://huggingface.co/datasets/livecodebench/code_generation_lite
"""

import json
import logging
from random import Random

from huggingface_hub import hf_hub_download

from agentlab2.benchmark import Benchmark
from agentlab2.benchmarks.livecodebench.task import LiveCodeBenchTask

logger = logging.getLogger(__name__)

# JSONL files in the dataset
JSONL_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]


class LiveCodeBenchBenchmark(Benchmark):
    """LiveCodeBench benchmark for code generation tasks.

    Uses DaytonaSWETool to execute code in sandboxed environments.
    """

    shuffle: bool = True
    shuffle_seed: int = 42
    max_tasks: int | None = None
    difficulty_filter: str | None = None  # "easy", "medium", "hard"
    jsonl_files: list[str] = ["test6.jsonl"]  # Which files to load

    _dataset: list | None = None

    model_config = {"arbitrary_types_allowed": True}

    def setup(self) -> None:
        """Download and prepare the dataset."""
        logger.info("Loading LiveCodeBench dataset")

        try:
            all_tasks = []
            for jsonl_file in self.jsonl_files:
                if jsonl_file not in JSONL_FILES:
                    logger.warning(f"Skipping unknown file: {jsonl_file}")
                    continue

                # Download the JSONL file
                local_path = hf_hub_download(
                    repo_id="livecodebench/code_generation_lite",
                    filename=jsonl_file,
                    repo_type="dataset",
                )

                # Load tasks from JSONL
                with open(local_path, encoding="utf-8") as f:
                    for line in f:
                        task_data = json.loads(line)
                        all_tasks.append(task_data)

                logger.info(f"Loaded {len(all_tasks)} tasks from {jsonl_file}")

            self._dataset = all_tasks
            logger.info(f"Total: {len(self._dataset)} tasks from LiveCodeBench")
        except Exception as e:
            raise RuntimeError(f"Failed to load LiveCodeBench dataset: {e}")

    def load_tasks(self) -> list[LiveCodeBenchTask]:
        """Load tasks from the dataset."""
        if self._dataset is None:
            self.setup()

        tasks_data = self._dataset

        # Filter by difficulty if specified
        if self.difficulty_filter:
            tasks_data = [t for t in tasks_data if t.get("difficulty", "").lower() == self.difficulty_filter.lower()]
            logger.info(f"Filtered to {len(tasks_data)} {self.difficulty_filter} tasks")

        # Shuffle if requested
        if self.shuffle:
            rng = Random(self.shuffle_seed)
            tasks_data = list(tasks_data)
            rng.shuffle(tasks_data)

        # Limit tasks if specified
        if self.max_tasks:
            tasks_data = tasks_data[: self.max_tasks]

        tasks = [
            LiveCodeBenchTask(
                id=task_data["question_id"],
                question_title=task_data["question_title"],
                question_content=task_data["question_content"],
                platform=task_data["platform"],
                difficulty=task_data["difficulty"],
                starter_code=task_data.get("starter_code", ""),
                public_test_cases=task_data.get("public_test_cases", ""),
                private_test_cases=task_data.get("private_test_cases", ""),
                metadata=task_data.get("metadata", ""),
            )
            for task_data in tasks_data
        ]

        logger.info(f"Created {len(tasks)} LiveCodeBench tasks")
        return tasks

    def close(self) -> None:
        """Clean up resources."""
        self._dataset = None
