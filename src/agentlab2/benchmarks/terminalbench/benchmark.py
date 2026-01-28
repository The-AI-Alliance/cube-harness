"""Terminal-Bench benchmark implementation."""

import io
import logging
import tarfile
from pathlib import Path
from random import Random

from datasets import load_from_disk

from agentlab2.benchmark import Benchmark
from agentlab2.benchmarks.terminalbench.task import DEFAULT_IMAGE, TerminalBenchTask
from agentlab2.environment import EnvConfig
from agentlab2.tools.daytona import DaytonaSWEToolConfig

logger = logging.getLogger(__name__)

# Default dataset path
DEFAULT_DATASET_PATH = "./data/terminal_bench"


def _has_dockerfile(archive: bytes) -> bool:
    """Check if archive contains a Dockerfile."""
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            for member in tar.getnames():
                if member.endswith("Dockerfile") or "/Dockerfile" in member:
                    return True
    except Exception as e:
        logger.warning(f"Failed to check archive for Dockerfile: {e}")
    return False


class TerminalBenchBenchmark(Benchmark):
    """Terminal-Bench benchmark for evaluating agents on real-world terminal tasks.

    Terminal-Bench evaluates how well agents handle tasks like:
    - Compiling code and building software
    - Training ML models
    - Setting up servers and debugging systems
    - File operations and data processing

    Tasks are loaded from a local HuggingFace dataset (exported via export_terminal_bench.py).
    Each task includes an instruction, a Docker environment, and pytest-based validation.

    Scoring is binary: all tests must pass for reward=1.0.

    Attributes:
        dataset_path: Path to the local dataset directory.
        shuffle: Whether to shuffle tasks.
        shuffle_seed: Random seed for shuffling.
        max_tasks: Maximum number of tasks to load (None = all).
        difficulty_filter: Filter by difficulty ("easy", "medium", "hard").
        category_filter: Filter by category (e.g., "scientific-computing").
    """

    dataset_path: str = DEFAULT_DATASET_PATH
    shuffle: bool = True
    shuffle_seed: int = 42
    max_tasks: int | None = None
    difficulty_filter: str | None = None
    category_filter: str | None = None
    task_ids: list[str] | None = None  # Specific task IDs to load
    oracle_mode: bool = False  # If True, upload solution.sh for oracle agent

    _dataset: list | None = None
    _tasks: list[TerminalBenchTask] | None = None

    model_config = {"arbitrary_types_allowed": True}

    def setup(self) -> None:
        """Load the dataset from disk."""
        logger.info(f"Loading Terminal-Bench dataset from {self.dataset_path}")

        dataset_path = Path(self.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Terminal-Bench dataset not found at {self.dataset_path}. "
                "Run `uv run scripts/export_terminal_bench.py` to download it."
            )

        try:
            ds = load_from_disk(str(dataset_path))
            self._dataset = list(ds)
            logger.info(f"Loaded {len(self._dataset)} tasks from Terminal-Bench")
        except Exception as e:
            raise RuntimeError(f"Failed to load Terminal-Bench dataset: {e}")

    def load_tasks(self) -> list[TerminalBenchTask]:
        """Load tasks from the dataset."""
        if self._tasks is not None:
            return self._tasks

        if self._dataset is None:
            self.setup()

        tasks_data = self._dataset

        # Filter by specific task IDs
        if self.task_ids:
            task_id_set = set(self.task_ids)
            tasks_data = [t for t in tasks_data if t["task_id"] in task_id_set]
            logger.info(f"Filtered to {len(tasks_data)} tasks by ID")

        # Filter by difficulty
        if self.difficulty_filter:
            tasks_data = [t for t in tasks_data if t.get("difficulty", "").lower() == self.difficulty_filter.lower()]
            logger.info(f"Filtered to {len(tasks_data)} {self.difficulty_filter} tasks")

        # Filter by category
        if self.category_filter:
            tasks_data = [t for t in tasks_data if t.get("category", "").lower() == self.category_filter.lower()]
            logger.info(f"Filtered to {len(tasks_data)} {self.category_filter} tasks")

        # Shuffle if requested
        if self.shuffle:
            rng = Random(self.shuffle_seed)
            tasks_data = list(tasks_data)
            rng.shuffle(tasks_data)

        # Limit tasks if specified
        if self.max_tasks:
            tasks_data = tasks_data[: self.max_tasks]

        # Create task objects (Dockerfile content is extracted separately for env_configs)
        tasks = []
        for t in tasks_data:
            task = TerminalBenchTask(
                id=t["task_id"],
                instruction=t["base_description"],
                archive=t["archive"],
                difficulty=t.get("difficulty", "unknown"),
                category=t.get("category", ""),
                tags=t.get("tags", []),
                max_agent_timeout_sec=t.get("max_agent_timeout_sec", 900),
                max_test_timeout_sec=t.get("max_test_timeout_sec", 180),
                oracle_mode=self.oracle_mode,
            )
            tasks.append(task)

        logger.info(f"Created {len(tasks)} Terminal-Bench tasks")
        self._tasks = tasks
        return tasks

    def env_configs(self) -> list[EnvConfig]:
        """Generate environment configurations with task-specific Dockerfiles.

        Each task gets its own tool config with the archive bytes. Daytona extracts
        the archive and builds the image from the Dockerfile (if present).
        """
        tasks = self.load_tasks()

        # Create a task-specific tool config for each task
        configs = []
        for task in tasks:
            # Get base config settings
            base_config = self.tool_config
            if not isinstance(base_config, DaytonaSWEToolConfig):
                raise TypeError(f"TerminalBenchBenchmark requires DaytonaSWEToolConfig, got {type(base_config)}")

            # Check if archive has a Dockerfile
            has_dockerfile = _has_dockerfile(task.archive)

            # Create task-specific config with archive (for Dockerfile build context)
            task_tool_config = DaytonaSWEToolConfig(
                api_key=base_config.api_key,
                image=DEFAULT_IMAGE,  # Fallback if no Dockerfile
                dockerfile_archive=task.archive if has_dockerfile else None,
                cpus=base_config.cpus,
                memory_gb=base_config.memory_gb,
                disk_gb=base_config.disk_gb,
                ephemeral=base_config.ephemeral,
            )

            configs.append(EnvConfig(task=task, tool_config=task_tool_config))
            if has_dockerfile:
                logger.debug(f"Task {task.id}: building from Dockerfile")
            else:
                logger.debug(f"Task {task.id}: using default image {DEFAULT_IMAGE}")

        return configs

    def close(self) -> None:
        """Clean up resources."""
        self._dataset = None
        self._tasks = None
