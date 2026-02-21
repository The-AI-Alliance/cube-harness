"""Terminal-Bench 2 benchmark implementation."""

import io
import logging
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path
from random import Random

from datasets import Dataset, load_from_disk

from agentlab2.benchmark import Benchmark
from agentlab2.benchmarks.terminalbench.task import DEFAULT_IMAGE, TerminalBenchTask
from agentlab2.environment import EnvConfig
from agentlab2.tools.daytona import DaytonaSWEToolConfig
from agentlab2.tools.docker import DockerSWEToolConfig

# Type alias for supported SWE tool configs
SWEToolConfig = DaytonaSWEToolConfig | DockerSWEToolConfig

logger = logging.getLogger(__name__)

# Default dataset path (Terminal-Bench 2)
DEFAULT_DATASET_PATH = str(Path.home() / ".agentlab" / "data" / "terminal_bench_v2")


def _create_task_archive(task_dir: Path) -> bytes:
    """Create a tar.gz archive of a task directory."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for item in task_dir.rglob("*"):
            if item.is_file():
                arcname = str(item.relative_to(task_dir))
                tar.add(item, arcname=arcname)
    return buffer.getvalue()


def _load_task_from_repo(task_dir: Path) -> dict | None:
    """Load a single task from a Terminal-Bench repo directory."""
    task_toml = task_dir / "task.toml"
    instruction_md = task_dir / "instruction.md"

    if not task_toml.exists() or not instruction_md.exists():
        return None

    with open(task_toml, "rb") as f:
        config = tomllib.load(f)

    instruction = instruction_md.read_text(encoding="utf-8").strip()

    metadata = config.get("metadata", {})
    env_config = config.get("environment", {})
    agent_config = config.get("agent", {})
    verifier_config = config.get("verifier", {})

    archive = _create_task_archive(task_dir)

    return {
        "task_id": task_dir.name,
        "base_description": instruction,
        "archive": archive,
        "difficulty": metadata.get("difficulty", "unknown"),
        "category": metadata.get("category", ""),
        "tags": metadata.get("tags", []),
        "docker_image": env_config.get("docker_image", "python:3.13"),
        "cpus": env_config.get("cpus", 1),
        "memory": env_config.get("memory", "4G"),
        "storage": env_config.get("storage", "10G"),
        "max_agent_timeout_sec": int(agent_config.get("timeout_sec", 900)),
        "max_test_timeout_sec": int(verifier_config.get("timeout_sec", 900)),
    }


def _parse_memory_str(mem_str: str) -> int:
    """Parse memory string like '4G' to GB integer."""
    if isinstance(mem_str, int):
        return mem_str
    mem_str = str(mem_str).upper().strip()
    if mem_str.endswith("G"):
        return int(mem_str[:-1])
    if mem_str.endswith("GB"):
        return int(mem_str[:-2])
    if mem_str.endswith("M"):
        return max(1, int(mem_str[:-1]) // 1024)
    return 4  # Default


def _parse_storage_str(storage_str: str) -> int:
    """Parse storage string like '10G' to GB integer."""
    if isinstance(storage_str, int):
        return storage_str
    storage_str = str(storage_str).upper().strip()
    if storage_str.endswith("G"):
        return int(storage_str[:-1])
    if storage_str.endswith("GB"):
        return int(storage_str[:-2])
    return 10  # Default


class TerminalBenchBenchmark(Benchmark):
    """Terminal-Bench 2 benchmark — real-world terminal tasks with pytest-based validation.

    Tasks are loaded from a local HuggingFace dataset (see ``install()``).
    Scoring is binary: all tests must pass for reward=1.0.
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
        """Load the pre-installed dataset from disk into memory."""
        logger.info(f"Loading Terminal-Bench dataset from {self.dataset_path}")

        dataset_path = Path(self.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Terminal-Bench dataset not found at {self.dataset_path}. Run benchmark.install() to download it."
            )

        try:
            ds = load_from_disk(str(dataset_path))
            self._dataset = list(ds)
            logger.info(f"Loaded {len(self._dataset)} tasks from Terminal-Bench")
        except Exception as e:
            raise RuntimeError(f"Failed to load Terminal-Bench dataset: {e}")

    def load_tasks(self) -> list[TerminalBenchTask]:
        """Load, filter, and return task objects from the dataset."""
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

        # Create task objects with TB2 fields
        tasks = []
        for t in tasks_data:
            task = TerminalBenchTask(
                id=t["task_id"],
                instruction=t["base_description"],
                archive=t["archive"],
                difficulty=t.get("difficulty", "unknown"),
                category=t.get("category", ""),
                tags=t.get("tags", []),
                docker_image=t.get("docker_image", DEFAULT_IMAGE),
                cpus=t.get("cpus", 1),
                memory=t.get("memory", "4G"),
                storage=t.get("storage", "10G"),
                max_agent_timeout_sec=t.get("max_agent_timeout_sec", 900),
                max_test_timeout_sec=t.get("max_test_timeout_sec", 900),
                oracle_mode=self.oracle_mode,
            )
            tasks.append(task)

        logger.info(f"Created {len(tasks)} Terminal-Bench tasks")
        self._tasks = tasks
        return tasks

    def env_configs(self) -> list[EnvConfig]:
        """Generate per-task environment configs with task-specific Docker images."""
        tasks = self.load_tasks()

        # Create a task-specific tool config for each task
        configs = []
        for task in tasks:
            base_config = self.tool_config
            task_tool_config = self._create_task_tool_config(base_config, task)
            configs.append(EnvConfig(task=task, tool_config=task_tool_config))
            logger.debug(f"Task {task.id}: using image {task.docker_image}")

        return configs

    def _create_task_tool_config(self, base_config: SWEToolConfig, task: TerminalBenchTask) -> SWEToolConfig:
        """Clone ``base_config`` with task-specific image, CPU, memory, and disk settings."""
        if isinstance(base_config, DaytonaSWEToolConfig):
            return DaytonaSWEToolConfig(
                api_key=base_config.api_key,
                image=task.docker_image,
                cpus=task.cpus,
                memory_gb=_parse_memory_str(task.memory),
                disk_gb=_parse_storage_str(task.storage),
                max_output_bytes=base_config.max_output_bytes,
                ephemeral=base_config.ephemeral,
                auto_stop_minutes=base_config.auto_stop_minutes,
                auto_delete_minutes=base_config.auto_delete_minutes,
            )
        elif isinstance(base_config, DockerSWEToolConfig):
            return DockerSWEToolConfig(
                image=task.docker_image,
                cpus=task.cpus,
                memory_gb=_parse_memory_str(task.memory),
                disk_gb=_parse_storage_str(task.storage),
                working_dir=base_config.working_dir,
                network_mode=base_config.network_mode,
                user=base_config.user,
                enforce_disk_quota=base_config.enforce_disk_quota,
                writable_tmpfs_dirs=base_config.writable_tmpfs_dirs,
                max_output_bytes=base_config.max_output_bytes,
                remove_on_close=base_config.remove_on_close,
                pull_policy=base_config.pull_policy,
            )
        else:
            raise TypeError(
                f"TerminalBenchBenchmark requires DaytonaSWEToolConfig or DockerSWEToolConfig, "
                f"got {type(base_config).__name__}"
            )

    def install(self) -> None:
        """Clone the terminal-bench-2 repo and export tasks as a HuggingFace dataset."""
        outdir = Path(self.dataset_path).resolve()
        if outdir.exists():
            logger.info(f"Dataset already exists at {outdir}, skipping install")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "terminal-bench-2"

            logger.info("Cloning laude-institute/terminal-bench-2...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/laude-institute/terminal-bench-2.git",
                    str(repo_dir),
                ],
                check=True,
                timeout=300,
            )

            tasks = []
            for item in sorted(repo_dir.iterdir()):
                if item.is_dir() and (item / "task.toml").exists():
                    task = _load_task_from_repo(item)
                    if task:
                        tasks.append(task)
                        logger.info(
                            f"  Loaded task: {task['task_id']} ({task['difficulty']}, image: {task['docker_image']})"
                        )

            logger.info(f"Loaded {len(tasks)} tasks from Terminal-Bench")

            ds = Dataset.from_list(tasks)
            outdir.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(outdir))

        logger.info(f"Dataset saved to: {outdir}")

    def close(self) -> None:
        """Clean up resources."""
        self._dataset = None
        self._tasks = None
