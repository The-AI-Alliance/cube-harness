"""Benchmark for terminalbench-cube — real-world terminal tasks with pytest-based validation."""

import io
import logging
import subprocess
import tarfile
import tempfile
import tomllib
from collections.abc import Generator
from pathlib import Path
from random import Random
from typing import Any, ClassVar

from datasets import Dataset, load_from_disk

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.container import ContainerConfig
from cube.task import TaskConfig, TaskMetadata
from terminalbench_cube.task import TerminalBenchTaskConfig

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = str(Path.home() / ".agentlab" / "data" / "terminal_bench_v2")


def _create_task_archive(task_dir: Path) -> bytes:
    """Create a tar.gz archive of a task directory."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for item in task_dir.rglob("*"):
            if item.is_file():
                tar.add(item, arcname=str(item.relative_to(task_dir)))
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

    return {
        "task_id": task_dir.name,
        "base_description": instruction,
        "archive": _create_task_archive(task_dir),
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


def _parse_gb(value: str | int) -> float:
    """Parse a memory/storage string like '4G' to a float in GB."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).upper().strip()
    if s.endswith("GB"):
        return float(s[:-2])
    if s.endswith("G"):
        return float(s[:-1])
    if s.endswith("M"):
        return float(s[:-1]) / 1024
    return 4.0


class TerminalBenchBenchmark(Benchmark):
    """Terminal-Bench 2 — real-world terminal tasks with pytest-based validation."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="terminalbench-cube",
        version="0.1.0",
        description="Real-world terminal tasks (compile, debug, deploy) with pytest-based validation",
        tags=["terminal", "swe", "docker"],
    )

    # Populated dynamically in _setup() from the HF dataset.
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {}
    task_config_class: ClassVar[type[TaskConfig]] = TerminalBenchTaskConfig

    # User-configurable fields
    dataset_path: str = DEFAULT_DATASET_PATH
    shuffle: bool = True
    shuffle_seed: int = 42
    max_tasks: int | None = None
    difficulty_filter: str | None = None
    category_filter: str | None = None
    task_ids: list[str] | None = None
    oracle_mode: bool = False

    def _setup(self) -> None:
        """Load dataset and populate task_metadata dynamically."""
        dataset_path = Path(self.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Terminal-Bench dataset not found at {self.dataset_path}. "
                "Run benchmark.install() first."
            )

        ds = load_from_disk(str(dataset_path))
        tasks_data = list(ds)
        logger.info(f"Loaded {len(tasks_data)} tasks from Terminal-Bench")

        if self.task_ids:
            id_set = set(self.task_ids)
            tasks_data = [t for t in tasks_data if t["task_id"] in id_set]
        if self.difficulty_filter:
            tasks_data = [t for t in tasks_data if t.get("difficulty", "").lower() == self.difficulty_filter.lower()]
        if self.category_filter:
            tasks_data = [t for t in tasks_data if t.get("category", "").lower() == self.category_filter.lower()]
        if self.shuffle:
            rng = Random(self.shuffle_seed)
            tasks_data = list(tasks_data)
            rng.shuffle(tasks_data)
        if self.max_tasks:
            tasks_data = tasks_data[: self.max_tasks]

        metadata: dict[str, TaskMetadata] = {}
        for t in tasks_data:
            tid = t["task_id"]
            metadata[tid] = TaskMetadata(
                id=tid,
                abstract_description=t["base_description"][:200],
                recommended_max_steps=t.get("max_agent_timeout_sec", 900) // 10,
                container_config=ContainerConfig(
                    image=t.get("docker_image", "python:3.13"),
                    cpu_cores=float(t.get("cpus", 1)),
                    ram_gb=_parse_gb(t.get("memory", "4G")),
                    disk_gb=_parse_gb(t.get("storage", "10G")),
                ),
                extra_info={
                    "instruction": t["base_description"],
                    "archive": t["archive"],
                    "difficulty": t.get("difficulty", "unknown"),
                    "category": t.get("category", ""),
                    "tags": t.get("tags", []),
                    "max_agent_timeout_sec": t.get("max_agent_timeout_sec", 900),
                    "max_test_timeout_sec": t.get("max_test_timeout_sec", 900),
                    "oracle_mode": self.oracle_mode,
                },
            )

        # Set on the class so TaskConfig.make() can look it up
        TerminalBenchBenchmark.task_metadata = metadata
        TerminalBenchBenchmark.benchmark_metadata = BenchmarkMetadata(
            name=self.benchmark_metadata.name,
            version=self.benchmark_metadata.version,
            description=self.benchmark_metadata.description,
            num_tasks=len(metadata),
            tags=self.benchmark_metadata.tags,
        )
        logger.info(f"Terminal-Bench setup complete: {len(metadata)} tasks")

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        """Yield TaskConfigs with container_backend injected."""
        for tm in self.task_metadata.values():
            yield TerminalBenchTaskConfig(
                task_id=tm.id,
                tool_config=self.default_tool_config,
                container_backend=self.container_backend,
            )

    def close(self) -> None:
        """Release dataset from memory."""
        pass

    def install(self) -> None:
        """Clone terminal-bench-2 repo and export as HuggingFace dataset."""
        outdir = Path(self.dataset_path).resolve()
        if outdir.exists():
            logger.info(f"Dataset already exists at {outdir}, skipping install")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "terminal-bench-2"
            logger.info("Cloning laude-institute/terminal-bench-2...")
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/laude-institute/terminal-bench-2.git", str(repo_dir)],
                check=True,
                timeout=300,
            )

            tasks = []
            for item in sorted(repo_dir.iterdir()):
                if item.is_dir() and (item / "task.toml").exists():
                    task = _load_task_from_repo(item)
                    if task:
                        tasks.append(task)
                        logger.info(f"  Loaded: {task['task_id']} ({task['difficulty']})")

            ds = Dataset.from_list(tasks)
            outdir.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(outdir))
        logger.info(f"Dataset saved to: {outdir}")
