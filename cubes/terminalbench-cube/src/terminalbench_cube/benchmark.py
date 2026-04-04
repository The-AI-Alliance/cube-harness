"""Benchmark for terminalbench-cube — real-world terminal tasks with pytest-based validation."""

import base64
import io
import json
import logging
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path
from random import Random
from typing import ClassVar

from cube import get_cache_dir
from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.container import ContainerConfig
from cube.task import TaskConfig, TaskMetadata
from terminalbench_cube.task import TerminalBenchTaskConfig

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = get_cache_dir("terminal_bench_v2")

REPO_URL = "https://github.com/laude-institute/terminal-bench-2.git"

_TASK_METADATA_JSON = Path(__file__).parent / "task_metadata.json"


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


def _build_task_metadata(tasks: list[dict]) -> dict[str, TaskMetadata]:
    """Build task_metadata from raw task dicts. archive is stored as a base64 string."""
    metadata: dict[str, TaskMetadata] = {}
    for t in tasks:
        tid = t["task_id"]
        archive = t["archive"]
        archive_b64 = base64.b64encode(archive).decode() if isinstance(archive, bytes) else archive
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
                "archive": archive_b64,
                "difficulty": t.get("difficulty", "unknown"),
                "category": t.get("category", ""),
                "tags": t.get("tags", []),
                "max_agent_timeout_sec": t.get("max_agent_timeout_sec", 900),
                "max_test_timeout_sec": t.get("max_test_timeout_sec", 900),
                "oracle_mode": False,
            },
        )
    return metadata


class TerminalBenchBenchmark(Benchmark):
    """Terminal-Bench 2 — real-world terminal tasks with pytest-based validation."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="terminalbench-cube",
        version="0.1.0",
        description="Real-world terminal tasks (compile, debug, deploy) with pytest-based validation",
        tags=["terminal", "swe", "docker"],
        num_tasks=89,
    )

    task_metadata: ClassVar[dict[str, TaskMetadata]] = {}
    task_config_class: ClassVar[type[TaskConfig]] = TerminalBenchTaskConfig

    # User-configurable fields
    shuffle: bool = True
    shuffle_seed: int = 42
    max_tasks: int | None = None
    difficulty_filter: str | None = None
    category_filter: str | None = None
    task_ids: list[str] | None = None
    oracle_mode: bool = False

    # ── Benchmark lifecycle ────────────────────────────────────────

    @classmethod
    def install(cls) -> None:
        """Clone terminal-bench-2 repo and save task_metadata.json.

        Clones laude-institute/terminal-bench-2, reads each task directory,
        and saves all task data (including base64-encoded archives) to
        task_metadata.json next to this module.

        Safe to call multiple times: skips if task_metadata.json already exists.
        """
        if _TASK_METADATA_JSON.exists():
            logger.info("task_metadata.json already exists, skipping installation")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "terminal-bench-2"
            logger.info("Cloning laude-institute/terminal-bench-2...")
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
                check=True,
                timeout=300,
            )

            tasks = []
            for item in sorted(repo_dir.iterdir()):
                if item.is_dir() and (item / "task.toml").exists():
                    task = cls._load_task_from_repo(item)
                    if task:
                        tasks.append(task)
                        logger.info(f"  Loaded: {task['task_id']} ({task['difficulty']})")

        metadata = _build_task_metadata(tasks)
        _TASK_METADATA_JSON.write_text(json.dumps([tm.model_dump() for tm in metadata.values()], indent=2))
        cls.task_metadata = metadata
        logger.info(f"Saved {len(metadata)} tasks to {_TASK_METADATA_JSON}")

    @classmethod
    def uninstall(cls) -> None:
        """Remove task_metadata.json."""
        if _TASK_METADATA_JSON.exists():
            _TASK_METADATA_JSON.unlink()
            cls.task_metadata = {}
            logger.info(f"Removed {_TASK_METADATA_JSON}")

    def _setup(self) -> None:
        """Apply instance-level filters and runtime config to the pre-loaded task_metadata."""
        if "task_metadata" in self.__dict__:
            logger.info("Task metadata already loaded, skipping setup")
            return

        tasks = list(type(self).task_metadata.values())
        tasks = self._filter_tasks(tasks)

        if self.oracle_mode:
            tasks = [
                t.model_copy(update={"extra_info": {**t.extra_info, "oracle_mode": True}})
                for t in tasks
            ]

        metadata = {t.id: t for t in tasks}
        object.__setattr__(self, "task_metadata", metadata)
        type(self).task_metadata = metadata
        logger.info(f"Terminal-Bench setup complete: {len(metadata)} tasks")

    def close(self) -> None:
        logger.info("Terminal-Bench benchmark closed")

    # ── Private helpers ────────────────────────────────────────────

    def _filter_tasks(self, tasks: list[TaskMetadata]) -> list[TaskMetadata]:
        """Apply filtering, shuffling, and slicing to a list of TaskMetadata."""
        if self.task_ids:
            id_set = set(self.task_ids)
            tasks = [t for t in tasks if t.id in id_set]
        if self.difficulty_filter:
            tasks = [t for t in tasks if t.extra_info.get("difficulty", "").lower() == self.difficulty_filter.lower()]
        if self.category_filter:
            tasks = [t for t in tasks if t.extra_info.get("category", "").lower() == self.category_filter.lower()]
        if self.shuffle:
            Random(self.shuffle_seed).shuffle(tasks)
        if self.max_tasks:
            tasks = tasks[: self.max_tasks]
        return tasks

    @staticmethod
    def _load_task_from_repo(task_dir: Path) -> dict | None:
        """Load a single task from a Terminal-Bench repo directory."""
        if not (task_dir / "task.toml").exists() or not (task_dir / "instruction.md").exists():
            return None

        with open(task_dir / "task.toml", "rb") as f:
            config = tomllib.load(f)

        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
        meta = config.get("metadata", {})
        env = config.get("environment", {})
        agent = config.get("agent", {})
        verifier = config.get("verifier", {})

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for item in task_dir.rglob("*"):
                if item.is_file():
                    tar.add(item, arcname=str(item.relative_to(task_dir)))

        return {
            "task_id": task_dir.name,
            "base_description": instruction,
            "archive": buf.getvalue(),
            "difficulty": meta.get("difficulty", "unknown"),
            "category": meta.get("category", ""),
            "tags": meta.get("tags", []),
            "docker_image": env.get("docker_image", "python:3.13"),
            "cpus": env.get("cpus", 1),
            "memory": env.get("memory", "4G"),
            "storage": env.get("storage", "10G"),
            "max_agent_timeout_sec": int(agent.get("timeout_sec", 900)),
            "max_test_timeout_sec": int(verifier.get("timeout_sec", 900)),
        }
