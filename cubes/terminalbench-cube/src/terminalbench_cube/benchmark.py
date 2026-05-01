"""Benchmark for terminalbench-cube — real-world terminal tasks with pytest-based validation."""

from __future__ import annotations

import base64
import io
import json
import logging
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

from cube import LocalInfraConfig
from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.resource import InfraConfig
from cube.task import TaskConfig

from terminalbench_cube.task import TerminalBenchTaskConfig, TerminalBenchTaskMetadata

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/laude-institute/terminal-bench-2.git"


def _build_execution_info(task: dict) -> dict:
    """Extract execution-only fields from a raw task dict.

    These fields are only needed when a task runs; they are never loaded at
    import time. Stored in the per-task execution cache by install().
    """
    archive = task["archive"]
    archive_b64 = base64.b64encode(archive).decode() if isinstance(archive, bytes) else archive
    return {
        "instruction": task["base_description"],
        "archive": archive_b64,
        "max_test_timeout_sec": int(task.get("max_test_timeout_sec", 900)),
    }


class TerminalBenchBenchmark(Benchmark["TerminalBenchBenchmarkConfig"]):
    """Runtime pair — owns the infra reference passed to ``make(infra)`` and
    publishes it into ``runtime_context["infra"]`` so per-task container launches
    flow through ``Task.runtime_context``.
    """

    def __init__(self, config: "TerminalBenchBenchmarkConfig", infra: InfraConfig | None = None) -> None:
        super().__init__(config)
        self._infra = infra

    def _setup(self) -> None:
        """Publish the shared InfraConfig to runtime_context; per-task containers are launched per-task in make()."""
        if self._infra is not None:
            # GC orphans from earlier crashed runs so stale containers don't pile up.
            self._infra.cleanup_stale()
            self._runtime_context["infra"] = self._infra
        logger.info(
            "TerminalBenchBenchmark ready with %d tasks (infra=%s)",
            self.config.num_tasks,
            self._infra.fingerprint() if self._infra is not None else "<none>",
        )

    def close(self) -> None:
        logger.info("Terminal-Bench benchmark closed")


class TerminalBenchBenchmarkConfig(BenchmarkConfig[TerminalBenchTaskMetadata]):
    """Terminal-Bench 2 — real-world terminal tasks with pytest-based validation."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="terminalbench-cube",
        version="0.1.0",
        description=(
            "Real-world terminal tasks (compile, debug, deploy) with pytest-based validation.\n"
            "\n"
            "CUBE DEVELOPER NOTES:\n"
            "---------------------\n"
            "task_metadata.json is a shipped package resource containing lightweight public fields. "
            "Heavy execution data (instruction, archive) is stored in the per-task execution cache "
            "populated by install(). To regenerate task_metadata.json (developer use only), run: "
            "scripts/create_task_metadata.py"
        ),
        tags=["terminal", "swe", "docker"],
        num_tasks=89,
        named_subsets={
            # Difficulty levels
            "easy": ("difficulty", "easy"),
            "medium": ("difficulty", "medium"),
            "hard": ("difficulty", "hard"),
            # Categories
            "data-processing": ("category", "data-processing"),
            "data-querying": ("category", "data-querying"),
            "data-science": ("category", "data-science"),
            "debugging": ("category", "debugging"),
            "file-operations": ("category", "file-operations"),
            "games": ("category", "games"),
            "machine-learning": ("category", "machine-learning"),
            "mathematics": ("category", "mathematics"),
            "model-training": ("category", "model-training"),
            "optimization": ("category", "optimization"),
            "personal-assistant": ("category", "personal-assistant"),
            "scientific-computing": ("category", "scientific-computing"),
            "security": ("category", "security"),
            "software-engineering": ("category", "software-engineering"),
            "system-administration": ("category", "system-administration"),
            "video-processing": ("category", "video-processing"),
        },
    )
    task_config_class: ClassVar[type[TaskConfig]] = TerminalBenchTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = TerminalBenchBenchmark

    # User-configurable fields
    oracle_mode: bool = False

    # ── Benchmark lifecycle ────────────────────────────────────────

    @classmethod
    def install(cls) -> None:
        """Clone terminal-bench-2 repo and populate the per-task execution cache.

        Downloads the task archive and instruction for each task and writes one
        JSON file per task into task_execution_cache_dir(). Idempotent: skips if
        the cache directory already exists and is non-empty.

        The shipped task_metadata.json is a package resource and is not modified here.
        To regenerate task_metadata.json (developer use only), run:
            scripts/create_task_metadata.py
        """
        exec_cache_dir = cls.task_config_class.task_execution_cache_dir()
        if exec_cache_dir.exists() and any(exec_cache_dir.iterdir()):
            logger.info("Execution cache already populated, skipping installation")
            return
        exec_cache_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "terminal-bench-2"
            logger.info("Cloning laude-institute/terminal-bench-2...")
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
                check=True,
                timeout=300,
            )

            n = 0
            for item in sorted(repo_dir.iterdir()):
                if item.is_dir() and (item / "task.toml").exists():
                    task = cls._load_task_from_repo(item)
                    if task:
                        exec_info = _build_execution_info(task)
                        (exec_cache_dir / f"{task['task_id']}.json").write_text(json.dumps(exec_info))
                        n += 1
                        logger.info(f"  Cached: {task['task_id']}")

        logger.info(f"Saved {n} execution cache files to {exec_cache_dir}")

    @classmethod
    def uninstall(cls) -> None:
        """Remove the per-task execution cache.

        The shipped task_metadata.json is not removed.
        """
        exec_cache_dir = cls.task_config_class.task_execution_cache_dir()
        if exec_cache_dir.exists():
            shutil.rmtree(exec_cache_dir)
            logger.info(f"Removed execution cache at {exec_cache_dir}")

    def make(self, infra: InfraConfig | None = None) -> TerminalBenchBenchmark:
        """Override to forward ``infra`` into the runtime constructor."""
        resolved_infra = infra or LocalInfraConfig()
        if self.resources:
            for resource in self.resources:
                if resolved_infra.provision_status(resource) == "ready":
                    logger.info(
                        "Resource %s already provisioned on %s",
                        resource.name,
                        resolved_infra.fingerprint(),
                    )
                    continue
                logger.info("Provisioning resource %s on %s...", resource.name, resolved_infra.fingerprint())
                resolved_infra.provision(resource)
        bench = TerminalBenchBenchmark(config=self, infra=resolved_infra)
        bench.setup()
        return bench

    def get_task_configs(self) -> Generator[TerminalBenchTaskConfig, None, None]:
        """Yield TaskConfigs with oracle_mode forwarded from benchmark settings."""
        for tm in self.tasks().values():
            yield TerminalBenchTaskConfig(
                metadata=tm,
                tool_config=self.tool_config,
                oracle_mode=self.oracle_mode,
            )

    # ── Private helpers ────────────────────────────────────────────

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
