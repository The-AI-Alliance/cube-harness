"""Benchmark for swebench-verified-cube — SWE-bench Verified with test-based validation."""

import json
import logging
import shutil
from pathlib import Path
from random import Random
from typing import Any, ClassVar

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.container import ContainerConfig
from cube.task import TaskConfig, TaskMetadata
from datasets import load_dataset

from swebench_verified_cube.task import SWEBenchVerifiedTaskConfig

logger = logging.getLogger(__name__)

_DOCKER_NAMESPACE = "swebench"
_IMAGE_TAG = "latest"
_TASK_METADATA_JSON = Path(__file__).parent / "task_metadata.json"
_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"


def _normalize_instance_id(instance_id: str) -> str:
    """Normalize instance_id for Docker image naming: replace __ with _1776_ and lowercase."""
    return instance_id.replace("__", "_1776_").lower()


def _get_docker_image(instance_id: str) -> str:
    normalized = _normalize_instance_id(instance_id)
    return f"{_DOCKER_NAMESPACE}/sweb.eval.x86_64.{normalized}:{_IMAGE_TAG}"


def _build_task_metadata(tasks_data: list[dict[str, Any]]) -> dict[str, TaskMetadata]:
    """Build task_metadata from raw HuggingFace rows (no filtering, default runtime config)."""
    metadata: dict[str, TaskMetadata] = {}
    for t in tasks_data:
        instance_id = t["instance_id"]
        metadata[instance_id] = TaskMetadata(
            id=instance_id,
            abstract_description=t["problem_statement"][:200],
            recommended_max_steps=100,
            container_config=ContainerConfig(
                image=_get_docker_image(instance_id),
                cpu_cores=2.0,
                ram_gb=4.0,
                disk_gb=10.0,
            ),
            extra_info={
                "problem_statement": t["problem_statement"],
                "hints_text": t.get("hints_text", ""),
                "include_hints": False,
                "repo": t["repo"],
                "base_commit": t["base_commit"],
                "patch": t["patch"],
                "test_patch": t["test_patch"],
                "fail_to_pass": t["FAIL_TO_PASS"],
                "pass_to_pass": t["PASS_TO_PASS"],
                "difficulty": t.get("difficulty", "unknown"),
                "version": t.get("version", ""),
                "eval_timeout": 1800,
                "oracle_mode": False,
            },
        )
    return metadata


class SWEBenchVerifiedBenchmark(Benchmark):
    """SWE-bench Verified — 500 real-world GitHub issues with test-based validation."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="swebench-verified-cube",
        version="0.1.0",
        description="SWE-bench Verified — 500 real-world GitHub issues with test-based validation",
        num_tasks=500,
        tags=["swe", "github", "docker"],
    )
    # task_metadata: populated automatically at import time in Benchmark.__init_subclass__
    task_config_class: ClassVar[type[TaskConfig]] = SWEBenchVerifiedTaskConfig

    # User-configurable fields
    shuffle: bool = True
    shuffle_seed: int = 42
    max_tasks: int | None = None
    difficulty_filter: str | None = None
    repo_filter: str | None = None
    instance_ids: list[str] | None = None
    include_hints: bool = False
    oracle_mode: bool = False

    # ── Benchmark lifecycle ────────────────────────────────────────

    @classmethod
    def install(cls) -> None:
        """Download the SWE-bench Verified dataset and save task_metadata.json.

        Safe to call multiple times: skips if task_metadata.json already exists.
        Saves all 500 tasks with default runtime config (include_hints=False, oracle_mode=False).
        Use subset_from_glob / subset_from_list to filter at runtime.
        """
        if _TASK_METADATA_JSON.exists():
            logger.info("task_metadata.json already exists, skipping installation")
            return
        logger.info(f"Downloading {_DATASET_NAME} from HuggingFace...")
        ds = load_dataset(_DATASET_NAME, split="test")
        metadata = _build_task_metadata(list(ds))  # type: ignore[arg-type]
        _TASK_METADATA_JSON.write_text(json.dumps([tm.model_dump() for tm in metadata.values()], indent=2))
        cls.task_metadata = metadata
        logger.info(f"Saved {len(metadata)} tasks to {_TASK_METADATA_JSON}")

    @classmethod
    def uninstall(cls) -> None:
        """Remove task_metadata.json and the cached HuggingFace dataset."""
        if _TASK_METADATA_JSON.exists():
            _TASK_METADATA_JSON.unlink()
            cls.task_metadata = {}
            logger.info(f"Removed {_TASK_METADATA_JSON}")
        from datasets import config as ds_config

        cache_dir = Path(ds_config.HF_DATASETS_CACHE)
        dataset_dir = cache_dir / _DATASET_NAME.replace("/", "___").lower()
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            logger.info(f"Removed dataset cache at {dataset_dir}")

    def _setup(self) -> None:
        """Apply instance-level filters and runtime config to the pre-loaded task_metadata."""
        if "task_metadata" in self.__dict__:
            logger.info("SWE-bench Verified task_metadata already populated, skipping setup")
            return

        tasks = list(type(self).task_metadata.values())
        tasks = self._filter_tasks(tasks)

        # Apply per-instance runtime config if non-default
        if self.include_hints or self.oracle_mode:
            tasks = [
                t.model_copy(
                    update={
                        "extra_info": {
                            **t.extra_info,
                            "include_hints": self.include_hints,
                            "oracle_mode": self.oracle_mode,
                        }
                    }
                )
                for t in tasks
            ]

        metadata = {t.id: t for t in tasks}
        object.__setattr__(self, "task_metadata", metadata)
        type(self).task_metadata = metadata
        logger.info(f"SWE-bench Verified setup complete: {len(metadata)} tasks")

    def close(self) -> None:
        logger.info("SWE-bench Verified benchmark closed")

    # ── Private helpers ────────────────────────────────────────────

    def _filter_tasks(self, tasks: list[TaskMetadata]) -> list[TaskMetadata]:
        """Apply filtering, shuffling, and slicing to a list of TaskMetadata."""
        if self.instance_ids:
            id_set = set(self.instance_ids)
            tasks = [t for t in tasks if t.id in id_set]
        if self.difficulty_filter:
            tasks = [t for t in tasks if t.extra_info.get("difficulty", "").lower() == self.difficulty_filter.lower()]
        if self.repo_filter:
            tasks = [t for t in tasks if t.extra_info.get("repo", "").lower() == self.repo_filter.lower()]
        if self.shuffle:
            Random(self.shuffle_seed).shuffle(tasks)
        if self.max_tasks:
            tasks = tasks[: self.max_tasks]
        return tasks
