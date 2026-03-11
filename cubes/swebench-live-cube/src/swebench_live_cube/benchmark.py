"""Benchmark for swebench-live-cube — SWE-bench Live with test-based validation."""

import logging
from collections.abc import Generator
from random import Random
from typing import Any, ClassVar

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.container import ContainerConfig
from cube.task import TaskConfig, TaskMetadata
from datasets import load_dataset

from swebench_live_cube.task import SWEBenchLiveTaskConfig

logger = logging.getLogger(__name__)

_DOCKER_NAMESPACE = "starryzhang"
_IMAGE_TAG = "latest"


def _normalize_instance_id(instance_id: str) -> str:
    """Normalize instance_id for Docker image naming: replace __ with _1776_ and lowercase."""
    return instance_id.replace("__", "_1776_").lower()


class SWEBenchLiveBenchmark(Benchmark):
    """SWE-bench Live — continuously updated GitHub issue resolution benchmark."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="swebench-live-cube",
        version="0.1.0",
        description="SWE-bench Live — continuously updated, contamination-resistant GitHub issue resolution",
        tags=["swe", "github", "docker", "live"],
    )

    task_metadata: ClassVar[dict[str, TaskMetadata]] = {}
    task_config_class: ClassVar[type[TaskConfig]] = SWEBenchLiveTaskConfig

    # User-configurable fields
    dataset_name: str = "SWE-bench-Live/SWE-bench-Live"
    split: str = "lite"
    shuffle: bool = True
    shuffle_seed: int = 42
    max_tasks: int | None = None
    repo_filter: str | None = None
    instance_ids: list[str] | None = None
    include_hints: bool = False

    # ── Benchmark lifecycle ────────────────────────────────────────

    def _setup(self) -> None:
        """Load dataset from HuggingFace, apply filters, and populate task_metadata."""
        ds = load_dataset(self.dataset_name, split=self.split)
        tasks_data = self._filter_tasks(list(ds))

        metadata: dict[str, TaskMetadata] = {}
        for t in tasks_data:
            instance_id = t["instance_id"]
            docker_image = self._get_docker_image(instance_id)

            # FAIL_TO_PASS / PASS_TO_PASS are already lists in SWE-bench Live
            fail_to_pass = t["FAIL_TO_PASS"] if isinstance(t["FAIL_TO_PASS"], list) else []
            pass_to_pass = t["PASS_TO_PASS"] if isinstance(t["PASS_TO_PASS"], list) else []

            metadata[instance_id] = TaskMetadata(
                id=instance_id,
                abstract_description=t["problem_statement"][:200],
                recommended_max_steps=100,
                container_config=ContainerConfig(
                    image=docker_image,
                    cpu_cores=2.0,
                    ram_gb=8.0,
                    disk_gb=10.0,
                ),
                extra_info={
                    "problem_statement": t["problem_statement"],
                    "hints_text": t.get("hints_text", ""),
                    "include_hints": self.include_hints,
                    "repo": t["repo"],
                    "base_commit": t["base_commit"],
                    "patch": t["patch"],
                    "test_patch": t["test_patch"],
                    "fail_to_pass": fail_to_pass,
                    "pass_to_pass": pass_to_pass,
                    "test_cmds": t.get("test_cmds", []),
                    "log_parser": t.get("log_parser", "pytest"),
                    "eval_timeout": 1800,
                },
            )

        SWEBenchLiveBenchmark.task_metadata = metadata
        logger.info(f"SWE-bench Live setup complete: {len(metadata)} tasks (split={self.split})")

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        """Yield self-contained TaskConfigs with metadata snapshots for Ray workers."""
        for tm in self.task_metadata.values():
            yield SWEBenchLiveTaskConfig(
                task_id=tm.id,
                tool_config=self.default_tool_config,
                task_metadata_snapshot=tm,
            )

    def close(self) -> None:
        pass

    def install(self) -> None:
        """Pre-download the SWE-bench Live dataset from HuggingFace."""
        logger.info(f"Downloading {self.dataset_name} (split={self.split}) from HuggingFace...")
        load_dataset(self.dataset_name, split=self.split)
        logger.info("Dataset download complete")

    # ── Private helpers ────────────────────────────────────────────

    def _get_docker_image(self, instance_id: str) -> str:
        """Get the Docker image name for a given instance."""
        normalized = _normalize_instance_id(instance_id)
        return f"{_DOCKER_NAMESPACE}/sweb.eval.x86_64.{normalized}:{_IMAGE_TAG}"

    def _filter_tasks(self, tasks_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply filtering, shuffling, and slicing to raw task data."""
        if self.instance_ids:
            id_set = set(self.instance_ids)
            tasks_data = [t for t in tasks_data if t["instance_id"] in id_set]
        if self.repo_filter:
            tasks_data = [t for t in tasks_data if t.get("repo", "").lower() == self.repo_filter.lower()]
        if self.shuffle:
            Random(self.shuffle_seed).shuffle(tasks_data)
        if self.max_tasks:
            tasks_data = tasks_data[: self.max_tasks]
        return tasks_data
