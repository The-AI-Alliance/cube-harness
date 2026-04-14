"""Benchmark for swebench-live-cube — SWE-bench Live with test-based validation."""

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

from swebench_live_cube.task import SWEBenchLiveTaskConfig

logger = logging.getLogger(__name__)

_DOCKER_NAMESPACE = "starryzhang"
_IMAGE_TAG = "latest"
_DATASET_NAME = "SWE-bench-Live/SWE-bench-Live"
# Priority order for conflict resolution: earlier = higher priority.
# When the same instance_id appears in multiple splits with different data,
# the split with the highest priority wins.
_SPLIT_PRIORITY = ["verified", "full", "test", "lite"]
_TASK_METADATA_JSON = Path(__file__).parent / "task_metadata.json"


def _normalize_instance_id(instance_id: str) -> str:
    """Normalize instance_id for Docker image naming: replace __ with _1776_ and lowercase."""
    return instance_id.replace("__", "_1776_").lower()


def _get_docker_image(instance_id: str) -> str:
    normalized = _normalize_instance_id(instance_id)
    return f"{_DOCKER_NAMESPACE}/sweb.eval.x86_64.{normalized}:{_IMAGE_TAG}"


def _build_task_metadata(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, TaskMetadata]:
    """Build task_metadata from all splits, deduplicating by instance_id.

    When the same instance_id appears in multiple splits with different data,
    the split with the highest priority in _SPLIT_PRIORITY wins.
    Each task records which splits it belongs to in extra_info["splits"].
    """
    # First pass: collect all rows per instance_id across splits
    all_rows: dict[str, dict[str, dict[str, Any]]] = {}  # iid -> {split: row}
    for split in _SPLIT_PRIORITY:
        for row in rows_by_split.get(split, []):
            iid = row["instance_id"]
            all_rows.setdefault(iid, {})[split] = row

    # Second pass: pick the highest-priority row and report conflicts
    n_conflicts = 0
    chosen_rows: dict[str, tuple[str, dict[str, Any]]] = {}  # iid -> (winning_split, row)
    for iid, split_rows in all_rows.items():
        splits_present = [s for s in _SPLIT_PRIORITY if s in split_rows]
        winning_split = splits_present[0]  # highest priority
        winning_row = split_rows[winning_split]

        if len(splits_present) > 1:
            differing = [s for s in splits_present[1:] if split_rows[s] != winning_row]
            if differing:
                n_conflicts += 1
                diff_lines = []
                for other_split in differing:
                    other_row = split_rows[other_split]
                    changed_fields = {k for k in winning_row if winning_row[k] != other_row.get(k)}
                    for field in sorted(changed_fields):
                        diff_lines.append(
                            f"  field={field!r}:\n"
                            f"    {winning_split}: {repr(winning_row[field])[:200]}\n"
                            f"    {other_split}: {repr(other_row[field])[:200]}"
                        )
                logger.warning(
                    f"Conflict for {iid!r}: data differs between {differing} and {winning_split!r}:\n"
                    + "\n".join(diff_lines)
                    + f"\n  -> Using {winning_split!r}."
                )

        chosen_rows[iid] = (winning_split, winning_row)

    if n_conflicts:
        logger.warning(
            f"{n_conflicts} task(s) had conflicting data across splits. Split priority used: {_SPLIT_PRIORITY}."
        )

    metadata: dict[str, TaskMetadata] = {}
    for iid, (_, t) in chosen_rows.items():
        splits_present = [s for s in _SPLIT_PRIORITY if s in all_rows[iid]]
        fail_to_pass = t["FAIL_TO_PASS"] if isinstance(t["FAIL_TO_PASS"], list) else []
        pass_to_pass = t["PASS_TO_PASS"] if isinstance(t["PASS_TO_PASS"], list) else []
        metadata[iid] = TaskMetadata(
            id=iid,
            abstract_description=t["problem_statement"][:200],
            recommended_max_steps=100,
            container_config=ContainerConfig(
                image=_get_docker_image(iid),
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
                "fail_to_pass": fail_to_pass,
                "pass_to_pass": pass_to_pass,
                "test_cmds": t.get("test_cmds", []),
                "log_parser": t.get("log_parser", "pytest"),
                "eval_timeout": 1800,
                "oracle_mode": False,
                "splits": splits_present,
            },
        )
    return metadata


class SWEBenchLiveBenchmark(Benchmark):
    """SWE-bench Live — continuously updated GitHub issue resolution benchmark."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="swebench-live-cube",
        version="0.1.0",
        description=(
            "SWE-bench Live — continuously updated, contamination-resistant GitHub issue resolution. "
            "By default the benchmark contains all tasks across all splits. "
            "Use bench.named_subset('lite'), bench.named_subset('verified'), etc. to get a specific split.\n"
            "\n"
            "CUBE DEVELOPER NOTES:\n"
            "---------------------\n"
            "task_metadata.json contains all tasks from all splits (test, lite, verified, full), "
            "deduplicated by instance_id with split priority: verified > full > test > lite. "
            "Each task stores which splits it belongs to in extra_info['splits']."
        ),
        tags=["swe", "github", "docker", "live"],
        num_tasks=1895,  # total unique tasks across all splits as of 2026-04-02
        # Splits overlap heavily: full(1887) ⊇ verified(499) ⊇ lite(300); test(1000) adds 8 unique tasks not in full.
        named_subsets={
            "test": ("extra_info.splits", "*'test'*"),
            "lite": ("extra_info.splits", "*'lite'*"),
            "verified": ("extra_info.splits", "*'verified'*"),
            "full": ("extra_info.splits", "*'full'*"),
        },
    )
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {}
    task_config_class: ClassVar[type[TaskConfig]] = SWEBenchLiveTaskConfig

    # User-configurable fields
    shuffle: bool = True
    shuffle_seed: int = 42
    max_tasks: int | None = None
    repo_filter: str | None = None
    instance_ids: list[str] | None = None
    include_hints: bool = False
    oracle_mode: bool = False

    # ── Benchmark lifecycle ────────────────────────────────────────

    @classmethod
    def install(cls) -> None:
        """Download all SWE-bench Live splits and save task_metadata.json.

        Downloads test, lite, verified, and full splits from HuggingFace, merges
        them by instance_id (deduplicating with priority: verified > full > test > lite),
        and stores which splits each task belongs to in extra_info["splits"].

        Safe to call multiple times: skips if task_metadata.json already exists.
        Use subset_from_glob to filter by split at runtime, e.g.:
            bench.subset_from_glob("extra_info", '*"lite"*')
        """
        if _TASK_METADATA_JSON.exists():
            logger.info("task_metadata.json already exists, skipping installation")
            return
        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        for split in _SPLIT_PRIORITY:
            logger.info(f"Downloading {_DATASET_NAME} split={split!r} from HuggingFace...")
            ds = load_dataset(_DATASET_NAME, split=split)
            rows_by_split[split] = list(ds)  # type: ignore[arg-type]
            logger.info(f"  {len(rows_by_split[split])} tasks in split={split!r}")
        metadata = _build_task_metadata(rows_by_split)
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
        dataset_dir = cache_dir / _DATASET_NAME.replace("/", "___")
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            logger.info(f"Removed dataset cache at {dataset_dir}")

    def _setup(self) -> None:
        """Apply instance-level filters and runtime config to the pre-loaded task_metadata."""
        if "task_metadata" in self.__dict__:
            logger.info("SWE-bench Live task_metadata already populated, skipping setup")
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
        logger.info(f"SWE-bench Live setup complete: {len(metadata)} tasks")

    def close(self) -> None:
        logger.info("SWE-bench Live benchmark closed")

    # ── Private helpers ────────────────────────────────────────────

    def _filter_tasks(self, tasks: list[TaskMetadata]) -> list[TaskMetadata]:
        """Apply filtering, shuffling, and slicing to a list of TaskMetadata."""
        if self.instance_ids:
            id_set = set(self.instance_ids)
            tasks = [t for t in tasks if t.id in id_set]
        if self.repo_filter:
            tasks = [t for t in tasks if t.extra_info.get("repo", "").lower() == self.repo_filter.lower()]
        if self.shuffle:
            Random(self.shuffle_seed).shuffle(tasks)
        if self.max_tasks:
            tasks = tasks[: self.max_tasks]
        return tasks
