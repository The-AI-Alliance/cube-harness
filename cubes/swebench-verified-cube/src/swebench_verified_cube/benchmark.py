"""Benchmark for swebench-verified-cube — SWE-bench Verified with test-based validation."""

import json
import logging
import shutil
from collections.abc import Generator
from typing import Any, ClassVar

from pydantic import Field

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.infra_local import LocalInfraConfig
from cube.resource import InfraConfig
from cube.task import TaskConfig

from swebench_verified_cube.task import SWEBenchVerifiedTaskConfig, SWEBenchVerifiedTaskMetadata

logger = logging.getLogger(__name__)

_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"


def _build_execution_info(row: dict[str, Any]) -> dict[str, Any]:
    """Extract execution-only fields from a HuggingFace dataset row."""
    return {
        "problem_statement": row["problem_statement"],
        "hints_text": row.get("hints_text", ""),
        "patch": row["patch"],
        "test_patch": row["test_patch"],
        "fail_to_pass": row["FAIL_TO_PASS"],
        "pass_to_pass": row["PASS_TO_PASS"],
        "eval_timeout": 1800,
    }


class SWEBenchVerifiedBenchmark(Benchmark["SWEBenchVerifiedBenchmarkConfig"]):
    """Runtime pair for SWEBenchVerifiedBenchmarkConfig. Holds live infra state."""

    def _setup(self) -> None:
        """Publish the shared InfraConfig to runtime_context; containers are launched per-task."""
        infra = self.config.infra
        infra.cleanup_stale()
        self._runtime_context["infra"] = infra
        logger.info(
            f"SWEBenchVerifiedBenchmark ready with {len(self.config.task_metadata)} tasks"
            f" (infra={infra.fingerprint()})"
        )

    def close(self) -> None:
        logger.info("SWE-bench Verified benchmark closed")


class SWEBenchVerifiedBenchmarkConfig(BenchmarkConfig[SWEBenchVerifiedTaskMetadata]):
    """SWE-bench Verified — 500 real-world GitHub issues with test-based validation."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="swebench-verified-cube",
        version="0.1.0",
        description="SWE-bench Verified — 500 real-world GitHub issues with test-based validation",
        num_tasks=500,
        tags=["swe", "github", "docker"],
    )
    task_metadata: ClassVar[dict[str, SWEBenchVerifiedTaskMetadata]]  # type: ignore - populated automatically at import time in BenchmarkConfig.__init_subclass__
    task_config_class: ClassVar[type[TaskConfig]] = SWEBenchVerifiedTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = SWEBenchVerifiedBenchmark

    # User-configurable fields
    include_hints: bool = False
    oracle_mode: bool = False
    infra: InfraConfig = Field(default_factory=LocalInfraConfig)

    # ── Data lifecycle ─────────────────────────────────────────────

    @classmethod
    def install(cls) -> None:
        """Populate the per-task execution cache from HuggingFace.

        Downloads heavy fields (problem_statement, patch, test_patch, etc.) and writes
        one JSON file per task into task_execution_cache_dir(). Idempotent: skips if the
        cache directory already exists and is non-empty.
        """
        exec_cache_dir = cls.task_config_class.task_execution_cache_dir()
        if exec_cache_dir.exists() and any(exec_cache_dir.iterdir()):
            logger.info("Execution cache already populated, skipping installation")
            return
        exec_cache_dir.mkdir(parents=True, exist_ok=True)

        from datasets import load_dataset

        hf_cache = cls.cache_dir() / "huggingface_cache"
        logger.info(f"Downloading {_DATASET_NAME} from HuggingFace (cache: {hf_cache})...")
        ds = load_dataset(_DATASET_NAME, split="test", cache_dir=str(hf_cache))
        logger.info(f"  {len(ds)} tasks loaded")  # type: ignore[arg-type]

        n = 0
        for row in ds:
            iid = row["instance_id"]  # type: ignore
            (exec_cache_dir / f"{iid}.json").write_text(json.dumps(_build_execution_info(row)))  # type: ignore
            n += 1

        logger.info(f"Saved {n} execution cache files to {exec_cache_dir}")

    @classmethod
    def uninstall(cls) -> None:
        """Remove the per-task execution cache and the HuggingFace dataset cache."""
        exec_cache_dir = cls.task_config_class.task_execution_cache_dir()
        if exec_cache_dir.exists():
            shutil.rmtree(exec_cache_dir)
            logger.info(f"Removed execution cache at {exec_cache_dir}")

        hf_cache = cls.cache_dir() / "huggingface_cache"
        if hf_cache.exists():
            shutil.rmtree(hf_cache)
            logger.info(f"Removed HuggingFace dataset cache at {hf_cache}")

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        """Yield TaskConfigs with include_hints and oracle_mode forwarded from benchmark settings."""
        for tm in self.tasks().values():
            yield SWEBenchVerifiedTaskConfig(
                metadata=tm,
                tool_config=self.tool_config,
                seed=None,
                include_hints=self.include_hints,
                oracle_mode=self.oracle_mode,
            )
