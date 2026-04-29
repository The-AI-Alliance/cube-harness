"""Benchmark for swebench-verified-cube — SWE-bench Verified with test-based validation."""

import json
import logging
import shutil
import subprocess
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.infra_local import LocalInfraConfig
from cube.provision_store import ProvisionStore
from cube.resource import DockerServiceConfig, InfraConfig
from cube.task import TaskConfig

from swebench_verified_cube.task import SWEBenchVerifiedTaskConfig, SWEBenchVerifiedTaskMetadata

logger = logging.getLogger(__name__)


def _pull_with_retry(image: str, max_attempts: int = 5, base_delay: float = 30.0) -> None:
    """Pull a Docker image with exponential backoff on rate-limit (429) responses."""
    for attempt in range(max_attempts):
        result = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
        if result.returncode == 0:
            return
        combined = result.stdout + result.stderr
        if any(kw in combined for kw in ("toomanyrequests", "Too Many Requests", "rate limit")):
            delay = base_delay * (2**attempt)
            logger.warning(
                "Rate-limited pulling %r (attempt %d/%d), retrying in %.0fs…",
                image,
                attempt + 1,
                max_attempts,
                delay,
            )
            time.sleep(delay)
            continue
        raise subprocess.CalledProcessError(
            result.returncode, ["docker", "pull", image], result.stdout, result.stderr
        )
    raise RuntimeError(f"Rate limit not resolved after {max_attempts} attempts pulling {image!r}")


def _log_docker_auth_status() -> None:
    """Warn if Docker Hub credentials are absent (anonymous pulls are rate-limited)."""
    config_path = Path.home() / ".docker" / "config.json"
    try:
        config = json.loads(config_path.read_text())
        auths = config.get("auths", {})
        hub_keys = [k for k in auths if "docker.io" in k or "index.docker.io" in k]
        if hub_keys:
            logger.info(
                "prefetch_images: Docker Hub credentials found — higher pull rate applies. "
                "Run `docker login` again if you hit limits."
            )
        else:
            logger.warning(
                "prefetch_images: No Docker Hub credentials found — anonymous limit is 100 pulls/6h. "
                "Run `docker login` (or `podman login docker.io`) to authenticate."
            )
    except Exception:
        logger.warning(
            "prefetch_images: Could not read ~/.docker/config.json. "
            "Run `docker login` to authenticate and raise your pull rate limit."
        )

_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"


def _build_execution_info(row: dict[str, Any]) -> dict[str, Any]:
    """Extract execution-only fields from a HuggingFace dataset row.

    These fields are only needed when a task runs; they are never loaded at
    import time. Stored in the per-task execution cache by install().
    """
    return {
        "problem_statement": row["problem_statement"],
        "hints_text": row.get("hints_text", ""),
        "patch": row["patch"],
        "test_patch": row["test_patch"],
        "fail_to_pass": row["FAIL_TO_PASS"],
        "pass_to_pass": row["PASS_TO_PASS"],
        "eval_timeout": 1800,
    }


class SWEBenchVerifiedBenchmark(Benchmark):
    """SWE-bench Verified — 500 real-world GitHub issues with test-based validation."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="swebench-verified-cube",
        version="0.1.0",
        description="SWE-bench Verified — 500 real-world GitHub issues with test-based validation",
        num_tasks=500,
        tags=["swe", "github", "docker"],
    )
    task_metadata: ClassVar[dict[str, SWEBenchVerifiedTaskMetadata]]  # type: ignore - populated automatically at import time in Benchmark.__init_subclass__
    task_config_class: ClassVar[type[TaskConfig]] = SWEBenchVerifiedTaskConfig

    # User-configurable fields
    include_hints: bool = False
    oracle_mode: bool = False
    infra: InfraConfig = Field(default_factory=LocalInfraConfig)
    prefetch_images: bool = True
    """Pre-pull all task images sequentially in _setup() before Ray workers start.

    Prevents Docker Hub rate-limit errors under parallel runs.
    Workers find images already in ProvisionStore and skip Docker Hub entirely.
    Set False if images are already present locally or managed by your infra.
    """

    # ── Benchmark lifecycle ────────────────────────────────────────

    @classmethod
    def install(cls) -> None:
        """Populate the per-task execution cache from HuggingFace.

        Downloads heavy fields (problem_statement, patch, test_patch, etc.) and writes
        one JSON file per task into task_execution_cache_dir(). Idempotent: skips if the
        cache directory already exists and is non-empty. If the HuggingFace data has not
        been downloaded yet, it is fetched into cache_dir()/huggingface_cache/.

        The shipped task_metadata.json is a package resource and is not modified here.
        To regenerate task_metadata.json (developer use only), run:
            scripts/generate_task_metadata.py
        """
        exec_cache_dir = cls.task_execution_cache_dir()
        if exec_cache_dir.exists() and any(exec_cache_dir.iterdir()):
            logger.info("Execution cache already populated, skipping installation")
            return
        exec_cache_dir.mkdir(parents=True, exist_ok=True)

        # Download into our own cache folder (not the default ~/.cache/huggingface).
        # load_dataset is idempotent: if the data is already cached there, no download occurs.
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
        """Remove the per-task execution cache and the HuggingFace dataset cache.

        The shipped task_metadata.json is not removed.
        """
        exec_cache_dir = cls.task_execution_cache_dir()
        if exec_cache_dir.exists():
            shutil.rmtree(exec_cache_dir)
            logger.info(f"Removed execution cache at {exec_cache_dir}")

        hf_cache = cls.cache_dir() / "huggingface_cache"
        if hf_cache.exists():
            shutil.rmtree(hf_cache)
            logger.info(f"Removed HuggingFace dataset cache at {hf_cache}")

    def _setup(self) -> None:
        """Publish the shared InfraConfig to runtime_context; containers are launched per-task."""
        self.infra.cleanup_stale()
        self._runtime_context["infra"] = self.infra
        if self.prefetch_images:
            self._prefetch_images()
        logger.info(
            f"SWEBenchVerifiedBenchmark ready with {len(self.task_metadata)} tasks (infra={self.infra.fingerprint()})"
        )

    def _prefetch_images(self) -> None:
        """Pre-pull all task images sequentially before Ray workers start.

        Only runs for LocalInfraConfig — cloud infras manage image pulls themselves.
        Idempotent: images already in ProvisionStore are skipped.
        """
        if not isinstance(self.infra, LocalInfraConfig):
            return

        store = ProvisionStore()
        _log_docker_auth_status()

        # Collect (task_id, image) pairs not yet in ProvisionStore; deduplicate by image.
        to_provision: list[tuple[str, str]] = []
        seen_images: set[str] = set()
        for tm in self.task_metadata.values():
            cc = tm.container_config
            if cc is None or cc.image in seen_images:
                continue
            seen_images.add(cc.image)
            resource = DockerServiceConfig(name=tm.id, scope="task", docker_images=[cc.image])
            if store.get(resource, self.infra) is None:
                to_provision.append((tm.id, cc.image))

        if not to_provision:
            logger.info("prefetch_images: all %d images already provisioned — skipping", len(seen_images))
            return

        logger.info(
            "prefetch_images: pulling %d/%d images sequentially (this may take several minutes)…",
            len(to_provision),
            len(seen_images),
        )
        succeeded = 0
        for i, (task_id, image) in enumerate(to_provision, 1):
            logger.info("  [%d/%d] %s", i, len(to_provision), image)
            try:
                _pull_with_retry(image)
            except Exception as exc:
                logger.warning("  failed to pull %r: %s — workers will attempt pull themselves", image, exc)
                continue
            resource = DockerServiceConfig(name=task_id, scope="task", docker_images=[image])
            store.put(resource, self.infra, {"provisioned": True})
            succeeded += 1

        logger.info("prefetch_images: %d/%d images ready", succeeded, len(to_provision))

    def close(self) -> None:
        logger.info("SWE-bench Verified benchmark closed")

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        """Yield TaskConfigs with include_hints and oracle_mode forwarded from benchmark settings."""
        for tm in self.task_metadata.values():
            yield SWEBenchVerifiedTaskConfig(
                task_id=tm.id,
                tool_config=self.default_tool_config,
                seed=None,
                include_hints=self.include_hints,
                oracle_mode=self.oracle_mode,
            )
