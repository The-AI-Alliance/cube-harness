"""
OSWorldBenchmark and OSWorldTaskConfig — CUBE benchmark for the OSWorld desktop-automation suite.

Entry point:
    bench = OSWorldBenchmark(default_tool_config=ComputerConfig())
    bench.setup()
    for task_config in bench.get_task_configs():
        task = task_config.make()
        obs, info = task.reset()
        ...
        task.close()

Filter by domain or other metadata field after setup():
    chrome_bench = bench.subset_from_glob("domain", "chrome")
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Generator
from copy import deepcopy
from typing import cast
from dotenv import load_dotenv
from pathlib import Path
from typing import ClassVar

from cube import LocalInfraConfig
from pydantic import Field

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.container import ContainerBackend
from cube.resource import InfraConfig, ResourceConfig
from cube.task import TaskConfig

from osworld_cube._paths import OSWORLD_BASE_DIR, OSWORLD_REPO_DIR, OSWORLD_VM_DIR
from osworld_cube.computer import ComputerConfig
from osworld_cube.task import OSWORLD_UBUNTU_RESOURCE, OSWorldTask, OSWorldTaskMetadata

logger = logging.getLogger(__name__)


# Pinned OSWorld commit for reproducibility
OSWORLD_COMMIT = "e695a10"


# ---------------------------------------------------------------------------
# .env helper
# ---------------------------------------------------------------------------


def ensure_proxy_config_in_env(env_path: Path = Path(".env")) -> None:
    """Append PROXY_CONFIG_FILE to .env if it is not already defined there.

    The value mirrors the default set in computer.py so that desktop_env
    resolves the path correctly regardless of the current working directory.
    """
    key = "PROXY_CONFIG_FILE"
    value = str(OSWORLD_REPO_DIR / "evaluation_examples" / "settings" / "proxy" / "dataimpulse.json")

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
                logger.debug(f"{key} already present in {env_path}, skipping.")
                return

    with env_path.open("a") as f:
        f.write(f"\n{key}={value}\n")
    logger.info(f"Appended {key} to {env_path}")


_TASK_METADATA_JSON = Path(__file__).with_name("task_metadata.json")
_TASK_METADATA = cast(dict[str, OSWorldTaskMetadata], Benchmark.task_metadata_from_json(_TASK_METADATA_JSON))


class OSWorldTaskConfig(TaskConfig):
    """
    Serialisable config for a single OSWorld task.

    Fields:
        task_id:     inherited from TaskConfig
        tool_config: inherited from TaskConfig
        seed:        inherited (ignored for OSWorld — tasks are deterministic)
        metadata:    OSWorldTaskMetadata for this task. Stored directly on the config so
                     the config is self-contained and safe to send to Ray workers.
        infra:       InfraConfig to use for this task.
    """

    metadata: OSWorldTaskMetadata
    infra: InfraConfig | None = None

    def make(
        self,
        runtime_context: dict | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> OSWorldTask:
        """Instantiate OSWorldTask from this config."""
        if self.tool_config is None:
            raise ValueError(
                f"OSWorldTaskConfig for task '{self.task_id}' has no tool_config. "
                "Pass default_tool_config=ComputerConfig(...) to OSWorldBenchmark."
            )
        extra = dict(self.metadata.extra_info)
        if "config" not in extra or "evaluator" not in extra:
            exec_info = OSWorldBenchmark.load_task_execution_info(self.task_id)
            extra.update(exec_info)
        hydrated_metadata = self.metadata.model_copy(update={"extra_info": extra})
        return OSWorldTask(
            metadata=hydrated_metadata,
            tool_config=self.tool_config,
            infra=self.infra,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )


# ---------------------------------------------------------------------------
# OSWorldBenchmark
# ---------------------------------------------------------------------------


class OSWorldBenchmark(Benchmark):
    """
    CUBE benchmark wrapping the OSWorld desktop-automation evaluation suite.

    Reference: https://github.com/xlang-ai/OSWorld

    Class-level attributes (required by cube.benchmark.Benchmark):
        benchmark_metadata:  ClassVar[BenchmarkMetadata]
        task_metadata:       ClassVar[dict[str, OSWorldTaskMetadata]]
        task_config_class:   type[TaskConfig] = OSWorldTaskConfig

    Constructor params (set by benchmark users):
        default_tool_config:  ComputerConfig  — how to connect to the VM (action_space selects variant)
        use_som:              bool            — Set-of-Marks mode for all tasks

    To filter by domain or any other metadata field, call subset_from_glob() after setup():
        bench.setup()
        chrome_bench = bench.subset_from_glob("domain", "chrome")
    """

    # ------------------------------------------------------------------
    # Required class variables
    # ------------------------------------------------------------------

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="osworld-cube",
        version="1.0.0",
        description=("OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments"),
        authors=["Tianbao Xie et al."],
        license="CC-BY-4.0",
        requirements={
            "vm": "Ubuntu 22.04 (docker or vmware)",
            "ram_gb": 8,
            "disk_gb": 40,
        },
        num_tasks=len(_TASK_METADATA),
        tags=["desktop", "gui", "multimodal"],
        named_subsets={
            "test_all": ("test_sets", "*test_all*"),
            "test_small": ("test_sets", "*test_small*"),
            "test_nogdrive": ("test_sets", "*test_nogdrive*"),
            "test_infeasible": ("test_sets", "*test_infeasible*"),
        },
    )

    task_metadata: ClassVar[dict[str, OSWorldTaskMetadata]] = _TASK_METADATA

    task_config_class: ClassVar[type[TaskConfig]] = OSWorldTaskConfig

    # ------------------------------------------------------------------
    # Instance fields
    # ------------------------------------------------------------------
    default_tool_config: ComputerConfig = ComputerConfig()

    use_som: bool = False
    """Enable Set-of-Marks annotation for all tasks in this benchmark run."""

    infra: InfraConfig | None = Field(default_factory=LocalInfraConfig)
    """InfraConfig (AWSInfraConfig, AzureInfraConfig, LocalInfraConfig).
    Each task gets a fresh VM launched from the provisioned image."""

    resources: list[ResourceConfig] = [OSWORLD_UBUNTU_RESOURCE]
    """VM image required to run OSWorld tasks (declared for the harness resource lifecycle)."""

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        """Yield OSWorldTaskConfig objects, injecting infra."""
        for tm in self.task_metadata.values():
            yield OSWorldTaskConfig(
                task_id=tm.id,
                metadata=tm,
                tool_config=self.default_tool_config,
                seed=None,
                infra=self.infra,
            )

    # ------------------------------------------------------------------
    # _setup()
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """
        Prepare benchmark for task spawning.

        Steps:
          1. Ensure install() has populated execution info
          2. Prepare benchmark runtime context
        """
        self.install()

        logger.info(f"Setting up OSWorldBenchmark (provider={self._get_provider()})")

        # OSWorld manages its own VM lifecycle via desktop_env — no shared runtime
        # infrastructure is needed. Populate _runtime_context to suppress the
        # Benchmark.setup() warning that fires when it is left empty.
        self._runtime_context = {"osworld": True}

        logger.info(f"OSWorldBenchmark ready with {len(self.task_metadata)} tasks")

    def close(self) -> None:
        """
        Clean up benchmark resources.

        VM teardown is handled per-task by Computer.close() / OSWorldTask.close().
        No global VM resources to release here.
        """
        logger.info("Closing OSWorldBenchmark — no global resources to release")

    def _build_task_execution_info_from_repo(self) -> dict[str, dict]:
        """
        Build heavy per-task execution info from the OSWorld repo.
        """
        eval_examples_dir = OSWORLD_REPO_DIR / "evaluation_examples"
        exec_info_by_id: dict[str, dict] = {}

        for test_set_file in eval_examples_dir.glob("test_*.json"):
            with open(test_set_file) as f:
                tasks_by_domain: dict[str, list[str]] = json.load(f)
            for domain_name, task_ids in tasks_by_domain.items():
                for task_id in task_ids:
                    task_file = eval_examples_dir / "examples" / domain_name / f"{task_id}.json"
                    if not task_file.exists():
                        logger.warning("Task file not found: %s", task_file)
                        continue
                    try:
                        with open(task_file) as f:
                            td = json.load(f)
                        exec_info_by_id[task_id] = {
                            "config": self._fix_config_paths(td.get("config", [])),
                            "evaluator": td.get("evaluator", {}),
                        }
                    except Exception as e:
                        logger.error("Failed to load task %s: %s", task_id, e)

        logger.info("Built %d task execution info entries from OSWorld repo", len(exec_info_by_id))
        return exec_info_by_id

    @staticmethod
    def _fix_config_paths(config: list[dict]) -> list[dict]:
        """
        Prepend OSWorld repo path to settings_file paths in config items.

        Keeps relative paths working regardless of CWD.
        """
        updated = deepcopy(config)
        for config_item in updated:
            params = config_item.get("parameters", {})
            if "settings_file" in params:
                params["settings_file"] = str(OSWORLD_REPO_DIR / params["settings_file"])
        return updated

    @classmethod
    def cache_dir(cls) -> Path:
        """Return the benchmark cache directory."""
        return OSWORLD_BASE_DIR

    @classmethod
    def task_execution_cache_dir(cls) -> Path:
        """Return the directory containing per-task execution info."""
        return cls.cache_dir() / "tasks_execution_info"

    @classmethod
    def load_task_execution_info(cls, task_id: str) -> dict:
        """Load per-task execution info for the given task id."""
        cache_file = cls.task_execution_cache_dir() / f"{task_id}.json"
        if not cache_file.exists():
            raise RuntimeError(
                f"No execution data for {task_id!r}. Run `OSWorldBenchmark.install()` to populate the execution cache."
            )
        return json.loads(cache_file.read_text())

    def _clone_osworld_repo(self) -> None:
        """Clone and pin the OSWorld repository to OSWORLD_COMMIT."""
        OSWORLD_BASE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/xlang-ai/OSWorld", str(OSWORLD_REPO_DIR)],
            check=True,
        )
        subprocess.run(
            ["git", "checkout", OSWORLD_COMMIT],
            cwd=str(OSWORLD_REPO_DIR),
            check=True,
        )
        OSWORLD_VM_DIR.mkdir(parents=True, exist_ok=True)

    def _get_provider(self) -> str:
        """Return provider name derived from infra type."""
        if self.infra is None:
            return "none"
        return type(self.infra).__name__

    # ------------------------------------------------------------------
    # install() — available for manual invocation; called from _setup()
    # ------------------------------------------------------------------

    def install(self) -> None:
        """
        Clone OSWorld repo and cache per-task execution info.

        Also sets PROXY_CONFIG_FILE env var to the correct path inside
        the cloned repo so desktop_env finds it at import time.
        """
        logger.info("Installing OSWorld benchmark...")
        if not OSWORLD_REPO_DIR.exists():
            self._clone_osworld_repo()
            logger.info(f"OSWorld repo cloned to {OSWORLD_REPO_DIR}")
        else:
            logger.info(f"OSWorld repo already present at {OSWORLD_REPO_DIR}")
        if OSWORLD_REPO_DIR.exists():
            ensure_proxy_config_in_env()
            load_dotenv()  # Load the .env file
            logger.info(f"Set PROXY_CONFIG_FILE={os.environ.get('PROXY_CONFIG_FILE', 'not set')}")
        else:
            logger.info("Skipping PROXY_CONFIG_FILE setup because the OSWorld repo is not present.")

        exec_info_by_id = self._build_task_execution_info_from_repo()

        exec_cache_dir = type(self).task_execution_cache_dir()
        exec_cache_dir.mkdir(parents=True, exist_ok=True)
        for task_id, exec_info in exec_info_by_id.items():
            cache_file = exec_cache_dir / f"{task_id}.json"
            cache_file.write_text(json.dumps(exec_info, indent=2))

        if isinstance(self.infra, LocalInfraConfig):
            for resource in self.resources:
                if self.infra.provision_status(resource) == "ready":
                    logger.info("Local resource %s already provisioned", resource.name)
                    continue
                logger.info("Provisioning local resource %s...", resource.name)
                self.infra.provision(resource)

        logger.info(
            "OSWorld install complete: %d execution cache files",
            len(exec_info_by_id),
        )
