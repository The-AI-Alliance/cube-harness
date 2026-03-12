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
    chrome_bench = bench.subset_from_glob("extra_info.domain", "chrome")
"""

from __future__ import annotations

import enum
import json
import logging
import os
import subprocess
from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from typing import ClassVar

from pydantic import model_validator

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.container import ContainerBackend
from cube.task import TaskConfig, TaskMetadata

from osworld_cube.computer import ComputerConfig, _CUBE_CACHE_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths — rooted under CUBE_CACHE_DIR (default ~/.agentlab2)
# ---------------------------------------------------------------------------

OSWORLD_BASE_DIR = _CUBE_CACHE_ROOT
OSWORLD_REPO_DIR = OSWORLD_BASE_DIR / "OSWorld"
OSWORLD_VM_DIR = OSWORLD_BASE_DIR / "vm_data"
OSWORLD_CACHE_DIR = OSWORLD_BASE_DIR / "cache"

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
    value = str(_CUBE_CACHE_ROOT / "OSWorld" / "evaluation_examples" / "settings" / "proxy" / "dataimpulse.json")

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
                logger.debug(f"{key} already present in {env_path}, skipping.")
                return

    with env_path.open("a") as f:
        f.write(f"\n{key}={value}\n")
    logger.info(f"Appended {key} to {env_path}")


# ---------------------------------------------------------------------------
# OSWorldTestSet
# ---------------------------------------------------------------------------


class OSWorldTestSet(str, enum.Enum):
    """Valid test-set index files shipped with the OSWorld repo."""

    TEST_ALL = "test_all.json"
    TEST_INFEASIBLE = "test_infeasible.json"
    TEST_NOGDRIVE = "test_nogdrive.json"
    TEST_SMALL = "test_small.json"


# ---------------------------------------------------------------------------
# OSWorldTaskConfig
# ---------------------------------------------------------------------------


class OSWorldTaskConfig(TaskConfig):
    """
    Serialisable config for a single OSWorld task.

    Carries a copy of the full TaskMetadata so that make() can instantiate
    OSWorldTask without importing OSWorldBenchmark (avoids circular import).

    Fields:
        task_id:    inherited from TaskConfig
        tool_config: inherited from TaskConfig
        seed:       inherited (ignored for OSWorld — tasks are deterministic)
        metadata:   full TaskMetadata copy injected by OSWorldBenchmark.get_task_configs()
    """

    metadata: TaskMetadata | None = None

    def make(
        self,
        runtime_context: dict | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> "OSWorldTask":  # noqa: F821
        """Instantiate OSWorldTask from this config."""
        from osworld_cube.task import OSWorldTask

        if self.metadata is None:
            raise ValueError(
                f"OSWorldTaskConfig for task '{self.task_id}' has no metadata. "
                "Ensure get_task_configs() was called on an OSWorldBenchmark instance "
                "that has been setup() first."
            )
        if self.tool_config is None:
            raise ValueError(
                f"OSWorldTaskConfig for task '{self.task_id}' has no tool_config. "
                "Pass default_tool_config=ComputerConfig(...) to OSWorldBenchmark."
            )
        return OSWorldTask(
            metadata=self.metadata,
            tool_config=self.tool_config,
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
        task_metadata:       ClassVar[dict[str, TaskMetadata]]  (placeholder {}; populated in _setup())
        task_config_class:   type[TaskConfig] = OSWorldTaskConfig

    Constructor params (set by benchmark users):
        default_tool_config:  ComputerConfig  — how to connect to the VM (action_space selects variant)
        tasks_file:           str | None         — flat JSON task file; mutually exclusive with test_set_name
        test_set_name:        OSWorldTestSet     — which test-set index file to load (default: TEST_ALL)
        use_som:              bool               — Set-of-Marks mode for all tasks

    To filter by domain or any other metadata field, call subset_from_glob() after setup():
        bench.setup()
        chrome_bench = bench.subset_from_glob("extra_info.domain", "chrome")
    """

    # ------------------------------------------------------------------
    # Required class variables
    # ------------------------------------------------------------------

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="osworld",
        version="1.0.0",
        description=("OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments"),
        authors=["Tianbao Xie et al."],
        license="CC-BY-4.0",
        requirements={
            "vm": "Ubuntu 22.04 (docker or vmware)",
            "ram_gb": 8,
            "disk_gb": 40,
        },
        num_tasks=369,
        tags=["desktop", "gui", "multimodal"],
    )

    # Placeholder: populated per-instance in _setup() via object.__setattr__
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {}

    task_config_class: ClassVar[type[TaskConfig]] = OSWorldTaskConfig

    # ------------------------------------------------------------------
    # Instance fields
    # ------------------------------------------------------------------
    default_tool_config: ComputerConfig = ComputerConfig()

    tasks_file: str | None = None
    """Path to a flat JSON array of task dicts (overrides OSWorld repo structure)."""

    test_set_path: str | None = None
    """Path to an evaluation_examples directory for testing (overrides OSWORLD_REPO_DIR)."""

    test_set_name: OSWorldTestSet = OSWorldTestSet.TEST_ALL
    """Filename of the test set index inside <evaluation_examples>/."""

    use_som: bool = False
    """Enable Set-of-Marks annotation for all tasks in this benchmark run."""

    @model_validator(mode="after")
    def _warn_on_conflicting_task_source(self) -> "OSWorldBenchmark":
        if self.tasks_file is not None and "test_set_name" in self.model_fields_set:
            logger.warning("Both 'tasks_file' and 'test_set_name' were specified — 'tasks_file' takes precedence.")
        return self

    # ------------------------------------------------------------------
    # get_task_configs() override — inject metadata into each config
    # ------------------------------------------------------------------

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        """
        Yield OSWorldTaskConfig objects, each carrying a copy of the TaskMetadata.

        Overrides the base class to inject metadata into configs so that
        OSWorldTaskConfig.make() can instantiate OSWorldTask without looking up
        the benchmark class (avoids circular import).
        """
        for tm in self.task_metadata.values():
            yield OSWorldTaskConfig(
                task_id=tm.id,
                tool_config=self.default_tool_config,
                seed=None,
                metadata=tm,
            )

    # ------------------------------------------------------------------
    # _setup() — replaces kusha's setup()
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """
        Prepare benchmark for task spawning.

        Steps:
          1. Check desktop_env is installed
          2. Ensure OSWorld repo is cloned (or validate tasks_file)
          3. Load task metadata from JSON files → populate instance shadow of task_metadata
        """
        self.install()

        logger.info(f"Setting up OSWorldBenchmark (provider={self._get_provider()})")

        if not self.task_metadata:
            if self.tasks_file:
                if not Path(self.tasks_file).exists():
                    raise FileNotFoundError(f"tasks_file not found: {self.tasks_file}")
                loaded = self._load_task_metadata_from_file(self.tasks_file)
            else:
                if not OSWORLD_REPO_DIR.exists():
                    logger.info("OSWorld repo not found — cloning...")
                    self._clone_osworld_repo()
                loaded = self._load_task_metadata_from_repo()

            # Populate instance-level shadow of the ClassVar
            # (same pattern used by cube.Benchmark.subset_from_list)
            object.__setattr__(self, "task_metadata", loaded)

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

    # ------------------------------------------------------------------
    # Task metadata loading helpers
    # ------------------------------------------------------------------

    def _load_task_metadata_from_file(self, tasks_file: str) -> dict[str, TaskMetadata]:
        """
        Load TaskMetadata from a flat JSON array.

        Expected format: [{"id": "...", "instruction": "...", "domain": "...", ...}, ...]
        Uses "instruction" as the task goal (abstract_description).
        """
        with open(tasks_file) as f:
            task_list = json.load(f)

        result = {}
        for td in task_list:
            task_id = td["id"]
            metadata = TaskMetadata(
                id=task_id,
                abstract_description=td.get("instruction", td.get("desc", "")),
                extra_info={
                    "domain": td.get("domain", "general"),
                    "snapshot": td.get("snapshot", "init_state"),
                    "config": td.get("config", []),
                    "evaluator": td.get("evaluator", {}),
                    "related_apps": td.get("related_apps", []),
                },
            )
            result[task_id] = metadata

        logger.info(f"Loaded {len(result)} task metadata entries from {tasks_file}")
        return result

    def _load_task_metadata_from_repo(self) -> dict[str, TaskMetadata]:
        """
        Load TaskMetadata from the OSWorld repo directory structure.

        Reads test_set_path/test_set_name → {domain: [task_id, ...]}
        Then reads test_set_path/examples/<domain>/<task_id>.json per task.
        """
        eval_examples_dir = Path(self.test_set_path) if self.test_set_path else OSWORLD_REPO_DIR / "evaluation_examples"
        test_set_file = eval_examples_dir / self.test_set_name

        if not test_set_file.exists():
            raise FileNotFoundError(
                f"Test set not found: {test_set_file}\nEnsure OSWorld is cloned and task files are present."
            )

        with open(test_set_file) as f:
            tasks_by_domain: dict[str, list[str]] = json.load(f)

        result = {}
        for domain_name, task_ids in tasks_by_domain.items():
            for task_id in task_ids:
                task_file = eval_examples_dir / "examples" / domain_name / f"{task_id}.json"
                if not task_file.exists():
                    logger.warning(f"Task file not found: {task_file}")
                    continue
                try:
                    with open(task_file) as f:
                        td = json.load(f)
                    td = self._fix_settings_paths(td)

                    metadata = TaskMetadata(
                        id=td.get("id", task_id),
                        abstract_description=td.get("instruction", ""),
                        extra_info={
                            "domain": domain_name,
                            "snapshot": td.get("snapshot", "init_state"),
                            "config": td.get("config", []),
                            "evaluator": td.get("evaluator", {}),
                            "related_apps": td.get("related_apps", []),
                        },
                    )
                    result[metadata.id] = metadata
                except Exception as e:
                    logger.error(f"Failed to load task {task_id}: {e}")

        logger.info(f"Loaded {len(result)} task metadata entries from OSWorld repo")
        return result

    def _fix_settings_paths(self, task: dict) -> dict:
        """
        Prepend OSWorld repo path to settings_file paths in task config items.

        Keeps relative paths in the task JSON working regardless of CWD.
        The OSWORLD_REPO env var overrides the default repo location.
        """
        updated = deepcopy(task)
        repo_dir = Path(os.environ.get("OSWORLD_REPO", str(OSWORLD_REPO_DIR)))
        for config_item in updated.get("config", []):
            params = config_item.get("parameters", {})
            if "settings_file" in params:
                params["settings_file"] = str(repo_dir / params["settings_file"])
        return updated

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
        """Return provider name from default_tool_config, if set."""
        if self.default_tool_config and hasattr(self.default_tool_config, "provider"):
            provider = self.default_tool_config.provider
            return provider.value if hasattr(provider, "value") else str(provider)
        return "local_qemu"

    # ------------------------------------------------------------------
    # install() — available for manual invocation; called from _setup()
    # ------------------------------------------------------------------

    def install(self) -> None:
        """
        Clone OSWorld repo and set up directory structure.

        Also sets PROXY_CONFIG_FILE env var to the correct path inside
        the cloned repo so desktop_env finds it at import time.
        """
        logger.info("Installing OSWorld benchmark...")
        if not OSWORLD_REPO_DIR.exists():
            self._clone_osworld_repo()
            logger.info(f"OSWorld repo cloned to {OSWORLD_REPO_DIR}")
        else:
            logger.info(f"OSWorld repo already present at {OSWORLD_REPO_DIR}")
        ensure_proxy_config_in_env()
        logger.info(f"Set PROXY_CONFIG_FILE={os.environ.get('PROXY_CONFIG_FILE', 'not set')}")

        logger.info("VM images will be downloaded automatically on first task reset.")
