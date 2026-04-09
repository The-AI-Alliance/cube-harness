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

import enum
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Generator
from copy import deepcopy
from dotenv import load_dotenv
from pathlib import Path
from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.container import ContainerBackend
from cube.task import TaskConfig
from cube.vm import VMBackend

from osworld_cube._paths import OSWORLD_BASE_DIR, OSWORLD_REPO_DIR, OSWORLD_VM_DIR
from osworld_cube.computer import ComputerConfig
from osworld_cube.task import OSWorldTask, OSWorldTaskMetadata

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

    Fields:
        task_id:     inherited from TaskConfig
        tool_config: inherited from TaskConfig
        seed:        inherited (ignored for OSWorld — tasks are deterministic)
        vm_backend:  VMBackend to use for this task (passed by benchmark.get_task_configs()).

    make() looks up TaskMetadata from OSWorldBenchmark.task_metadata (a ClassVar
    populated by OSWorldBenchmark.setup()).
    """

    vm_backend: VMBackend | None = None
    use_som: bool = False

    def make(
        self,
        runtime_context: dict | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> OSWorldTask:
        """Instantiate OSWorldTask from this config.

        Loads per-task execution data (config, evaluator) from the local cache,
        fixes machine-specific settings_file paths, and merges into metadata.extra_info.
        """
        metadata = OSWorldBenchmark.task_metadata[self.task_id]
        exec_info = OSWorldBenchmark.load_task_execution_info(self.task_id)
        fixed_config = OSWorldBenchmark._fix_config_paths(exec_info.get("config", []))
        metadata = metadata.model_copy(update={"extra_info": {**exec_info, "config": fixed_config}})

        if self.tool_config is None:
            raise ValueError(
                f"OSWorldTaskConfig for task '{self.task_id}' has no tool_config. "
                "Pass default_tool_config=ComputerConfig(...) to OSWorldBenchmark."
            )
        return OSWorldTask(
            metadata=metadata,
            tool_config=self.tool_config,
            vm_backend=self.vm_backend,
            runtime_context=runtime_context,
            container_backend=container_backend,
            use_som=self.use_som,
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
        use_som:              bool            — Set-of-Marks mode for all tasks
        vm_backend:           VMBackend | None — backend to provision VMs; None = attach externally

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
        num_tasks=369,
        tags=["desktop", "gui", "multimodal"],
        named_subsets={
            "test_all": ("test_sets", "*'test_all'*"),
            "test_small": ("test_sets", "*'test_small'*"),
            "test_nogdrive": ("test_sets", "*'test_nogdrive'*"),
            "test_infeasible": ("test_sets", "*'test_infeasible'*"),
        },
    )
    task_metadata: ClassVar[dict[str, OSWorldTaskMetadata]]  # type: ignore[assignment]  # narrowed subtype
    task_config_class: ClassVar[type[TaskConfig]] = OSWorldTaskConfig

    # ------------------------------------------------------------------
    # Instance fields
    # ------------------------------------------------------------------
    default_tool_config: ComputerConfig = ComputerConfig()  # type: ignore[assignment]

    use_som: bool = False
    """Enable Set-of-Marks annotation for all tasks in this benchmark run."""

    vm_backend: VMBackend | None = None
    """VM backend used to provision VMs for each task. If None, tasks will fail
    unless a VM is attached externally via computer.attach_vm()."""

    # ------------------------------------------------------------------
    # cache_dir() override — set to OSWORLD_BASE_DIR so that task execution info and repo clone are stored under the same directory
    # ------------------------------------------------------------------
    @classmethod
    def cache_dir(cls) -> Path:
        return OSWORLD_BASE_DIR

    # ------------------------------------------------------------------
    # get_task_configs() override — inject vm_backend into each config
    # ------------------------------------------------------------------

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        """Yield task config objects, injecting vm_backend and use_som from the benchmark."""
        tc_cls = self.task_config_class
        assert issubclass(tc_cls, OSWorldTaskConfig)
        for tm in self.task_metadata.values():
            yield tc_cls(
                task_id=tm.id,
                tool_config=self.default_tool_config,
                seed=None,
                vm_backend=self.vm_backend,
                use_som=self.use_som,
            )

    # ------------------------------------------------------------------
    # _setup()
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Prepare benchmark for task execution. Essentially a no-op in this case."""
        provider = type(self.vm_backend).__name__ if self.vm_backend else "none"
        logger.info(f"Setting up OSWorldBenchmark (provider={provider})...")
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

    @staticmethod
    def _fix_config_paths(config: list) -> list:
        """Rewrite relative settings_file paths in config items to absolute paths."""
        result = deepcopy(config)
        for item in result:
            params = item.get("parameters", {})
            if "settings_file" in params:
                params["settings_file"] = str(OSWORLD_REPO_DIR / params["settings_file"])
        return result

    @staticmethod
    def _clone_osworld_repo() -> None:
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

    # ------------------------------------------------------------------
    # install() / uninstall()
    # ------------------------------------------------------------------

    @classmethod
    def install(cls) -> None:
        """Clone the OSWorld repo (if missing) and populate the per-task execution cache.

        task_metadata.json is a shipped package resource and is NOT generated here.
        Run scripts/create_task_metadata.py to regenerate it from the repo.
        """
        if not cls.task_metadata:
            raise RuntimeError(
                "task_metadata is empty — task_metadata.json is missing or was not loaded. "
                "Run scripts/create_task_metadata.py to generate it."
            )

        logger.info("Installing OSWorld benchmark...")
        # Repo is needed at task execution time; always ensure it is present.
        if not OSWORLD_REPO_DIR.exists():
            OSWorldBenchmark._clone_osworld_repo()
            logger.info(f"OSWorld repo cloned to {OSWORLD_REPO_DIR}")
        else:
            logger.info(f"OSWorld repo already present at {OSWORLD_REPO_DIR}")
        ensure_proxy_config_in_env()
        load_dotenv()  # Load the .env file
        logger.info(f"Set PROXY_CONFIG_FILE={os.environ.get('PROXY_CONFIG_FILE', 'not set')}")
        logger.info("VM images will be downloaded automatically on first task reset.")

        eval_examples_dir = OSWORLD_REPO_DIR / "evaluation_examples"
        exec_cache_dir = cls.task_execution_cache_dir()
        exec_cache_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for task_id, tm in cls.task_metadata.items():
            cache_file = exec_cache_dir / f"{task_id}.json"
            if cache_file.exists():
                continue
            task_file = eval_examples_dir / "examples" / tm.domain / f"{task_id}.json"
            if not task_file.exists():
                logger.warning(f"Task file not found: {task_file}")
                continue
            try:
                with open(task_file) as f:
                    td = json.load(f)
                exec_info = {
                    "config": td.get("config", []),
                    "evaluator": td.get("evaluator", {}),
                }
                cache_file.write_text(json.dumps(exec_info, indent=2))
                written += 1
            except Exception as e:
                logger.error(f"Failed to write execution cache for {task_id}: {e}")

        logger.info(f"Wrote {written} new execution cache files to {exec_cache_dir}")
        logger.info("OSWorldBenchmark.install() done")

    @classmethod
    def uninstall(cls) -> None:
        """Remove the execution cache and the cloned OSWorld repo."""
        exec_cache_dir = cls.task_execution_cache_dir()
        if exec_cache_dir.exists():
            shutil.rmtree(exec_cache_dir)
            logger.info(f"Removed execution cache at {exec_cache_dir}")
        if OSWORLD_REPO_DIR.exists():
            shutil.rmtree(OSWORLD_REPO_DIR)
            logger.info(f"Removed OSWorld repo at {OSWORLD_REPO_DIR}")
