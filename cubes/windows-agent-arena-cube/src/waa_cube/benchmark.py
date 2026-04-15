"""WAABenchmark and WAATaskConfig — CUBE benchmark for WindowsAgentArena.

Entry point:
    bench = WAABenchmark(eval_examples_dir="/path/to/evaluation_examples_windows")
    bench.setup()
    for task_config in bench.get_task_configs():
        task = task_config.make()
        obs, info = task.reset()
        ...
        task.close()

Filter by domain after setup():
    vscode_bench = bench.subset_from_glob("extra_info.domain", "vscode")
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.container import ContainerBackend
from cube.task import TaskConfig, TaskMetadata
from cube.vm import VMBackend

from waa_cube.computer import ComputerConfig
from waa_cube.task import WAATask
from waa_cube.vm_backend.backend import WAADockerVMBackend

logger = logging.getLogger(__name__)

WAA_EVAL_EXAMPLES_ENV = "WAA_EVAL_EXAMPLES_DIR"


# ---------------------------------------------------------------------------
# WAATaskConfig
# ---------------------------------------------------------------------------


class WAATaskConfig(TaskConfig):
    """Serialisable config for a single WAA task.

    Fields:
        task_id:     inherited from TaskConfig
        tool_config: inherited from TaskConfig
        seed:        inherited (ignored — tasks are deterministic)
        vm_backend:  VMBackend to use (passed by WAABenchmark.get_task_configs())
        metadata:    TaskMetadata embedded at config creation time so Ray workers
                     don't need to look it up via the class variable.
    """

    vm_backend: VMBackend | None = None
    metadata: TaskMetadata | None = None

    def make(
        self,
        runtime_context: dict | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> WAATask:
        if self.metadata is not None:
            metadata = self.metadata
        else:
            metadata = WAABenchmark.task_metadata[self.task_id]
        if self.tool_config is None:
            raise ValueError(
                f"WAATaskConfig for task '{self.task_id}' has no tool_config. "
                "Pass default_tool_config=ComputerConfig(...) to WAABenchmark."
            )
        return WAATask(
            metadata=metadata,
            tool_config=self.tool_config,
            vm_backend=self.vm_backend,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )


# ---------------------------------------------------------------------------
# WAABenchmark
# ---------------------------------------------------------------------------


class WAABenchmark(Benchmark):
    """CUBE benchmark wrapping the WindowsAgentArena evaluation suite.

    Reference: https://github.com/microsoft/WindowsAgentArena

    Class-level attributes (required by cube.benchmark.Benchmark):
        benchmark_metadata:  ClassVar[BenchmarkMetadata]
        task_metadata:       ClassVar[dict[str, TaskMetadata]]  (populated in _setup())
        task_config_class:   type[TaskConfig] = WAATaskConfig

    Constructor params:
        eval_examples_dir:    str | None  — path to evaluation_examples_windows/
                                            Falls back to WAA_EVAL_EXAMPLES_DIR env var.
        test_set_name:        str         — index file inside eval_examples_dir/ (default: test_all.json)
        default_tool_config:  ComputerConfig
        use_som:              bool
        vm_backend:           WAADockerVMBackend | None
    """

    # ------------------------------------------------------------------
    # Required class variables
    # ------------------------------------------------------------------

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="waa",
        version="1.0.0",
        description="WindowsAgentArena: Benchmarking AI Agents on Windows 11",
        authors=["Rogerio Bonatti et al."],
        license="MIT",
        requirements={
            "vm": "Windows 11 (Docker + QEMU)",
            "ram_gb": 8,
            "disk_gb": 60,
        },
        num_tasks=154,
        tags=["desktop", "gui", "windows", "multimodal"],
    )

    # Placeholder: populated per-instance in _setup() via object.__setattr__
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {}

    task_config_class: ClassVar[type[TaskConfig]] = WAATaskConfig

    # ------------------------------------------------------------------
    # Instance fields
    # ------------------------------------------------------------------

    default_tool_config: ComputerConfig = ComputerConfig()

    eval_examples_dir: str | None = None
    """Path to evaluation_examples_windows/. Falls back to WAA_EVAL_EXAMPLES_DIR env var."""

    test_set_name: str = "test_all.json"
    """Index file inside eval_examples_dir/ to load."""

    tasks_file: str | None = None
    """Flat JSON task list (list of task dicts). Mutually exclusive with eval_examples_dir/test_set_name.
    Used by the debug suite to load tasks from a bundled file without a full eval dir."""

    use_som: bool = False
    """Enable Set-of-Marks annotation for all tasks."""

    vm_backend: VMBackend | None = None
    """VM backend used to provision the WAA Docker VM. If None, tasks will fail unless
    a VM is attached externally via computer.attach_vm()."""

    # ------------------------------------------------------------------
    # get_task_configs() — inject vm_backend into each config
    # ------------------------------------------------------------------

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        """Yield WAATaskConfig objects with vm_backend and metadata injected."""
        for tm in self.task_metadata.values():
            yield WAATaskConfig(
                task_id=tm.id,
                tool_config=self.default_tool_config,
                seed=None,
                vm_backend=self.vm_backend,
                metadata=tm,
            )

    # ------------------------------------------------------------------
    # _setup()
    # ------------------------------------------------------------------

    def _ensure_task_metadata(self) -> None:
        """Load task metadata if not already populated (idempotent)."""
        if self.task_metadata:
            return
        if self.tasks_file is not None:
            if not Path(self.tasks_file).exists():
                raise FileNotFoundError(f"tasks_file not found: {self.tasks_file}")
            loaded = self._load_task_metadata_from_file(self.tasks_file)
        else:
            loaded = self._load_task_metadata()
        object.__setattr__(self, "task_metadata", loaded)
        type(self).task_metadata = loaded

    def _setup(self) -> None:
        """Initialise runtime state. Loads task metadata if needed (Ray workers)."""
        self._ensure_task_metadata()

        if isinstance(self.vm_backend, WAADockerVMBackend):
            self.vm_backend.cleanup_stale_overlays()

        self._runtime_context = {"waa": True}
        logger.info("WAABenchmark ready with %d tasks", len(self.task_metadata))

    def setup(self) -> None:
        """Override base setup() — load metadata before the guard check."""
        self._ensure_task_metadata()
        super().setup()

    def install(self) -> None:
        """Prepare the Windows VM image if not already built.

        Delegates to WAADockerVMBackend.install() when vm_backend is configured.
        No-op if no vm_backend is set (e.g. when using an externally managed VM).
        """
        self._ensure_task_metadata()

        if self.vm_backend is None:
            logger.info("No vm_backend set — skipping WAA image preparation")
            return
        if not isinstance(self.vm_backend, WAADockerVMBackend):
            logger.info("vm_backend is not WAADockerVMBackend — skipping image preparation")
            return
        self.vm_backend.install()

    def close(self) -> None:
        """No global resources to release — VMs are stopped per-task."""
        logger.info("Closing WAABenchmark — no global resources to release")

    # ------------------------------------------------------------------
    # Task metadata loading
    # ------------------------------------------------------------------

    def _get_eval_examples_dir(self) -> Path:
        if self.eval_examples_dir:
            return Path(self.eval_examples_dir)
        env_val = os.environ.get(WAA_EVAL_EXAMPLES_ENV)
        if env_val:
            return Path(env_val)
        raise ValueError(
            f"eval_examples_dir is not set and {WAA_EVAL_EXAMPLES_ENV} env var is missing.\n"
            "Pass eval_examples_dir='/path/to/evaluation_examples_windows' to WAABenchmark or\n"
            f"set the {WAA_EVAL_EXAMPLES_ENV} environment variable."
        )

    def _load_task_metadata_from_file(self, tasks_file: str) -> dict[str, TaskMetadata]:
        """Load TaskMetadata from a flat JSON list (used by the debug suite).

        The file must contain a JSON array of task dicts, each with at minimum
        ``id``, ``instruction``, ``snapshot``, ``config``, ``evaluator``, and
        optionally ``domain`` and ``related_apps``.
        """
        with open(tasks_file) as f:
            tasks: list[dict] = json.load(f)
        result: dict[str, TaskMetadata] = {}
        for td in tasks:
            task_id = td.get("id", "")
            if not task_id:
                logger.warning("Skipping task with missing 'id' in %s", tasks_file)
                continue
            metadata = TaskMetadata(
                id=task_id,
                abstract_description=td.get("instruction", ""),
                extra_info={
                    "domain": td.get("domain", "debug"),
                    "snapshot": td.get("snapshot", "init_state"),
                    "config": td.get("config", []),
                    "evaluator": td.get("evaluator", {}),
                    "related_apps": td.get("related_apps", []),
                },
            )
            result[task_id] = metadata
        logger.info("Loaded %d task metadata entries from %s", len(result), tasks_file)
        return result

    def _load_task_metadata(self) -> dict[str, TaskMetadata]:
        """Load TaskMetadata from evaluation_examples_windows/ directory structure.

        Reads <eval_examples_dir>/test_set_name → {domain: [task_id, ...]}
        Then reads <eval_examples_dir>/examples/<domain>/<task_id>.json per task.
        """
        eval_dir = self._get_eval_examples_dir()
        test_set_file = eval_dir / self.test_set_name

        if not test_set_file.exists():
            raise FileNotFoundError(
                f"Test set file not found: {test_set_file}\n"
                "Ensure eval_examples_dir points to evaluation_examples_windows/."
            )

        with open(test_set_file) as f:
            tasks_by_domain: dict[str, list[str]] = json.load(f)

        result: dict[str, TaskMetadata] = {}
        for domain_name, task_ids in tasks_by_domain.items():
            for task_id in task_ids:
                task_file = eval_dir / "examples" / domain_name / f"{task_id}.json"
                if not task_file.exists():
                    logger.warning("Task file not found: %s", task_file)
                    continue
                try:
                    with open(task_file) as f:
                        td = json.load(f)
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
                except Exception as exc:
                    logger.error("Failed to load task %s: %s", task_id, exc)

        logger.info("Loaded %d task metadata entries from WAA repo", len(result))
        return result
