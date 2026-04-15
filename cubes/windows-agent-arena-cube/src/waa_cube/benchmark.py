"""WAABenchmark and WAATaskConfig — CUBE benchmark for WindowsAgentArena.

Entry point:
    bench = WAABenchmark()
    bench.install()
    bench.setup()
    for task_config in bench.get_task_configs():
        task = task_config.make()
        obs, info = task.reset()
        ...
        task.close()

Filter by domain after setup():
    vscode_bench = bench.subset_from_glob("extra_info.domain", "vscode")

Task metadata is shipped as task_metadata.json (auto-loaded at import time).
To regenerate: python scripts/create_task_metadata.py --force
"""

from __future__ import annotations

import json
import logging
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
        task_metadata:       ClassVar[dict[str, TaskMetadata]]  (auto-loaded from task_metadata.json)
        task_config_class:   type[TaskConfig] = WAATaskConfig

    Constructor params:
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

    # Auto-loaded from task_metadata.json by __init_subclass__
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {}

    task_config_class: ClassVar[type[TaskConfig]] = WAATaskConfig

    # ------------------------------------------------------------------
    # Instance fields
    # ------------------------------------------------------------------

    default_tool_config: ComputerConfig = ComputerConfig()

    tasks_file: str | None = None
    """Optional flat JSON task list for debug overlay (merged on top of shipped metadata)."""

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

    def _setup(self) -> None:
        """Initialise runtime state.

        If tasks_file is set (debug mode), overlay its tasks onto the shipped
        task_metadata.json so the debug task is available at runtime.
        """
        if self.tasks_file is not None:
            loaded = self._load_task_metadata_from_file(self.tasks_file)
            object.__setattr__(self, "task_metadata", {**self.task_metadata, **loaded})
            type(self).task_metadata = self.task_metadata

        if isinstance(self.vm_backend, WAADockerVMBackend):
            self.vm_backend.cleanup_stale_overlays()

        self._runtime_context = {"waa": True}
        logger.info("WAABenchmark ready with %d tasks", len(self.task_metadata))

    def install(self) -> None:
        """Prepare the Windows VM image if not already built.

        Delegates to WAADockerVMBackend.install() when vm_backend is configured.
        No-op if no vm_backend is set (e.g. when using an externally managed VM).
        """
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
    # Debug task loading
    # ------------------------------------------------------------------

    def _load_task_metadata_from_file(self, tasks_file: str) -> dict[str, TaskMetadata]:
        """Load TaskMetadata from a flat JSON list (used by the debug suite)."""
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
