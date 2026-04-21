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
from typing import ClassVar

from cube import LocalInfraConfig
from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.container import ContainerBackend
from cube.resource import InfraConfig, ResourceConfig
from cube.task import TaskConfig, TaskMetadata
from pydantic import Field

from waa_cube.azure import WAA_WINDOWS_RESOURCE
from waa_cube.computer import ComputerConfig
from waa_cube.task import WAATask

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
        metadata:    TaskMetadata embedded at config creation time so Ray workers
                     don't need to look it up via the class variable.
    """

    infra: InfraConfig | None = None
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
        exec_info = WAABenchmark.load_task_execution_info(self.task_id)
        metadata = metadata.model_copy(update={"extra_info": exec_info})
        return WAATask(
            metadata=metadata,
            tool_config=self.tool_config,
            infra=self.infra,
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
            "vm": "Windows 11 (infra-managed VM)",
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

    infra: InfraConfig | None = Field(default_factory=LocalInfraConfig)
    """InfraConfig used to launch VMs (defaults to LocalInfraConfig)."""

    resources: list[ResourceConfig] = [WAA_WINDOWS_RESOURCE]
    """Resources required by this benchmark. Passed to infra.provision() before eval."""

    # ------------------------------------------------------------------
    # get_task_configs() — inject infra into each config
    # ------------------------------------------------------------------

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        """Yield WAATaskConfig objects with infra and metadata injected."""
        for tm in self.task_metadata.values():
            yield WAATaskConfig(
                task_id=tm.id,
                tool_config=self.default_tool_config,
                seed=None,
                infra=self.infra,
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

        if self.infra is None:
            raise RuntimeError("WAABenchmark requires an InfraConfig. Set infra= when constructing.")

        # Ensure all declared resources are provisioned (idempotent).
        for resource in self.resources:
            if self.infra.provision_status(resource) == "ready":
                logger.info("Resource %s already provisioned", resource.name)
                continue
            logger.info("Provisioning resource %s...", resource.name)
            self.infra.provision(resource)

        self._runtime_context = {"waa": True}
        logger.info("WAABenchmark ready with %d tasks", len(self.task_metadata))

    def install(self) -> None:
        """Populate per-task execution cache from shipped task_metadata.json."""
        exec_cache_dir = self.task_execution_cache_dir()
        exec_cache_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for task_id, tm in self.task_metadata.items():
            cache_file = exec_cache_dir / f"{task_id}.json"
            new_content = json.dumps(tm.extra_info, indent=2)
            if cache_file.exists() and cache_file.read_text() == new_content:
                continue
            cache_file.write_text(new_content)
            written += 1
        logger.info("WAABenchmark.install() — wrote %d execution cache files to %s", written, exec_cache_dir)

    def close(self) -> None:
        """No global resources to release — VMs are stopped per-task."""
        logger.info("Closing WAABenchmark — no global resources to release")

    # ------------------------------------------------------------------
    # Debug task loading
    # ------------------------------------------------------------------

    def _load_task_metadata_from_file(self, tasks_file: str) -> dict[str, TaskMetadata]:
        """Load TaskMetadata from a flat JSON list and write exec_info cache entries (used by the debug suite)."""
        with open(tasks_file) as f:
            tasks: list[dict] = json.load(f)
        exec_cache_dir = self.task_execution_cache_dir()
        exec_cache_dir.mkdir(parents=True, exist_ok=True)
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
            (exec_cache_dir / f"{task_id}.json").write_text(json.dumps(metadata.extra_info, indent=2))
        logger.info("Loaded %d task metadata entries from %s", len(result), tasks_file)
        return result
