"""WorkArena benchmark implementation for the CUBE framework."""

import json
import logging
from pathlib import Path
from typing import ClassVar, Literal

from browsergym.workarena import get_all_tasks_agents
from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.seed import AbstractSeedGenerator
from cube.task import TaskConfig, TaskMetadata
from pydantic import PrivateAttr

from workarena_cube.task import WorkArenaTaskConfig

logger = logging.getLogger(__name__)

_TASK_METADATA_JSON = Path(__file__).parent / "task_metadata.json"


class WorkArenaSeedGenerator(AbstractSeedGenerator):
    """Generates seeds for WorkArena tasks by delegating to get_all_tasks_agents().

    Seeds are derived from WorkArena's own RNG (seeded by meta_seed) to maintain
    compatibility with the original benchmark's evaluation protocol.

    Lazily loads on first call and caches {task_id: [seeds]} for the lifetime
    of this generator.
    """

    level: Literal["l1", "l2", "l3"]
    meta_seed: int = 42
    n_seeds_l1: int = 10
    is_agent_curriculum: bool = True

    _cache: dict[str, list[int]] | None = PrivateAttr(default=None)

    def _ensure_loaded(self) -> None:
        if self._cache is not None:
            return
        tuples = get_all_tasks_agents(
            filter=self.level,
            meta_seed=self.meta_seed,
            n_seed_l1=self.n_seeds_l1,
            is_agent_curriculum=self.is_agent_curriculum,
        )
        cache: dict[str, list[int]] = {}
        for task_class, seed in tuples:
            task_id = task_class.get_task_id()
            cache.setdefault(task_id, []).append(seed)
        self._cache = cache

    def __call__(self, task_metadata: TaskMetadata) -> list[int]:
        self._ensure_loaded()
        assert self._cache
        return self._cache.get(task_metadata.id, [])


class WorkArenaBenchmark(Benchmark):
    """CUBE Benchmark for WorkArena ServiceNow tasks.

    Task levels:
        - l1: Atomic tasks (~33 unique tasks x n_seeds_l1 seeds)
        - l2: Compositional tasks built from atomic subtasks
        - l3: Extended compositional tasks with company protocols

    Required environment variables:
        SNOW_INSTANCE_URL, SNOW_INSTANCE_UNAME, SNOW_INSTANCE_PWD
        or HUGGING_FACE_HUB_TOKEN for the hosted instance pool.
    """

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="workarena-cube",
        version="1.0.0",
        description=(
            "WorkArena ServiceNow benchmark tasks across three levels. "
            "By default all task types from all levels are loaded. "
            "Use named_subset('l1'/'l2'/'l3') to filter by level. "
            "For human curriculum: bench.named_subset('l2').subset_from_glob('extra_info.in_human_curriculum', 'True')."
        ),
        tags=["browser", "web", "servicenow"],
        named_subsets={
            "l1": ("extra_info.level", "l1"),
            "l2": ("extra_info.level", "l2"),
            "l3": ("extra_info.level", "l3"),
        },
        num_tasks=333,
    )
    # task_metadata: populated automatically at import time in Benchmark.__init_subclass__
    task_config_class: ClassVar[type[TaskConfig]] = WorkArenaTaskConfig

    level: Literal["l1", "l2", "l3"] = "l1"
    meta_seed: int = 42
    n_seeds_l1: int = 10
    is_agent_curriculum: bool = True

    # ── install / uninstall ────────────────────────────────────────

    @classmethod
    def install(cls) -> None:
        """Enumerate all WorkArena task types and save task_metadata.json.

        Calls get_all_tasks_agents for L1, L2 agent, L3 agent (superset), and also
        L2/L3 human curriculum to mark which tasks are in in_human_curriculum.
        Uses n_seed_l1=1 and meta_seed=0 — seeds are irrelevant here; we only need
        the task classes.

        Safe to call multiple times: skips if task_metadata.json already exists.
        """
        if _TASK_METADATA_JSON.exists():
            logger.info("task_metadata.json already exists, skipping installation")
            return

        metadata: dict[str, TaskMetadata] = {}

        # L1 — no curriculum concept
        for task_class, _ in get_all_tasks_agents(filter="l1", meta_seed=0, n_seed_l1=1):
            task_id = task_class.get_task_id()
            if task_id not in metadata:
                metadata[task_id] = TaskMetadata(
                    id=task_id,
                    extra_info={
                        "task_class_path": f"{task_class.__module__}.{task_class.__qualname__}",
                        "level": "l1",
                        "in_human_curriculum": False,
                    },
                )

        # L2 and L3 — agent curriculum is the superset; mark human curriculum tasks
        for level in ("l2", "l3"):
            human_ids: set[str] = {
                task_class.get_task_id()
                for task_class, _ in get_all_tasks_agents(
                    filter=level, meta_seed=0, n_seed_l1=1, is_agent_curriculum=False
                )
            }
            for task_class, _ in get_all_tasks_agents(filter=level, meta_seed=0, n_seed_l1=1, is_agent_curriculum=True):
                task_id = task_class.get_task_id()
                if task_id not in metadata:
                    metadata[task_id] = TaskMetadata(
                        id=task_id,
                        extra_info={
                            "task_class_path": f"{task_class.__module__}.{task_class.__qualname__}",
                            "level": level,
                            "in_human_curriculum": task_id in human_ids,
                        },
                    )

        _TASK_METADATA_JSON.write_text(json.dumps([tm.model_dump() for tm in metadata.values()], indent=2))
        cls.task_metadata = metadata
        n_l1 = sum(1 for tm in metadata.values() if tm.extra_info["level"] == "l1")
        n_l2 = sum(1 for tm in metadata.values() if tm.extra_info["level"] == "l2")
        n_l3 = sum(1 for tm in metadata.values() if tm.extra_info["level"] == "l3")
        logger.info(f"Saved {len(metadata)} tasks to {_TASK_METADATA_JSON} (l1={n_l1}, l2={n_l2}, l3={n_l3})")

    @classmethod
    def uninstall(cls) -> None:
        """Remove task_metadata.json."""
        if _TASK_METADATA_JSON.exists():
            _TASK_METADATA_JSON.unlink()
            cls.task_metadata = {}
            logger.info(f"Removed {_TASK_METADATA_JSON}")

    # ── lifecycle ──────────────────────────────────────────────────

    def _setup(self) -> None:
        """Configure seed_generator from pre-loaded task_metadata."""
        # Check the instance-level shadow, not the class-level ClassVar — this ensures
        # each instance sets up its own task_metadata independently, even if another
        # instance already populated the class-level dict.
        if "task_metadata" in self.__dict__:
            logger.debug("WorkArena benchmark already set up, skipping.")
            return
        logger.info(f"Setting up WorkArena benchmark (level={self.level})")

        self.seed_generator = WorkArenaSeedGenerator(
            level=self.level,
            meta_seed=self.meta_seed,
            n_seeds_l1=self.n_seeds_l1,
            is_agent_curriculum=self.is_agent_curriculum,
        )
        # Force seed loading now so we can report the task count
        self.seed_generator._ensure_loaded()

        # Populate instance-level shadow so each instance sees its own task view.
        # Filter to the configured level so get_task_configs() only iterates relevant tasks.
        level_metadata = {
            tid: tm for tid, tm in type(self).task_metadata.items() if tm.extra_info.get("level") == self.level
        }
        object.__setattr__(self, "task_metadata", level_metadata)
        type(self).task_metadata = level_metadata

        assert self.seed_generator._cache is not None
        n_task_configs = sum(len(seeds) for seeds in self.seed_generator._cache.values())
        self._runtime_context = {"level": self.level, "n_task_configs": n_task_configs}
        logger.info(
            f"WorkArena benchmark setup complete: {len(level_metadata)} task type(s), {n_task_configs} task config(s)"
        )

    def close(self) -> None:
        """No-op: WorkArena has no server process to shut down."""
        logger.info("WorkArena benchmark closed.")
