"""WorkArena benchmark implementation for the CUBE framework."""

import logging
from typing import ClassVar, Self

from browsergym.workarena import get_all_tasks_agents
from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.seed import AbstractSeedGenerator
from cube.task import TaskConfig, TaskMetadata
from pydantic import PrivateAttr, model_validator

from workarena_cube.task import WorkArenaTaskConfig, WorkArenaTaskMetadata

logger = logging.getLogger(__name__)


class WorkArenaSeedGenerator(AbstractSeedGenerator):
    """Generates seeds for WorkArena tasks by delegating to get_all_tasks_agents().

    Covers all three levels (l1, l2, l3) so it works naturally with any subset
    produced by named_subset() or subset_from_glob().

    Seeds are derived from WorkArena's own RNG (seeded by meta_seed) to maintain
    compatibility with the original benchmark's evaluation protocol.

    Lazily loads on first call and caches {task_id: [seeds]} for the lifetime
    of this generator.
    """

    meta_seed: int = 42
    n_seeds_l1: int = 10
    is_agent_curriculum: bool = True

    _cache: dict[str, list[int]] | None = PrivateAttr(default=None)

    def _ensure_loaded(self) -> None:
        if self._cache is not None:
            return
        cache: dict[str, list[int]] = {}
        for level in ("l1", "l2", "l3"):
            for task_class, seed in get_all_tasks_agents(
                filter=level,
                meta_seed=self.meta_seed,
                n_seed_l1=self.n_seeds_l1,
                is_agent_curriculum=self.is_agent_curriculum,
            ):
                task_id = task_class.get_task_id()
                cache.setdefault(task_id, []).append(seed)
        self._cache = cache

    def __call__(self, task_metadata: TaskMetadata) -> list[int]:
        self._ensure_loaded()
        assert self._cache
        return self._cache.get(task_metadata.id, [])


class WorkArenaBenchmark(Benchmark["WorkArenaBenchmarkConfig"]):
    """Runtime pair — WorkArena tasks connect to a remote ServiceNow instance,
    so there is no shared infrastructure to provision in _setup().
    """

    def _setup(self) -> None:
        logger.info(f"WorkArena benchmark ready with {self.config.num_tasks} tasks")

    def close(self) -> None:
        logger.info("WorkArena benchmark closed.")


class WorkArenaBenchmarkConfig(BenchmarkConfig[WorkArenaTaskMetadata]):
    """CUBE BenchmarkConfig for WorkArena ServiceNow tasks.

    By default loads all task types from all levels (l1, l2, l3).
    Use named_subset() or subset_from_glob() in user-land to filter:

        cfg.named_subset("l1")                                                  # L1 only
        cfg.named_subset("l2").subset_from_glob("in_human_curriculum", "True")  # L2 human curriculum

    Required environment variables:
        SNOW_INSTANCE_URL, SNOW_INSTANCE_UNAME, SNOW_INSTANCE_PWD
        or HUGGING_FACE_HUB_TOKEN for the hosted instance pool.

    task_metadata.json is a shipped package resource containing lightweight public fields
    (level, in_human_curriculum, task_class_path). No heavy execution data exists — all
    task logic is available from the browsergym-workarena library at runtime.

    To regenerate task_metadata.json (developer use only), run:
        scripts/generate_task_metadata.py
    """

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="workarena-cube",
        version="1.0.0",
        description=(
            "WorkArena ServiceNow benchmark tasks across three levels. "
            "By default all task types from all levels are loaded. "
            "Use named_subset('l1'/'l2'/'l3') to filter by level. "
            "For human curriculum: cfg.named_subset('l2').subset_from_glob('in_human_curriculum', 'True')."
        ),
        tags=["browser", "web", "servicenow"],
        named_subsets={
            "l1": ("level", "l1"),
            "l2": ("level", "l2"),
            "l3": ("level", "l3"),
        },
        num_tasks=333,
    )
    task_config_class: ClassVar[type[TaskConfig]] = WorkArenaTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = WorkArenaBenchmark

    meta_seed: int = 42
    n_seeds_l1: int = 10
    is_agent_curriculum: bool = True

    @model_validator(mode="after")
    def _init_seed_generator(self) -> Self:
        """Initialize seed_generator at construction time from config fields."""
        if self.seed_generator is None:
            self.seed_generator = WorkArenaSeedGenerator(
                meta_seed=self.meta_seed,
                n_seeds_l1=self.n_seeds_l1,
                is_agent_curriculum=self.is_agent_curriculum,
            )
        return self
