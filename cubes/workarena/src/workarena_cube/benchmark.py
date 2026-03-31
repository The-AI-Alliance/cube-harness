"""WorkArena benchmark implementation for the CUBE framework."""

import logging
import random
from typing import ClassVar, Generator, Literal

from browsergym.workarena import get_all_tasks_agents
from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata
from pydantic import PrivateAttr

from workarena_cube.task import WorkArenaTaskConfig

logger = logging.getLogger(__name__)


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
        description="WorkArena ServiceNow benchmark tasks",
        tags=["browser", "web", "servicenow"],
        num_tasks=165,  # L1 default: ~33 unique tasks × 5 seeds
    )
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {}
    task_config_class: ClassVar[type[TaskConfig]] = WorkArenaTaskConfig

    level: Literal["l1", "l2", "l3"] = "l1"
    meta_seed: int = 42
    n_seeds_l1: int = 5
    shuffle: bool = True
    shuffle_seed: int = 42
    is_agent_curriculum: bool = False

    _task_tuples: list = PrivateAttr(default_factory=list)

    def _setup(self) -> None:
        """Enumerate WorkArena task classes and seeds for the configured level."""
        # Check the instance-level shadow, not the class-level ClassVar — this ensures
        # each instance sets up its own task_metadata independently, even if another
        # instance already populated the class-level dict.
        if "task_metadata" in self.__dict__:
            logger.debug("WorkArena benchmark already set up, skipping.")
            return
        logger.info(f"Setting up WorkArena benchmark (level={self.level})")
        task_tuples = get_all_tasks_agents(
            filter=self.level,
            meta_seed=self.meta_seed,
            n_seed_l1=self.n_seeds_l1,
            is_agent_curriculum=self.is_agent_curriculum,
        )
        if self.shuffle:
            random.seed(self.shuffle_seed)
            random.shuffle(task_tuples)
        self._task_tuples = task_tuples
        self._runtime_context = {"level": self.level, "n_tasks": len(task_tuples)}
        metadata: dict[str, TaskMetadata] = {}
        for task_class, _seed in task_tuples:
            task_id = task_class.get_task_id()
            task_class_path = f"{task_class.__module__}.{task_class.__qualname__}"
            metadata[task_id] = TaskMetadata(
                id=task_id,
                extra_info={"task_class_path": task_class_path, "level": self.level},
            )
        # Populate instance-level shadow so each instance sees its own task view
        # (e.g. after subset_from_list / subset_from_glob). Also update the class-level
        # attr so TaskConfig.make() can look up tasks via the ClassVar without re-running setup().
        object.__setattr__(self, "task_metadata", metadata)
        type(self).task_metadata = metadata
        logger.info(f"WorkArena benchmark setup complete: {len(task_tuples)} task(s)")

    def get_task_configs(self) -> Generator[WorkArenaTaskConfig, None, None]:
        """Yield one WorkArenaTaskConfig per (task_class, seed) tuple."""
        for task_class, seed in self._task_tuples:
            task_id = task_class.get_task_id()
            yield WorkArenaTaskConfig(
                task_id=task_id,
                seed=seed,
                tool_config=self.default_tool_config,
            )

    def close(self) -> None:
        """No-op: WorkArena has no server process to shut down."""
        logger.info("WorkArena benchmark closed.")
