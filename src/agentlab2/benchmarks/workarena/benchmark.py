"""WorkArena benchmark for AgentLab2."""

import logging
import random
from typing import Literal

from browsergym.workarena import get_all_tasks_agents

from agentlab2.benchmark import Benchmark
from agentlab2.benchmarks.workarena.task import WorkArenaTask

logger = logging.getLogger(__name__)


class WorkArenaBenchmark(Benchmark):
    """AgentLab2 Benchmark for WorkArena ServiceNow tasks.

    WorkArena is a benchmark suite for evaluating web agents on ServiceNow-based
    knowledge work tasks. It includes atomic tasks (L1) and compositional tasks
    (L2, L3) with increasing complexity.

    Task Levels:
        - L1: Atomic tasks (33 tasks) - single-purpose tasks like form filling,
              navigation, list operations, service catalog orders.
        - L2: Compositional tasks - multi-step tasks composed of atomic subtasks.
        - L3: Extended compositional tasks with company protocols and more
              complex reasoning requirements.

    Environment Variables Required:
        - SNOW_INSTANCE_URL: ServiceNow instance URL
        - SNOW_INSTANCE_UNAME: ServiceNow username
        - SNOW_INSTANCE_PWD: ServiceNow password
        OR
        - HUGGING_FACE_HUB_TOKEN: For accessing gated instance pool dataset

    Example:
        ```python
        from agentlab2.benchmarks.workarena import WorkArenaBenchmark
        from agentlab2.tools.browsergym import BrowsergymConfig

        benchmark = WorkArenaBenchmark(
            tool_config=BrowsergymConfig(headless=False),
            level="l1",
            n_seeds=5,
        )
        benchmark.setup()
        tasks = benchmark.load_tasks()
        ```
    """

    level: Literal["l1", "l2", "l3"] = "l1"
    meta_seed: int = 42
    n_seeds_l1: int = 5
    shuffle: bool = True
    shuffle_seed: int = 42
    is_agent_curriculum: bool = False

    def setup(self) -> None:
        """Set up the WorkArena benchmark.

        This method verifies that the WorkArena package is available
        and that ServiceNow credentials are configured.
        """
        logger.info(f"Setting up WorkArena benchmark (level={self.level})")

        logger.info("WorkArena benchmark setup complete")

    def load_tasks(self) -> list[WorkArenaTask]:
        """Load WorkArena tasks based on the configured level.

        Returns:
            List of WorkArenaTask instances.
        """

        logger.info(f"Loading WorkArena tasks (level={self.level}, meta_seed={self.meta_seed})")

        # Get task tuples from WorkArena
        task_tuples = get_all_tasks_agents(
            filter=self.level,
            meta_seed=self.meta_seed,
            n_seed_l1=self.n_seeds_l1,
            is_agent_curriculum=self.is_agent_curriculum,
        )

        # Create AgentLab2 tasks
        tasks = [
            WorkArenaTask(
                id=task_class.get_task_id(),
                workarena_task_class=task_class,
                seed=seed,
                level=self.level,
            )
            for task_class, seed in task_tuples
        ]

        if self.shuffle:
            random.seed(self.shuffle_seed)
            random.shuffle(tasks)

        logger.info(f"Loaded {len(tasks)} WorkArena tasks")
        return tasks

    def close(self) -> None:
        """Clean up WorkArena benchmark resources."""
        logger.info("Closing WorkArena benchmark")
