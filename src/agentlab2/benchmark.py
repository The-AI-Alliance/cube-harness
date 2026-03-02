import warnings
from abc import ABC, abstractmethod

from pydantic import Field

from agentlab2.core import Task, TypedBaseModel
from agentlab2.environment import EnvConfig
from cube.tool import ToolConfig


class Benchmark(TypedBaseModel, ABC):
    """DEPRECATED. Inherit from cube.benchmark.Benchmark for new benchmarks.

    This class is kept for backward compatibility with MiniWob and WorkArena.
    New benchmarks should use cube.benchmark.Benchmark directly.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"{cls.__name__} inherits from agentlab2.benchmark.Benchmark which is deprecated. "
            "New benchmarks should inherit from cube.benchmark.Benchmark instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    metadata: dict = Field(default_factory=dict)
    tool_config: ToolConfig

    @abstractmethod
    def setup(self):
        """
        Perform common steps necessary to prepare the environment for all tasks,
        like running web server, launching containers, etc.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up resources after all tasks are done.
        """
        pass

    @abstractmethod
    def load_tasks(self) -> list[Task]:
        """DEPRECATED. Use cube.benchmark.Benchmark.task_metadata or get_task_configs() instead.

        This method is kept for backward compatibility with MiniWob and WorkArena.
        New benchmarks should define task_metadata as a ClassVar and use get_task_configs().
        """
        pass

    def env_configs(self) -> list[EnvConfig]:
        """DEPRECATED. Use cube.benchmark.Benchmark.get_task_configs() instead."""
        warnings.warn(
            "Benchmark.env_configs() is deprecated. "
            "New benchmarks should inherit from cube.benchmark.Benchmark and use get_task_configs().",
            DeprecationWarning,
            stacklevel=2,
        )
        tasks = self.load_tasks()
        configs = [EnvConfig(task=task, tool_config=self.tool_config) for task in tasks]
        return configs

