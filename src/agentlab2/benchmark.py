from abc import ABC, abstractmethod

from pydantic import Field

from agentlab2.core import Task, TypedBaseModel
from agentlab2.environment import EnvConfig
from agentlab2.tool import ToolConfig


class Benchmark(TypedBaseModel, ABC):
    """Represents a benchmark consisting of multiple tasks and an environment."""

    metadata: dict = Field(default_factory=dict)
    tool_config: ToolConfig
    n_attempts: int = 1
    debug_task_limit: int | None = None

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
        """
        Load and return the list of tasks for this benchmark.
        """
        pass

    def env_configs(self) -> list[EnvConfig]:
        """Generate environment configurations for all tasks in the benchmark."""
        tasks = self.load_tasks()
        if self.debug_task_limit is not None:
            tasks = tasks[:self.debug_task_limit]
        interleaved = [task for _ in range(self.n_attempts) for task in tasks]
        return [EnvConfig(task=task, tool_config=self.tool_config) for task in interleaved]

    def install(self):
        """
        Optional method to download and prepare any resources required by the benchmark.
        """
        pass

    def uninstall(self):
        """
        Optional method to remove any resources used by the benchmark.
        """
        pass
