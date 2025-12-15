from abc import ABC, abstractmethod
from typing import Sequence

from pydantic import BaseModel, Field

from agentlab2.environment import EnvironmentConfig


class Benchmark(BaseModel, ABC):
    """Represents a benchmark consisting of multiple tasks and an environment."""

    metadata: dict = Field(default_factory=dict)
    env_config: EnvironmentConfig

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
    def env_configs(self) -> Sequence[EnvironmentConfig]:
        """Return the list of environment configurations for each task in the benchmark."""
        pass

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
