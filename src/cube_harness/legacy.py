"""
This file contains all legacy classes that are kept for backward compatibility with MiniWob and WorkArena.
New benchmarks should not use any of these classes and should instead use the new classes defined in cube.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, List

from cube.core import Action, ActionSchema, EnvironmentOutput, Observation, TypedBaseModel
from cube.task import STOP_ACTION
from cube.tool import AbstractTool, ToolConfig
from pydantic import Field


class Task(ABC):
    """DEPRECATED. Inherit from cube.task.Task for new benchmarks.

    This class is kept for backward compatibility with MiniWob and WorkArena.
    New benchmarks should use cube.task.Task directly.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"{cls.__name__} inherits from cube_harness.core.Task which is deprecated. "
            "New benchmarks should inherit from cube.task.Task instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    id: str
    validate_per_step: bool = False
    _tool: Any  # access to the environment tool, initialized in setup()

    @abstractmethod
    def setup(self, tool: Any) -> tuple[Observation, dict]:
        """
        Set up the task in the given environment.

        Returns:
            Tuple of (Observation, dict with additional task info)
        """
        pass

    def teardown(self) -> None:
        """Optional clean up after task completion."""
        pass

    @abstractmethod
    def validate_task(self, obs: Observation) -> tuple[float, dict]:
        """Validate the current state of the task and return (reward, info)."""
        pass

    @abstractmethod
    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        """Allows the task to whitelist subset of all the actions provided by the environment."""
        pass

    def cheat(self):
        """
        Solve the task using a pre-defined solution (optional).
        """
        raise NotImplementedError

    def obs_postprocess(self, obs: Observation) -> Observation:
        """Optional post-processing of observation before returning it to the agent."""
        return obs

    def finished(self) -> bool:
        """Check if the task is finished."""
        return False

    def accept_agent_stop(self) -> bool:
        """Optional, whether the task accepts the agent stopping the task right now. Default is True."""
        return True


class EnvConfig:
    """DEPRECATED. Use cube.task.TaskConfig instead.

    This class is kept for backward compatibility with MiniWob and WorkArena.
    New benchmarks should use cube.task.TaskConfig directly.
    """

    def __init__(self, task: "Task", tool_config: ToolConfig) -> None:
        warnings.warn(
            "EnvConfig is deprecated. Use cube.task.TaskConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.task = task
        self.tool_config = tool_config

    def make(self) -> "Environment":
        tool = self.tool_config.make()
        return Environment(self.task, tool)


class AbstractEnvironment(ABC):
    """DEPRECATED. Use cube.task.Task instead.

    This class is kept for backward compatibility with MiniWob and WorkArena.
    New benchmarks should use cube.task.Task directly.
    """

    def __init__(self, task: "Task", *args, **kwargs) -> None:
        warnings.warn(
            "AbstractEnvironment is deprecated. Use cube.task.Task instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self.task: "Task" = task

    @abstractmethod
    def setup(self) -> EnvironmentOutput:
        """Set up the environment before starting a task."""
        pass

    @property
    @abstractmethod
    def action_set(self) -> List[ActionSchema]:
        """Returns list of actions supported by that environment."""
        pass

    @abstractmethod
    def step(self, action: Action | list[Action]) -> EnvironmentOutput:
        """Execute a single or multiple actions and return the observation."""
        pass

    def close(self) -> None:
        """Optional clean up environment resources."""
        pass


class Environment(AbstractEnvironment):
    """DEPRECATED. Use cube.task.Task instead.

    This class is kept for backward compatibility with MiniWob and WorkArena.
    New benchmarks should use cube.task.Task directly.
    """

    def __init__(self, task: Task, tool: AbstractTool):
        warnings.warn(
            "Environment is deprecated. Use cube.task.Task instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.task = task
        self.tool = tool

    @property
    def action_set(self) -> list[ActionSchema]:
        all_actions = self.tool.action_set
        return self.task.filter_actions(all_actions)

    def setup(self) -> EnvironmentOutput:
        """Prepare tool and set up the task."""
        obs, info = self.task.setup(self.tool)
        return EnvironmentOutput(obs=obs, info=info)

    def step(self, action: Action | list[Action]) -> EnvironmentOutput:
        """Execute a single or multiple actions using the appropriate tools, combine observations."""
        actions = [action] if isinstance(action, Action) else action
        done = False
        reward = 0.0
        info = {}
        tool_results: list[Observation] = []
        for action in actions:
            if action.name == STOP_ACTION.name and self.task.accept_agent_stop():
                tool_results.append(Observation.from_text("Task finished by the agent."))
                done = True
                break
            tool_results.append(self.tool.execute_action(action))
        obs = Observation(contents=[c for o in tool_results for c in o.contents])
        done = done or self.task.finished()
        if self.task.validate_per_step or done:
            reward, info = self.task.validate_task(obs)
        obs = self.task.obs_postprocess(obs)
        return EnvironmentOutput(obs=obs, reward=reward, info=info, done=done)

    def close(self):
        """Clean up resources used by all tools and the task in the right order."""
        self.task.teardown()
        self.tool.close()


class Benchmark(TypedBaseModel, ABC):
    """DEPRECATED. Inherit from cube.benchmark.Benchmark for new benchmarks.

    This class is kept for backward compatibility with MiniWob and WorkArena.
    New benchmarks should use cube.benchmark.Benchmark directly.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"{cls.__name__} inherits from cube_harness.benchmark.Benchmark which is deprecated. "
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
