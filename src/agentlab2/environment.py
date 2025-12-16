"""Environment, Benchmark and Task abstractions."""

from abc import ABC, abstractmethod
from typing import Callable, List, TypeVar

from agentlab2.core import Action, ActionSchema, AL2BaseModel, EnvironmentOutput, Observation
from agentlab2.tool import AbstractTool, AbstractToolConfig

STOP_ACTION = ActionSchema(name="final_step", description="Stop the task execution.")


class EnvironmentConfig(AL2BaseModel, ABC):
    """Configuration for Environment."""

    _task: "Task | None" = None

    @abstractmethod
    def make(self) -> "Environment":
        pass


class Environment(ABC):
    """Base class for environments that agents interact with."""

    def __init__(self, task: "Task", *args, **kwargs) -> None:
        super().__init__()
        self.task: Task = task

    @abstractmethod
    def setup(self) -> EnvironmentOutput:
        """Set up the environment before starting a task."""
        pass

    def action_set(self) -> List[ActionSchema]:
        """Returns list of actions supported by that environment."""
        return []

    @abstractmethod
    def step(self, action: Action | list[Action]) -> EnvironmentOutput:
        """Execute a single or multiple actions and return the observation."""
        pass

    def close(self) -> None:
        """Optional clean up environment resources."""
        pass


class Task(ABC):
    """Represents a task that an agent must complete in an environment."""

    id: str
    validate_per_step: bool = False
    supported_actions: tuple[Callable, ...]

    @abstractmethod
    def setup(self, env: Environment) -> tuple[Observation, dict]:
        """
        Set up the task in the given environment.

        Returns:
            Tuple of (Observation, dict with additional task info)
        """
        pass

    def teardown(self, env: Environment) -> None:
        """Optional clean up after task completion."""
        pass

    @abstractmethod
    def validate_task(self, env: Environment, obs: Observation) -> tuple[float, dict]:
        """Validate the whole trajectory and state of the env at the end of the run."""
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

    def finished(self, env: Environment) -> bool:
        """Check if the task is finished."""
        return False


T = TypeVar("T", bound=AbstractTool)


class ToolboxConfig(EnvironmentConfig):
    """Configuration for ToolboxEnv."""

    tool_configs: list[AbstractToolConfig] = []

    def make(self) -> "ToolboxEnv":
        assert self._task is not None, "Task must be set in the EnvironmentConfig before making the environment."
        tools = [tc.make() for tc in self.tool_configs]
        return ToolboxEnv(task=self._task, tools=tools)


class ToolboxEnv(Environment):
    """Environment that uses a collection of tools for interaction."""

    def __init__(self, task: Task, tools: list[AbstractTool]):
        self.task = task
        self.tools = tools
        self._action_name_to_tool = {action.name: tool for tool in tools for action in tool.action_set()}

    def action_set(self) -> list[ActionSchema]:
        """Returns list of actions supported by that environment, union of all tool actions, filtered by the task."""
        actions_union = [action for tool in self.tools for action in tool.action_set()]
        return self.task.filter_actions(actions_union)

    def setup(self) -> EnvironmentOutput:
        """Prepare all tools and set up the task."""
        for tool in self.tools:
            tool.reset()
        obs, info = self.task.setup(self)
        return EnvironmentOutput(obs=obs, info=info)

    def step(self, action: Action | list[Action]) -> EnvironmentOutput:
        """Execute a single or multiple actions using the appropriate tools, combine observations."""
        actions = [action] if isinstance(action, Action) else action
        done = False
        reward = 0.0
        info = {}
        tool_results: list[Observation] = []
        for action in actions:
            if self.is_stop_action(action):
                tool_results.append(Observation.from_text("Task finished by the agent."))
                done = True
                break
            if action.name not in self._action_name_to_tool:
                raise ValueError(f"Action '{action.name}' is not supported by any tool in this environment.")
            tool = self._action_name_to_tool[action.name]
            tool_results.append(tool.execute_action(action))
        obs = Observation(contents=[c for o in tool_results for c in o.contents])
        done = done or self.task.finished(self)
        if self.task.validate_per_step or done:
            reward, info = self.task.validate_task(self, obs)
        obs = self.task.obs_postprocess(obs)
        return EnvironmentOutput(obs=obs, reward=reward, info=info, done=done)

    def find_tool(self, tool_cls: type[T]) -> T | None:
        """Find a tool of the given class in the environment."""
        for tool in self.tools:
            if isinstance(tool, tool_cls):
                return tool
        return None

    def is_stop_action(self, action: Action) -> bool:
        """Check if the action is the stop action."""
        return action.name == STOP_ACTION.name

    def close(self):
        """Clean up resources used by all tools and the task in the right order."""
        self.task.teardown(self)
        for tool in self.tools:
            tool.close()
