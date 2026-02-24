"""Environment, Benchmark and Task abstractions."""

from abc import ABC, abstractmethod
from typing import List

from agentlab2.core import Action, ActionSchema, EnvironmentOutput, Observation, Task
from agentlab2.tool import AbstractTool, ToolConfig

STOP_ACTION = ActionSchema(name="final_step", description="Stop the task execution.")


class EnvConfig:
    """Runtime configuration for the Environment."""

    def __init__(self, task: "Task", tool_config: ToolConfig) -> None:
        self.task = task
        self.tool_config = tool_config

    def make(self) -> "Environment":
        tool = self.tool_config.make()
        return Environment(self.task, tool)


class AbstractEnvironment(ABC):
    """Abstract interface for environments that agents interact with."""

    def __init__(self, task: "Task", *args, **kwargs) -> None:
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
    """Environment that encapsulates a tool for interaction and a task."""

    def __init__(self, task: Task, tool: AbstractTool):
        self.task = task
        self.tool = tool

    @property
    def action_set(self) -> list[ActionSchema]:
        all_actions = self.tool.action_set
        return self.task.filter_actions(all_actions)

    def setup(self) -> EnvironmentOutput:
        """Prepare tool and set up the task."""
        obs, info = self.task.setup(self.tool)
        obs = self.task.obs_postprocess(obs)
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
