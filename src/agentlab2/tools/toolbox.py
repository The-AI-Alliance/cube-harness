from typing import TypeVar

from cube.core import Action, ActionSchema, Observation
from cube.tool import AbstractTool

from cube.tool import ToolConfig
from agentlab2.tool import Tool


class ToolboxConfig(ToolConfig):
    """Configuration for ToolEnv."""

    tool_configs: list[ToolConfig] = []

    def make(self, container=None) -> "Toolbox":
        tools = [tc.make(container) for tc in self.tool_configs]
        return Toolbox(tools=tools)


T = TypeVar("T", bound=AbstractTool)


class Toolbox(Tool):
    """Composite tool that uses multiple tools for interaction."""

    def __init__(self, tools: list[AbstractTool]):
        self.tools = tools
        self._action_name_to_tool = {action.name: tool for tool in tools for action in tool.action_set}

    @property
    def action_set(self) -> list[ActionSchema]:
        """Returns list of actions supported by that environment, union of all tool actions, filtered by the task."""
        actions_union = [action for tool in self.tools for action in tool.action_set]
        return actions_union

    def reset(self):
        """Prepare all tools and set up the task."""
        for tool in self.tools:
            tool.reset()

    def execute_action(self, action: Action) -> Observation:
        """Find the appropriate tool for the action and execute it."""
        if action.name not in self._action_name_to_tool:
            raise ValueError(f"Action '{action.name}' is not supported by any tool in the toolbox.")
        tool = self._action_name_to_tool[action.name]
        return tool.execute_action(action)

    def find_tool(self, tool_cls: type[T]) -> T | None:
        """Find a tool of the given class in the environment."""
        for tool in self.tools:
            if isinstance(tool, tool_cls):
                return tool
        return None

    def close(self):
        """Clean up resources used by all tools and the task in the right order."""
        for tool in self.tools:
            tool.close()
