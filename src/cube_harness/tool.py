import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from cube.core import Action, ActionSchema, Content, Observation, StepError
from cube.tool import Tool

from cube_harness.metrics.tracer import GEN_AI_TOOL_CALL_RESULT, tool_span

logger = logging.getLogger(__name__)

# AsyncTool and its dependencies were added to cube-standard after the last PyPI
# release (0.1.0rc1). Define them locally until a new version is published.


class _AbstractAsyncTool(ABC):
    async def reset(self) -> None:
        pass

    async def close(self) -> None:
        pass

    @abstractmethod
    async def execute_action(self, action: Action) -> Any:
        pass

    @property
    @abstractmethod
    def action_set(self) -> list[ActionSchema]:
        pass


class _ToolActionsMixin:
    def get_action_method(self, action: Action) -> Callable:
        method = self.__dict__.get(action.name)
        if method and callable(method) and getattr(method, "_is_action", False):
            return method
        method = getattr(self, action.name, None)
        if not method:
            raise ValueError(f"Action {action.name} does not exist in {self.__class__.__name__}.")
        is_registered = any(
            getattr(cls.__dict__.get(action.name), "_is_action", False)
            for cls in type(self).__mro__
            if action.name in cls.__dict__
        )
        if not is_registered:
            raise ValueError(
                f"Action {action.name} exists in {self.__class__.__name__} but is not decorated with @tool_action."
            )
        return method

    @property
    def action_set(self) -> list[ActionSchema]:
        actions = []
        for attr_name in dir(self):
            if attr_name.startswith("_") or attr_name == "action_set":
                continue
            if any(
                isinstance(cls.__dict__.get(attr_name), property)
                for cls in type(self).__mro__
                if attr_name in cls.__dict__
            ):
                continue
            attr = getattr(self, attr_name)
            is_action = any(
                getattr(cls.__dict__.get(attr_name), "_is_action", False)
                for cls in type(self).__mro__
                if attr_name in cls.__dict__
            )
            if callable(attr) and is_action:
                actions.append(ActionSchema.from_function(attr))
        for name, attr in self.__dict__.items():
            if not name.startswith("_") and callable(attr) and getattr(attr, "_is_action", False):
                actions.append(ActionSchema.from_function(attr))
        return actions


class AsyncTool(_ToolActionsMixin, _AbstractAsyncTool):
    async def execute_action(self, action: Action) -> Observation | StepError:
        method = self.get_action_method(action)
        try:
            action_result = (await method(**action.arguments)) or "Success"
        except Exception as e:
            logger.exception(f"Error executing action {action.name}: {e}")
            return StepError.from_exception(e)
        return Observation(contents=[Content.from_data(action_result, tool_call_id=action.id)])


class ToolWithTelemetry(Tool):
    """Tool subclass that wraps execute_action with OpenTelemetry tracing.

    Subclasses must override _execute_action instead of execute_action so that
    the telemetry span always wraps the complete execution, including any
    subclass-specific post-processing (e.g. appending page observations).
    """

    def execute_action(self, action: Action) -> Observation | StepError:
        with tool_span(action) as span:
            result = self._execute_action(action)
            if isinstance(result, StepError):
                result_str = f"Error executing action {action.name}: {result.exception_str}"
            else:
                result_str = str(result.contents[0].data)
            span.set_attribute(GEN_AI_TOOL_CALL_RESULT, result_str)
        return result

    def _execute_action(self, action: Action) -> Observation | StepError:
        """Override this in subclasses instead of execute_action."""
        return super().execute_action(action)


class AsyncToolWithTelemetry(AsyncTool):
    """AsyncTool subclass that wraps execute_action with OpenTelemetry tracing.

    Subclasses must override _execute_action instead of execute_action so that
    the telemetry span always wraps the complete execution, including any
    subclass-specific post-processing (e.g. appending page observations).
    """

    async def execute_action(self, action: Action) -> Observation | StepError:
        with tool_span(action) as span:
            result = await self._execute_action(action)
            if isinstance(result, StepError):
                result_str = f"Error executing action {action.name}: {result.exception_str}"
            else:
                result_str = str(result.contents[0].data)
            span.set_attribute(GEN_AI_TOOL_CALL_RESULT, result_str)
        return result

    async def _execute_action(self, action: Action) -> Observation | StepError:
        """Override this in subclasses instead of execute_action."""
        return await super().execute_action(action)
