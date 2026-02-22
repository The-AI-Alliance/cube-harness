import base64
import io
import json
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Self

import litellm.utils
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from agentlab2.base import TypedBaseModel
from agentlab2.llm import LLMCall


class ActionSchema(TypedBaseModel):
    """
    Represents a function specification with a type, name, description and arguments.
    Compatible with OAI, Anthropic and VLLM definitions.

    Attributes:
        type (Literal["function"]): The type of the tool, which is always "function".
        name (str): The name of the function.
        description (str): A brief description of the function.
        parameters (dict): A dictionary containing the parameters of the function.
    """

    name: str
    description: str
    parameters: dict = Field(default_factory=dict)
    # examples: str | None = None # Maybe to-add

    @classmethod
    def from_function(cls, func: Callable) -> Self:
        """Create tool object from python function."""
        schema = litellm.utils.function_to_dict(func)
        return cls(**schema)

    def as_dict(self) -> dict[str, Any]:
        """Produce dict that could be passed as tool schema into LLM api."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Action(TypedBaseModel):
    """
    A class representing a function call.

    Attributes:
        id (str): The identifier for the tool call.
        name (str): The name of the function being called.
        arguments (Any): The arguments to be passed to the function.
    """

    id: str | None = None
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class StepError(TypedBaseModel):
    """Represents an error that occurred during a step execution."""

    error_type: str
    exception_str: str
    stack_trace: str

    @classmethod
    def from_exception(cls, exc: Exception) -> "StepError":
        """Create a StepError from an exception object."""
        return cls(
            error_type=type(exc).__name__,
            exception_str=str(exc),
            stack_trace="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )


class AgentOutput(TypedBaseModel):
    actions: list[Action] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)
    other_llm_calls: dict[str, list[LLMCall]] = Field(default_factory=dict)
    error: StepError | None = None
    # Maps label → (start_time, end_time) as absolute Unix timestamps.
    # Used by the XRay viewer to render a profiling breakdown inside each timeline segment.
    profiling: dict[str, tuple[float, float]] = Field(default_factory=dict)

    def __str__(self) -> str:
        return self.model_dump_json(exclude={"llm_calls", "other_llm_calls"})


_image_prefix = "data:image/png;base64,"


class Content(TypedBaseModel):
    """Represents a piece of content in an observation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    tool_call_id: str | None = None  # content could be result of a tool call
    name: str | None = None  # optional name of the content
    data: str | dict | list | BaseModel | Image.Image  # The actual content data

    @field_serializer("data")
    def serialize_data(self, data: str | Image.Image) -> str:
        if isinstance(data, str):
            return data
        image_str = self.as_base64_image_str(data)
        return image_str

    def as_base64_image_str(self, data):
        byte_arr = io.BytesIO()
        data.save(byte_arr, format="PNG")
        encoded_image = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
        return f"{_image_prefix}{encoded_image}"

    @field_validator("data", mode="before")
    @classmethod
    def deserialize_data(cls, v: str):
        if isinstance(v, str) and v.startswith(_image_prefix):
            v = v[len(_image_prefix) :]
            # Decode base64 string to bytes
            decoded_image = base64.b64decode(v)
            # Open bytes as PIL Image and load immediately to avoid lazy loading issues
            img = Image.open(io.BytesIO(decoded_image))
            img.load()  # Force load to prevent BytesIO buffer from being garbage collected
            return img
        return v  # Return original value if not a string (e.g., already an Image object)

    def to_message(self) -> dict:
        """Convert content to a message dict for LLM input."""
        if isinstance(self.data, Image.Image):
            image_base64 = self.as_base64_image_str(self.data)
            msg_content = [{"type": "image_url", "image_url": {"url": image_base64}}]
            if self.name:
                msg_content.insert(0, {"type": "text", "text": self.name})
        elif isinstance(self.data, BaseModel):
            msg_content = self.data.model_dump_json(serialize_as_any=True)
        elif isinstance(self.data, (dict, list)):
            msg_content = json.dumps(self.data)
            if self.name:
                msg_content = f"##{self.name}\n{msg_content}"
        else:
            msg_content = str(self.data)
            if self.name:
                msg_content = f"##{self.name}\n{msg_content}"
        role = "tool" if self.tool_call_id else "user"
        message = dict(role=role, content=msg_content)
        if self.tool_call_id:
            message["tool_call_id"] = self.tool_call_id
        return message


class Observation(TypedBaseModel):
    """Represents an observation from the environment."""

    contents: list[Content] = Field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> Self:
        return cls(contents=[Content(data=text)])

    def to_llm_messages(self) -> list[dict]:
        """Convert observation to a list of messages suitable for sending to LLM."""
        return [content.to_message() for content in self.contents]

    def __add__(self, other: Self) -> Self:
        self.contents += other.contents
        return self


class EnvironmentOutput(TypedBaseModel):
    """Represents the result of an environment step."""

    obs: Observation
    reward: float = 0.0
    done: bool = False
    info: dict = Field(default_factory=dict)
    error: StepError | None = None


class TrajectoryStep(TypedBaseModel):
    output: EnvironmentOutput | AgentOutput
    start_time: float | None = None
    end_time: float | None = None


class Trajectory(TypedBaseModel):
    """
    Stores history of the previous interaction.

    Metadata contains info about agent, env and task.
    reward_info represents episode level reward data.
    """

    id: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None
    reward_info: dict = Field(default_factory=dict)

    def last_env_step(self) -> EnvironmentOutput:
        for step in reversed(self.steps):
            if isinstance(step.output, EnvironmentOutput):
                return step.output
        raise ValueError("No EnvironmentOutput found in the trajectory.")


class ActionSpace(frozenset[Callable]):
    """A set of action callables representing a subset of an action space.

    Supports set operations (&, -, |) for composing action subsets.
    """

    def __new__(cls, *actions: Callable) -> "ActionSpace":
        return super().__new__(cls, actions)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(action.__name__ for action in self)


class Task(ABC):
    """Represents a task that an agent must complete in an environment."""

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
