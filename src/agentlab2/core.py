from abc import ABC, abstractmethod
from typing import Any, Callable, Self

from pydantic import Field

from agentlab2.llm import LLMCall
from cube.core import Action, ActionSchema, Content, StepError, TypedBaseModel


class AgentOutput(TypedBaseModel):
    actions: list[Action] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)
    error: StepError | None = None

    def __str__(self) -> str:
        return self.model_dump_json(exclude={"llm_calls"})



class Observation(TypedBaseModel):
    """Represents an observation from the environment."""

    contents: list[Content] = Field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> Self:
        return cls(contents=[Content.from_data(text)])

    def to_llm_messages(self) -> list[dict]:
        """Convert observation to a list of messages suitable for sending to LLM."""
        return [content.to_llm_message() for content in self.contents]

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
