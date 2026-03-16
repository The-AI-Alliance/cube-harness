from typing import Callable

from cube.core import Action, EnvironmentOutput, StepError, TypedBaseModel
from pydantic import Field

from cube_harness.llm import LLMCall


class AgentOutput(TypedBaseModel):
    actions: list[Action] = Field(default_factory=list)
    # All LLM calls made during this step. Set LLMCall.tag to label each call (e.g. "act", "summary").
    llm_calls: list[LLMCall] = Field(default_factory=list)
    error: StepError | None = None
    # Maps label → (start_time, end_time) as absolute Unix timestamps.
    # Used by the XRay viewer to render a profiling breakdown inside each timeline segment.
    profiling: dict[str, tuple[float, float]] = Field(default_factory=dict)
    # Agent's chain-of-thought, rationale, or extended thinking for this step.
    thoughts: str | None = None

    def __str__(self) -> str:
        return self.model_dump_json(exclude={"llm_calls"})


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
