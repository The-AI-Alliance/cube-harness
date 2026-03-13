"""Agent abstraction."""

from abc import ABC, abstractmethod

from cube.core import ActionSchema, Observation, TypedBaseModel

from cube_harness.core import AgentOutput


class AgentConfig(TypedBaseModel, ABC):
    """Configuration for creating an Agent."""

    @abstractmethod
    def make(self, action_set: list[ActionSchema] | None = None, **kwargs) -> "Agent":
        pass


class Agent(ABC):
    name: str
    description: str
    input_content_types: list[str]
    output_content_types: list[str]

    def __init__(self, config: AgentConfig):
        self.config = config

    @abstractmethod
    def step(self, obs: Observation) -> AgentOutput:
        """
        Perform a step given an observation and return the agent's output with actions.
        """
        pass

    def __repr__(self) -> str:
        return self.config.model_dump_json(indent=2, serialize_as_any=True)
