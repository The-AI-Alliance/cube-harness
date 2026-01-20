"""LLM interaction abstractions, LiteLLM based."""

import pprint
from datetime import datetime
from functools import partial
from typing import Callable, List, Literal
from uuid import uuid4

from litellm import Message, completion_with_retries
from litellm.utils import token_counter
from pydantic import Field, field_serializer

from agentlab2.base import TypedBaseModel


def _message_to_dict(msg: dict | Message) -> dict:
    """Convert a Message object to dict for serialization."""
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    return dict(msg)


class Prompt(TypedBaseModel):
    """Represents the input prompt to chat completion api of LLM."""

    messages: List[dict | Message]
    tools: List[dict] = Field(default_factory=list)

    @field_serializer("messages")
    def serialize_messages(self, messages: List[dict | Message]) -> List[dict]:
        """Convert Message objects to dicts for JSON serialization."""
        return [_message_to_dict(m) for m in messages]

    def __str__(self) -> str:
        """Debug view of the prompt."""
        messages = "\n".join([f"[{i}]{m}" for i, m in enumerate(self.messages)])
        tools = pprint.pformat(self.tools, width=120)
        return f"Tools:\n{tools}\nMessages[{len(self.messages)}]:\n{messages}"


class LLMConfig(TypedBaseModel):
    """Thin LLM wrapper around LiteLLM completion API."""

    model_name: str
    temperature: float = 1.0
    max_tokens: int = 128000
    max_completion_tokens: int = 8192
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    tool_choice: Literal["auto", "none", "required"] = "auto"
    parallel_tool_calls: bool = False
    num_retries: int = 5
    retry_strategy: Literal["exponential_backoff_retry", "constant_retry"] = "exponential_backoff_retry"

    def make(self) -> "LLM":
        """Create LLM instance from config."""
        return LLM(config=self)

    def make_counter(self) -> Callable[..., int]:
        """Get a token counter function for the LLM model."""
        return partial(token_counter, model=self.model_name)


class LLM:
    def __init__(self, config: LLMConfig):
        self.config = config

    def __call__(self, prompt: Prompt) -> Message:
        kwargs = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_completion_tokens,
            "num_retries": self.config.num_retries,
            "retry_strategy": self.config.retry_strategy,
            "tool_choice": self.config.tool_choice,
            "parallel_tool_calls": self.config.parallel_tool_calls,
            "tools": prompt.tools,
            "messages": prompt.messages,
        }
        if self.config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        response = completion_with_retries(**kwargs)
        return response.choices[0].message  # type: ignore


class LLMCall(TypedBaseModel):
    """Represents a call to an LLM model."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_config: LLMConfig
    prompt: Prompt
    output: dict | Message

    @field_serializer("output")
    def serialize_output(self, output: dict | Message) -> dict:
        """Convert Message to dict for JSON serialization."""
        return _message_to_dict(output)
