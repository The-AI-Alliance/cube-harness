"""LLM interaction abstractions, LiteLLM based."""

import pprint
from datetime import datetime
from functools import partial
from typing import Callable, List, Literal
from uuid import uuid4

from litellm import Message, completion_with_retries
from litellm.utils import token_counter
from pydantic import Field

from agentlab2.base import TypedBaseModel

# NOTE: Do not set litellm.callbacks = ["otel"] here at module level.
# When no TracerProvider is configured, litellm falls back to ConsoleSpanExporter
# which dumps huge JSON span dicts to stdout. Instead, enable the callback only
# after a proper TracerProvider has been set up (see metrics/tracer.py).


class Prompt(TypedBaseModel):
    """Represents the input prompt to chat completion api of LLM."""

    messages: List[dict | Message]
    tools: List[dict] = Field(default_factory=list)

    def __str__(self) -> str:
        """Debug view of the prompt."""
        messages = "\n".join([f"[{i}]{m}" for i, m in enumerate(self.messages)])
        tools = pprint.pformat(self.tools, width=120)
        return f"Tools:\n{tools}\nMessages[{len(self.messages)}]:\n{messages}"


class LLMConfig(TypedBaseModel):
    """Thin LLM wrapper around LiteLLM completion API."""

    model_name: str
    api_base: str | None = None
    api_key: str | None = None
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


class Usage(TypedBaseModel):
    """Token usage information from LLM response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0  # tokens read from cache (cache hit)
    cache_creation_tokens: int = 0  # tokens written to cache (Anthropic)
    cost: float = 0.0  # cost in USD from LiteLLM pricing


class LLMResponse(TypedBaseModel):
    """Response from LLM containing message and usage info."""

    message: Message
    usage: Usage


class LLM:
    def __init__(self, config: LLMConfig):
        self.config = config

    def __call__(self, prompt: Prompt) -> LLMResponse:
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
        if self.config.api_base is not None:
            kwargs["api_base"] = self.config.api_base
        if self.config.api_key is not None:
            kwargs["api_key"] = self.config.api_key
        if self.config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        response = completion_with_retries(**kwargs)
        usage = self._extract_usage(response)
        return LLMResponse(message=response.choices[0].message, usage=usage)

    def _extract_usage(self, response) -> Usage:
        """Extract usage info from LiteLLM response."""
        usage_data = getattr(response, "usage", None)
        if usage_data is None:
            return Usage()

        def safe_int(value: object) -> int:
            """Safely convert a value to int, returning 0 for non-numeric types."""
            if isinstance(value, int):
                return value
            return 0

        def safe_float(value: object) -> float:
            """Safely convert a value to float, returning 0.0 for non-numeric types."""
            if isinstance(value, (int, float)):
                return float(value)
            return 0.0

        cached_tokens = 0
        cache_creation_tokens = 0

        # Check prompt_tokens_details for cached_tokens (OpenAI/Anthropic)
        prompt_details = getattr(usage_data, "prompt_tokens_details", None)
        if prompt_details:
            cached_tokens = safe_int(getattr(prompt_details, "cached_tokens", 0))

        # Anthropic-specific fields
        cache_creation_tokens = safe_int(getattr(usage_data, "cache_creation_input_tokens", 0))
        cache_read = safe_int(getattr(usage_data, "cache_read_input_tokens", 0))
        if cache_read > 0:
            cached_tokens = cache_read  # Anthropic uses this field name

        # Extract cost from LiteLLM's hidden params
        cost = 0.0
        hidden_params = getattr(response, "_hidden_params", {})
        if isinstance(hidden_params, dict):
            cost = safe_float(hidden_params.get("response_cost", 0.0))

        return Usage(
            prompt_tokens=safe_int(getattr(usage_data, "prompt_tokens", 0)),
            completion_tokens=safe_int(getattr(usage_data, "completion_tokens", 0)),
            total_tokens=safe_int(getattr(usage_data, "total_tokens", 0)),
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cost=cost,
        )


class LLMCall(TypedBaseModel):
    """Represents a call to an LLM model."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_config: LLMConfig
    prompt: Prompt
    output: Message
    usage: Usage = Field(default_factory=Usage)
