"""LLM interaction abstractions, LiteLLM based."""

import pprint
from datetime import datetime
from functools import partial
from typing import Any, Callable, List, Literal
from uuid import uuid4

from litellm import Message, completion_with_retries, get_llm_provider
from litellm.utils import token_counter
from opentelemetry import trace
from pydantic import Field

from agentlab2.base import TypedBaseModel
from agentlab2.metrics.genai import chat_span, set_output_messages

UNKNOWN_PROVIDER = "unknown (see litellm.get_llm_provider)"


def _extract_provider_and_model(model_name: str) -> tuple[str, str]:
    try:
        extracted_model, provider, _, _ = get_llm_provider(model_name)
        return provider, extracted_model
    except Exception:
        return UNKNOWN_PROVIDER, model_name


def _trace_llm_call(
    config: "LLMConfig",
    response: Any,
) -> None:
    provider, model_name = _extract_provider_and_model(config.model_name)
    span = trace.get_current_span()

    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.provider.name", provider)
    span.set_attribute("gen_ai.request.model", model_name)
    span.set_attribute("gen_ai.request.temperature", config.temperature)
    span.set_attribute("gen_ai.request.max_tokens", config.max_completion_tokens)

    if hasattr(response, "id") and response.id:
        span.set_attribute("gen_ai.response.id", response.id)
    if hasattr(response, "model") and response.model:
        span.set_attribute("gen_ai.response.model", response.model)
    if hasattr(response, "usage") and response.usage:
        if hasattr(response.usage, "prompt_tokens"):
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        if hasattr(response.usage, "completion_tokens"):
            span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)


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

    def _call_completion(self, prompt: Prompt) -> Any:
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
        return completion_with_retries(**kwargs)

    def __call__(self, prompt: Prompt) -> Message:
        with chat_span(self.config.model_name, prompt.messages) as span:
            response = self._call_completion(prompt)
            msg = response.choices[0].message
            _trace_llm_call(self.config, response)
            set_output_messages(span, msg)
            return msg


class LLMCall(TypedBaseModel):
    """Represents a call to an LLM model."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_config: LLMConfig
    prompt: Prompt
    output: Message
