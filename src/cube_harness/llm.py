"""LLM interaction abstractions, LiteLLM based."""

import pprint
from datetime import datetime
from functools import partial
from typing import Any, Callable, List, Literal
from uuid import uuid4

import litellm
import tenacity
from cube.core import TypedBaseModel
from litellm import Message
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.utils import token_counter
from pydantic import Field

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
    temperature: float = 1.0
    max_tokens: int = 128000
    max_completion_tokens: int = 8192
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    tool_choice: Literal["auto", "none", "required"] = "auto"
    parallel_tool_calls: bool = False
    num_retries: int = 5
    retry_strategy: Literal["exponential_backoff_retry", "constant_retry"] = "exponential_backoff_retry"
    timeout: float | None = 120.0  # seconds per attempt; None = no timeout
    # Anthropic prompt caching. "auto" places ephemeral cache_control breakpoints at the
    # system message and the last assistant message, plus the last tool definition. This
    # gives a stable anchor (system + tools) and a rolling boundary (last assistant) that
    # extends across steps as the conversation grows. No-op for non-Anthropic models.
    set_cache_control: Literal["auto"] | None = None

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


def _is_anthropic_model(model_name: str) -> bool:
    """Heuristic: does this model route to Anthropic via LiteLLM?

    Covers direct Anthropic, Bedrock-Anthropic, and Vertex-Claude routings. Used to
    gate Anthropic-specific cache_control payloads so they don't leak to other providers.
    """
    name = model_name.lower()
    return "claude" in name or "anthropic" in name


def _msg_role(msg: Any) -> str | None:
    if isinstance(msg, dict):
        return msg.get("role")
    return getattr(msg, "role", None)


def _build_cache_injection_points(messages: list) -> list[dict]:
    """Return ephemeral cache_control breakpoints: first system msg + last assistant msg.

    Two breakpoints anchor a static prefix (system + tools above it) and a rolling
    boundary that naturally extends each step as new assistant turns are appended.
    Anthropic's longest-prefix match handles cross-step cache hits.
    """
    points: list[dict] = []
    control = {"type": "ephemeral"}
    for i, msg in enumerate(messages):
        if _msg_role(msg) == "system":
            points.append({"location": "message", "index": i, "control": control})
            break
    for i in range(len(messages) - 1, -1, -1):
        if _msg_role(messages[i]) == "assistant":
            if not any(p["index"] == i for p in points):
                points.append({"location": "message", "index": i, "control": control})
            break
    return points


def _mark_last_tool_for_cache(tools: list[dict]) -> list[dict]:
    """Return a copy of tools with ephemeral cache_control on the last entry.

    Caches the entire tools array prefix on Anthropic. LiteLLM passes the
    cache_control field through to the Anthropic API.
    """
    if not tools:
        return tools
    result = [dict(t) for t in tools]
    result[-1] = {**result[-1], "cache_control": {"type": "ephemeral"}}
    return result


class LLM:
    def __init__(self, config: LLMConfig):
        self.config = config

    def __call__(self, prompt: Prompt) -> LLMResponse:
        tools = prompt.tools
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_completion_tokens,
            "tool_choice": self.config.tool_choice,
            "parallel_tool_calls": self.config.parallel_tool_calls,
            "messages": prompt.messages,
            "timeout": self.config.timeout,
        }
        if self.config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        if self.config.set_cache_control == "auto" and _is_anthropic_model(self.config.model_name):
            injection_points = _build_cache_injection_points(prompt.messages)
            if injection_points:
                kwargs["cache_control_injection_points"] = injection_points
            tools = _mark_last_tool_for_cache(tools)
        if tools:
            kwargs["tools"] = tools
        if not tools or self.config.tool_choice is None:
            # Drop tool_choice / parallel_tool_calls when there are no tools (some providers
            # reject tool_choice without a tools list) or when the caller opted out (None).
            kwargs.pop("tool_choice", None)
            kwargs.pop("parallel_tool_calls", None)
        response = self._completion_with_retry(**kwargs)
        usage = self._extract_usage(response)
        return LLMResponse(message=response.choices[0].message, usage=usage)

    def _completion_with_retry(self, **kwargs: Any) -> Any:
        """Call litellm.completion with exponential backoff on transient errors.

        litellm's completion_with_retries caps its backoff at 10 s, which is too
        short for Anthropic overloaded_error responses under heavy load. We own the
        retry loop here to get a proper 120 s ceiling.
        """
        _RETRIABLE = (
            InternalServerError,
            ServiceUnavailableError,
            RateLimitError,
            Timeout,
            APIConnectionError,
        )
        retryer = tenacity.Retrying(
            wait=tenacity.wait_exponential(multiplier=2, max=120),
            stop=tenacity.stop_after_attempt(self.config.num_retries),
            retry=tenacity.retry_if_exception_type(_RETRIABLE),
            reraise=True,
        )
        return retryer(litellm.completion, **kwargs)

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

    id: str = Field(default_factory=lambda: uuid4().hex)  # unique storage key
    tag: str = ""  # optional label shown as tab name in viewers (e.g. "act", "summary")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_config: LLMConfig
    prompt: Prompt
    output: Message
    usage: Usage = Field(default_factory=Usage)
