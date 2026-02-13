# We emit our own gen_ai.input/output.messages spans instead of relying on LiteLLM's
# OTEL callback because of three gaps (pinned to BerriAI/litellm@ce3bb97d40):
#
# 1. Span name: LiteLLM uses "litellm_request" instead of "chat {model}"
#    https://github.com/BerriAI/litellm/blob/ce3bb97d40/litellm/integrations/opentelemetry.py#L60
#    per OTEL GenAI semconv Inference section:
#    https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#inference
#
# 2. Reasoning content: LiteLLM's _transform_messages_to_otel_semantic_conventions
#    drops reasoning_content (all providers — LiteLLM normalizes Anthropic thinking_blocks
#    into reasoning_content in anthropic/chat/handler.py:_handle_reasoning_content).
#    https://github.com/BerriAI/litellm/blob/ce3bb97d40/litellm/integrations/opentelemetry.py#L1733-L1774
#    OTEL semconv defines ReasoningPart (type: "reasoning") as a first-class part type:
#    https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-output-messages.json#ReasoningPart
#
# 3. Tool calls: LiteLLM passes tool_calls as a raw top-level key instead of
#    typed ToolCallRequestPart entries inside the parts array.
#    https://github.com/BerriAI/litellm/blob/ce3bb97d40/litellm/integrations/opentelemetry.py#L1768
#    https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-output-messages.json#ToolCallRequestPart

from contextlib import contextmanager
from typing import Any, Iterator, Literal

from litellm import Message
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator

_tracer = trace.get_tracer(__name__)


class _GenAITextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"] = "text"
    content: str


class _GenAIReasoningPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["reasoning"] = "reasoning"
    content: str


class _GenAIToolCallPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: str


_GenAIPart = _GenAITextPart | _GenAIReasoningPart | _GenAIToolCallPart


class _GenAIMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    parts: list[_GenAIPart]

    # Accepts OpenAI/LiteLLM message format and normalizes to OTEL GenAI format.
    # Handles: content (str or multimodal list), reasoning_content, and tool_calls.
    # LiteLLM normalizes all providers to reasoning_content (including Anthropic
    # thinking_blocks — see anthropic/chat/handler.py:_handle_reasoning_content).
    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> dict[str, Any]:
        if isinstance(data, BaseModel):
            data = data.model_dump()
        data = {"role": "assistant", "content": None, "reasoning_content": None, "tool_calls": None} | data
        parts: list[dict[str, Any]] = []

        if data["reasoning_content"]:
            parts.append({"type": "reasoning", "content": data["reasoning_content"]})

        content = data["content"]
        if isinstance(content, str) and content:
            parts.append({"type": "text", "content": content})
        elif isinstance(content, list):
            for item in content:
                if "text" in item:
                    parts.append({"type": "text", "content": item["text"]})
                else:
                    # intentionally drop images to avoid bloating span attributes
                    parts.append({"type": "text", "content": "[image]"})

        for tc in data["tool_calls"] or []:
            fn = tc["function"]
            parts.append({"type": "tool_call", "id": tc["id"], "name": fn["name"], "arguments": fn["arguments"]})

        return {"role": data["role"], "parts": parts}


_GenAIMessageList = TypeAdapter(list[_GenAIMessage])


@contextmanager
def chat_span(model: str, messages: list[dict | Message]) -> Iterator[trace.Span]:
    with _tracer.start_as_current_span(f"chat {model}") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model)
        validated = _GenAIMessageList.validate_python(messages)
        span.set_attribute("gen_ai.input.messages", _GenAIMessageList.dump_json(validated).decode())
        yield span


def set_output_messages(span: trace.Span, msg: Message) -> None:
    output = _GenAIMessage.model_validate(msg)
    if output.parts:
        span.set_attribute("gen_ai.output.messages", _GenAIMessageList.dump_json([output]).decode())
