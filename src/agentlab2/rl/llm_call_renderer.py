import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from litellm import Message
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from agentlab2.llm import LLMCall, Prompt

_APRIEL_MODEL_ID = "ServiceNow-AI/Apriel-1.6-15b-Thinker"


@dataclass
class TextPair:
    """A single training sample from an LLM call."""

    prompt_text: str
    response_text: str
    images: list[str] = field(default_factory=list)
    reward: float | None = None


def llm_call_to_text_pair(llm_call: LLMCall, step_reward: float | None) -> TextPair:
    """Convert an LLMCall to a (prompt, response, images) tuple.

    Returns:
        Tuple of (prompt_text, response_text, images) where images is a list of
        base64 data-URL strings in the order they appear in the prompt (matching [IMG] tokens).
    """
    model_name = llm_call.llm_config.model_name
    if model_name.endswith("Apriel-1.6-15b-Thinker"):
        return _render_apriel_thinker(llm_call.prompt, llm_call.output, step_reward)
    raise NotImplementedError(f"LLM call rendering not implemented for model {model_name}")


def _render_apriel_thinker(prompt: Prompt, output: Message, step_reward: float | None) -> TextPair:
    """Render Apriel Thinker LLM call into input/output text pair using the official chat template."""
    tokenizer = _load_apriel_tokenizer()
    messages = _normalize_messages(prompt.messages)
    text_in: str = tokenizer.apply_chat_template(
        messages,
        tools=prompt.tools or None,  # type: ignore
        tokenize=False,
        add_generation_prompt=True,
    )
    text_out = _render_apriel_output(output)
    images = _extract_images(messages)

    return TextPair(
        prompt_text=text_in,
        response_text=text_out,
        images=images,
        reward=step_reward,
    )


def _normalize_messages(messages: list[dict | Message]) -> list[dict[str, Any]]:
    """Convert messages to plain dicts so Jinja templates can access keys safely.

    LiteLLM Message objects cause errors in Jinja templates (AttributeError vs KeyError
    for missing keys). Multimodal content (image_url) is preserved — the Apriel template
    renders it as [IMG] tokens.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, Message):
            result.append(msg.model_dump(exclude_none=True))
        else:
            result.append(dict(msg))
    return result


def _extract_images(messages: list[dict[str, Any]]) -> list[str]:
    """Extract base64 image data-URLs from multimodal messages, in order.

    Each image corresponds to an [IMG] token in the rendered chat template.
    """
    images: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") in ("image", "image_url"):
                url = chunk.get("image_url", {}).get("url", "")
                if url:
                    images.append(url)
    return images


def _render_apriel_output(output: Message) -> str:
    """Render the assistant output message in Apriel chat format.

    Reconstructs the raw model output from the structured response fields:
    - reasoning_content: thinking text before [BEGIN FINAL RESPONSE]
    - content: response text after [BEGIN FINAL RESPONSE]
    - tool_calls: structured tool calls (rendered without IDs, matching template's last-message behavior)
    """
    parts: list[str] = []

    reasoning = getattr(output, "reasoning_content", None) or ""
    if reasoning:
        parts.append(reasoning)
        parts.append("\n[BEGIN FINAL RESPONSE]")

    content = getattr(output, "content", None) or ""
    if content:
        if reasoning:
            parts.append("\n")
        parts.append(content)

    tool_calls = getattr(output, "tool_calls", None)
    if tool_calls:
        tc_entries = []
        for tc in tool_calls:
            args = tc.function.arguments
            if isinstance(args, dict):
                args = json.dumps(args)
            tc_entries.append(f'{{"name": "{tc.function.name}", "arguments": {args}}}')
        parts.append(f"\n<tool_calls>[{', '.join(tc_entries)}]</tool_calls>")

    parts.append("\n<|end|>\n")

    return "".join(parts)


@lru_cache(maxsize=1)
def _load_apriel_tokenizer() -> PreTrainedTokenizerFast:
    """Load the Apriel Thinker tokenizer (cached in memory after first load)."""
    return AutoTokenizer.from_pretrained(_APRIEL_MODEL_ID)
