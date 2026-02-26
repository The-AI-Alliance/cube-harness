from dataclasses import dataclass, field
from typing import Any

from litellm import Message
from transformers import AutoTokenizer

from agentlab2.llm import LLMCall

ALLOWED_MODELS = {
    "ServiceNow-AI/Apriel-1.6-15b-Thinker",
    "Qwen/Qwen3-4B-Thinking-2507",
}


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
    normalized_model_name = "/".join(model_name.split("/")[-2:])
    if normalized_model_name in ALLOWED_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(normalized_model_name)
        messages = _normalize_messages(llm_call.prompt.messages)
        add_generation_prompt = normalized_model_name != "ServiceNow-AI/Apriel-1.6-15b-Thinker"
        text_in: str = tokenizer.apply_chat_template(
            messages,
            tools=llm_call.prompt.tools or None,  # type: ignore
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        text_full = tokenizer.apply_chat_template(
            _normalize_messages(llm_call.prompt.messages + [llm_call.output]),
            tools=llm_call.prompt.tools or None,  # type: ignore
            tokenize=False,
            add_generation_prompt=False,
            training_prompt=True,  # apriel-specific kwarg argument passed to jinja template
        )
        if normalized_model_name == "ServiceNow-AI/Apriel-1.6-15b-Thinker":
            text_out = _apriel_text_split(text_full)
        else:
            assert text_in in text_full, "Prompt text should be a prefix of the full text"
            text_out = text_full[len(text_in) :]
        images = _extract_images(messages)

        return TextPair(
            prompt_text=text_in,
            response_text=text_out,
            images=images,
            reward=step_reward,
        )
    raise NotImplementedError(f"LLM call rendering not implemented for model {model_name}")


def _apriel_text_split(full_text: str) -> str:
    """Split by last <|begin_assistant|>"""
    split_token = "<|begin_assistant|>"
    if split_token not in full_text:
        raise ValueError(f"Expected split token {split_token} not found in full text")
    return full_text.rsplit(split_token, 1)[-1].strip()


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
