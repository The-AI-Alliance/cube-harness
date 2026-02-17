import json
from functools import lru_cache

from litellm import Message
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from agentlab2.llm import LLMCall, Prompt

_APRIEL_MODEL_ID = "ServiceNow-AI/Apriel-1.6-15b-Thinker"


def llm_call_to_text_pair(llm_call: LLMCall) -> tuple[str, str]:
    """Convert an LLMCall to a (prompt, response) text pair."""
    prompt = llm_call.prompt
    output = llm_call.output
    model_name = llm_call.llm_config.model_name
    if model_name.endswith("Apriel-1.6-15b-Thinker"):
        return _render_apriel_thinker(prompt, output)
    raise NotImplementedError(f"LLM call rendering not implemented for model {model_name}")


def _render_apriel_thinker(prompt: Prompt, output: Message) -> tuple[str, str]:
    """Render Apriel Thinker LLM call into input/output text pair using the official chat template."""
    tokenizer = _load_apriel_tokenizer()

    text_in = tokenizer.apply_chat_template(
        prompt.messages,
        tools=prompt.tools or None,
        tokenize=False,
        add_generation_prompt=True,
    )

    text_out = _render_apriel_output(output)

    return text_in, text_out


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
