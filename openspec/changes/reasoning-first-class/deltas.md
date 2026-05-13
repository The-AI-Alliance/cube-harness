# Deltas: Reasoning First-Class

Applies to: `openspec/specs/llm/spec.md`

---

## MODIFIED — `Usage`

A new `reasoning_tokens` field is added. Default 0 so existing serialized records
deserialize unchanged.

```python
class Usage(TypedBaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0     # NEW — provider-reported reasoning/thinking tokens
    cost: float = 0.0
```

`LLM._extract_usage` MUST populate `reasoning_tokens` from
`response.usage.completion_tokens_details.reasoning_tokens` when present
(OpenAI o-series, gpt-5 family, and any provider LiteLLM normalizes to that
shape). Anthropic does not surface reasoning tokens separately — they are folded
into `completion_tokens`; `reasoning_tokens` stays 0 on Anthropic responses.

---

## ADDED — `get_reasoning(msg: Message) -> str`

Module-level helper in `cube_harness.llm`. Provider-agnostic extractor:

1. `msg.reasoning_content` if non-empty (OpenAI / streaming).
2. Concatenation of `msg.thinking_blocks[*].thinking` if present (Anthropic
   extended thinking).
3. Empty string.

The function operates on any `litellm.Message`, including those reconstructed
from persisted `LLMCall.output` records — making it the canonical reasoning
extractor for both live runs and offline trajectory analysis.

---

## ADDED — `LLMResponse.reasoning_text`

A computed property on `LLMResponse`:

```python
class LLMResponse(TypedBaseModel):
    message: Message
    usage: Usage

    @property
    def reasoning_text(self) -> str:
        return get_reasoning(self.message)
```

Agents SHOULD use `response.reasoning_text` to populate `AgentOutput.thoughts`.

---

## MODIFIED — `LLMConfig`

Adds a Pydantic `model_validator` that fails at construction time when
`reasoning_effort` is set on an Anthropic model with `temperature != 1.0`.

```python
@model_validator(mode="after")
def _check_anthropic_thinking_temperature(self) -> "LLMConfig":
    if (
        self.reasoning_effort is not None
        and _is_anthropic_model(self.model_name)
        and self.temperature != 1.0
    ):
        raise ValueError(...)
    return self
```

Anthropic extended thinking forbids `temperature != 1.0`; previously this
surfaced as a 400 mid-run. The validator surfaces it at config time.

---

## MODIFIED — `Prompt` invariants

A new invariant is added (currently true by virtue of
`Message.model_dump(exclude_none=True)`, but unprotected):

> **Round-trip invariant.** `Prompt._coerce_messages` MUST preserve provider
> reasoning fields on assistant messages so the messages can be re-sent to the
> provider in subsequent calls. Specifically: `thinking_blocks` (including each
> block's `signature`), `reasoning_content`, and `tool_calls` survive coercion
> from `litellm.Message` to dict.

A unit test in `tests/test_llm.py` MUST exercise this invariant.

---

## MODIFIED — Gotchas

The existing gotcha:

> Anthropic extended thinking: set `reasoning_effort`. The reasoning output lands in
> `message.reasoning_content` / `message.thinking_blocks` — log them via
> `AgentOutput.thoughts` so the XRay viewer can display them.

is replaced with:

> **Reasoning extraction.** Set `reasoning_effort` to activate native reasoning
> on supported models (OpenAI o-series and gpt-5, Anthropic Claude 3.7+/4.x,
> Gemini 2.5, Grok 3/4, DeepSeek R1/R2, Qwen3-thinking, Magistral). Use
> `response.reasoning_text` to obtain the thinking string for
> `AgentOutput.thoughts`. The structured form is preserved on
> `response.message.thinking_blocks` / `reasoning_content` for round-trip and
> offline analysis.
>
> **Anthropic + thinking constraint.** Anthropic forbids `temperature != 1.0`
> when extended thinking is active. `LLMConfig` validates this at construction
> time.
>
> **Tool-use loops with Anthropic thinking.** Each assistant turn's
> `thinking_blocks` (including `signature`) MUST be echoed back in subsequent
> calls. `Prompt._coerce_messages` preserves them automatically via
> `Message.model_dump(exclude_none=True)`. Do not strip these fields.

---

## MODIFIED — Public API surface in spec.md

The `Usage` and `LLMResponse` sections of the spec are updated to reflect the
new field and property. The "Public API" section gains an entry for
`get_reasoning`.
