# LLM Integration

**Module:** `cube_harness.llm`

## Purpose

Thin wrapper over [LiteLLM](https://docs.litellm.ai/) that standardizes prompt
construction, retry behavior, and usage accounting. All LLM calls in the harness
flow through this module — per the constitution, direct SDK use (OpenAI SDK,
Anthropic SDK) is forbidden (PS-002).

## Public API

### `LLMConfig`
```python
class LLMConfig(TypedBaseModel):
    model_name: str
    temperature: float = 1.0
    max_tokens: int = 128000
    max_completion_tokens: int = 8192
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    tool_choice: Literal["auto", "none", "required"] | None = "auto"   # None opts out
    parallel_tool_calls: bool = False
    num_retries: int = 5
    retry_strategy: Literal["exponential_backoff_retry", "constant_retry"] = "exponential_backoff_retry"
    timeout: float | None = 120.0       # seconds per attempt; None disables
    set_cache_control: Literal["auto"] | None = None   # Anthropic prompt caching, see "Caching"

    def make(self) -> LLM
    def make_counter(self) -> Callable[..., int]   # partial(token_counter, model=model_name)
```

### `Prompt`
```python
class Prompt(TypedBaseModel):
    messages: list[dict]              # litellm.Message inputs are coerced via a
                                      # field_validator (model_dump(exclude_none=True))
    tools: list[dict] = []
```

Callers may pass a mix of `dict` and `litellm.Message` objects — the validator
normalises to plain dicts at construction. This keeps serialisation noise-free
(Message's dynamic provider-specific fields like `thinking_blocks`,
`reasoning_content` would otherwise trip `PydanticSerializationUnexpectedValue`
on every `model_dump`) and gives downstream readers a single homogenous type
to work with.

### `LLMResponse` / `Usage`
```python
class LLMResponse(TypedBaseModel):
    message: Message          # litellm.Message
    usage: Usage

class Usage(TypedBaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0    # Anthropic prompt caching
    cost: float = 0.0                  # USD from LiteLLM pricing
```

### `LLM`
```python
class LLM:
    def __init__(self, config: LLMConfig)
    def __call__(self, prompt: Prompt) -> LLMResponse
    # Uses litellm.completion_with_retries under the hood with config.retry_strategy.
```

### `LLMCall` (logged record)
```python
class LLMCall(TypedBaseModel):
    id: str = field(default_factory=lambda: str(uuid4()))
    tag: str | None = None           # e.g. "act", "summary", "criticise"
    timestamp: datetime
    config: LLMConfig
    prompt: Prompt
    output: Message
    usage: Usage | None = None
```

Captured in `AgentOutput.llm_calls`. Agents MUST set `tag` to distinguish multi-call
steps in traces and training data.

## Invariants

1. All LLM calls route through `LLM.__call__` — no direct use of `litellm.completion`
   in the harness code.
2. Retry strategy is determined by `LLMConfig`, not the call site.
3. `LLMCall.tag` is the primary way to correlate multiple LLM calls in one agent step.
4. Module-level `litellm.callbacks` is intentionally NOT set. OTel callbacks are
   attached only after a proper `TracerProvider` is configured (see metrics spec) —
   otherwise litellm's default console exporter floods stdout.

## Caching (Anthropic)

When `LLMConfig.set_cache_control == "auto"` and the configured model routes to
Anthropic (direct, Bedrock, or Vertex — detected via `litellm.get_llm_provider`
with a substring fallback for model names LiteLLM's registry hasn't catalogued
yet), `LLM.__call__` places ephemeral `cache_control` breakpoints at:

1. **Message index 1** — the goal / first large user observation. Stable anchor
   that lifts the cached prefix above Anthropic's 1024-token minimum (the system
   message alone is typically under that floor).
2. **Last assistant message** — rolling per-step boundary. Each new step extends
   the cached prefix by one (obs, asst) pair via Anthropic's longest-prefix
   match.
3. **Last tool definition** — caches the entire tools array prefix.

Breakpoint injection is done via LiteLLM's `cache_control_injection_points`
hook (canonical public API; LiteLLM handles the wire-format reshape into
Anthropic's content-block-with-cache_control structure). For non-Anthropic
models the flag is a no-op — the payload is never emitted.

`Usage.cached_tokens` / `Usage.cache_creation_tokens` are populated from the
Anthropic response so trace consumers can see cache-hit rates per step.

## Contracts for implementers

- Agent implementations build a `Prompt` and call `self.llm(prompt)`. Record the
  call:
  ```python
  call = LLMCall(tag="act", config=self.config.llm_config, prompt=prompt,
                 output=resp.message, usage=resp.usage)
  output.llm_calls.append(call)
  ```
- For multi-model agents, use one `LLM` per model — the class holds a single config.
- Pass a token counter from `config.make_counter()` for prompt-size budgeting.

## Gotchas

- `completion_with_retries` returns on first success, but retries count toward the
  per-attempt timeout. Total call time is bounded by `num_retries * timeout` in the
  worst case.
- `Prompt.messages` accepts both dicts and `litellm.Message` objects; the
  `field_validator` coerces Messages to dicts at construction so the stored type
  is always `list[dict]`. Downstream readers don't need to handle the union.
- Anthropic extended thinking: set `reasoning_effort`. The reasoning output lands in
  `message.reasoning_content` / `message.thinking_blocks` — log them via
  `AgentOutput.thoughts` so the XRay viewer can display them.
- Cost is USD from LiteLLM's built-in pricing — may lag behind provider price changes.
