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
    tool_choice: Literal["auto", "none", "required"] = "auto"
    parallel_tool_calls: bool = False
    num_retries: int = 5
    retry_strategy: Literal["exponential_backoff_retry", "constant_retry"] = "exponential_backoff_retry"
    timeout: float | None = 120.0       # seconds per attempt; None disables

    def make(self) -> LLM
    def make_counter(self) -> Callable[..., int]   # partial(token_counter, model=model_name)
```

### `Prompt`
```python
class Prompt(TypedBaseModel):
    messages: list[dict | Message]
    tools: list[dict] = []
```

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
- `Prompt.messages` accepts both dicts and `litellm.Message` objects. Be consistent
  within an agent to keep the trace uniform.
- Anthropic extended thinking: set `reasoning_effort`. The reasoning output lands in
  `message.reasoning_content` / `message.thinking_blocks` — log them via
  `AgentOutput.thoughts` so the XRay viewer can display them.
- Cost is USD from LiteLLM's built-in pricing — may lag behind provider price changes.
