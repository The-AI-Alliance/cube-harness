# browsercomp-cube

[BrowseComp](https://openai.com/index/browsecomp/) ported to the [CUBE](../../) protocol — 1,266 hard web information-retrieval tasks that require multi-step browsing.

## Overview

Each task asks a hard, deliberately hard-to-find factual question (e.g. *"In what year did a particular obscure paper appear in NeurIPS?"*). The agent must browse the web, gather evidence, and submit an `Exact Answer`. An LLM judge (`[scorer_model](src/browsercomp_cube/benchmark.py)`) compares the agent's answer to the ground truth using the official BrowseComp grader prompt and emits `correct: yes|no`.

The dataset is shipped encrypted (XOR + Base64, per-row canary) by OpenAI. `BrowseCompBenchmarkConfig.install()` downloads the encrypted CSV once and splits it into a per-task execution cache; ciphertext is only decrypted in `BrowseCompTaskConfig.make()`, so cleartext never lands on disk.

## Prerequisites

- An LLM provider key for the agent and the grader (anything LiteLLM speaks works).
- A [Brave Search API key](https://brave.com/search/api/), if you use the default cube web-search tool.
- Network access to `openaipublic.blob.core.windows.net` for the one-time dataset download.

## Installation

```bash
uv pip install browsercomp-cube
```

## Usage

### Via recipe (full evaluation run)

```bash
uv run recipes/browsercomp.py --debug   # 2 tasks, sequential
uv run recipes/browsercomp.py           # full 1,266-task Ray run
```

### Programmatic

```python
from browsercomp_cube import BrowseCompBenchmarkConfig, SubmitAnswerToolConfig
from cube.tool import ToolboxConfig
from cube_web_tool import BraveWebSearchToolConfig, WebFetchToolConfig

cfg = BrowseCompBenchmarkConfig(
    tool_config=ToolboxConfig(
        tool_configs=[BraveWebSearchToolConfig(), WebFetchToolConfig(), SubmitAnswerToolConfig()]
    ),
    scorer_model="gpt-5.4-mini",
)
cfg.install()  # one-time: download + split encrypted dataset
bench = cfg.make()
for task_cfg in cfg.get_task_configs():
    task = bench.spawn(task_cfg)
    obs, _ = task.reset()
    # ... agent loop ...
    task.close()
bench.close()
```

## Tools


| Tool                 | Purpose                                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------------------- |
| `BraveWebSearchTool` | Web search via Brave's API                                                                                |
| `WebFetchTool`       | Fetch a URL and return cleaned-up text/markdown                                                           |
| `SubmitAnswerTool`   | Submit the final `Explanation / Exact Answer / Confidence` block; sets `last_answer` and ends the episode |


The expected agent answer format is enforced by the prompt suffix in `[task.py](src/browsercomp_cube/task.py)`:

```
Explanation: <your reasoning>
Exact Answer: <the precise answer>
Confidence: <0-100>
```

## Grading

`BrowseCompTask.evaluate()`:

1. Builds the official BrowseComp grader prompt (`question`, `response`, `correct_answer`).
2. Calls `scorer_model` via LiteLLM (retried up to `grader_retries=3` times on transport errors).
3. Parses `correct: yes|no` from the verdict (case-insensitive).
4. Returns `1.0` for `yes`, `0.0` otherwise, plus an info dict containing `correct`, `submitted`, and the **raw `grader_response`** for reproducibility.

If no answer was submitted, evaluation short-circuits to `0.0`. If the grader response can't be parsed after all retries, the info dict carries `grader_error` instead.

## Environment Variables


| Variable         | Default   | Description                                                                                                |
| ---------------- | --------- | ---------------------------------------------------------------------------------------------------------- |
| `CUBE_CACHE_DIR` | `~/.cube` | Root cache directory; encrypted dataset and execution cache live under `$CUBE_CACHE_DIR/browsercomp-cube/` |
| `OPENAI_API_KEY` | —         | Used by both the agent (when `LLMConfig` points at OpenAI) and the grader (default `gpt-5.4-mini`)         |
| `BRAVE_API_KEY`  | —         | Required by `BraveWebSearchTool`                                                                           |


Other LiteLLM-supported keys (`ANTHROPIC_API_KEY`, `AZURE_API_KEY`, …) work as drop-in replacements for the grader/agent if you change `scorer_model` / `LLMConfig`.

## Debug / Testing

A deterministic debug benchmark with two trivial tasks (no LLM, no network):

```bash
uv run python -m browsercomp_cube.debug
```

```python
from browsercomp_cube.debug import get_debug_benchmark, make_debug_agent

cfg = get_debug_benchmark()
bench = cfg.make()
for task_cfg in cfg.get_task_configs():
    task = bench.spawn(task_cfg)
    agent = make_debug_agent(task_cfg.task_id)
    obs, _ = task.reset()
    done = False
    while not done:
        action = agent(obs, task.action_set)
        env_out = task.step(action)
        obs, done = env_out.obs, env_out.done
    task.close()
bench.close()
```

Smoke tests cover benchmark construction, the grader regex, and the crypto round-trip:

```bash
uv run pytest tests/
```

## Reproducibility


| Item           | Value                                                                                                             |
| -------------- | ----------------------------------------------------------------------------------------------------------------- |
| Source         | [openai/simple-evals — `browsecomp_eval.py](https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py)` |
| Dataset        | `https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv`                                |
| Tasks          | 1,266                                                                                                             |
| Default scorer | `gpt-5.4-mini` (override via `BrowseCompBenchmarkConfig(scorer_model=...)`)                                       |
| Grader retries | 3 (set on `BrowseCompTask.grader_retries`)                                                                        |
| Encryption     | XOR with `SHA256(canary)`-derived key, Base64-wrapped (see `[crypto.py](src/browsercomp_cube/crypto.py)`)         |


`task_metadata.json` is a shipped package resource containing only public fields (`id`, `recommended_max_steps`, `topic`). To regenerate it after a dataset refresh:

```bash
uv run python scripts/generate_task_metadata.py --force
```

## Package Structure

```
src/browsercomp_cube/
├── __init__.py            # Public exports
├── benchmark.py           # BrowseCompBenchmarkConfig / BrowseCompBenchmark
├── task.py                # BrowseCompTask, BrowseCompTaskConfig, BrowseCompTaskMetadata, grader prompt
├── tool.py                # SubmitAnswerTool, SubmitAnswerToolConfig
├── crypto.py              # XOR + Base64 helpers ported from openai/simple-evals
├── debug.py               # DebugBrowseCompBenchmark, DebugAgent (no LLM, no network)
└── task_metadata.json     # Shipped public-field metadata for all 1,266 tasks
```
