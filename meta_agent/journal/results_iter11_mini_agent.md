# Iter 11 Results — MiniAgent on HAL-50

**Date:** 2026-05-01
**Branch:** feat/mini-agent
**Experiment dir:** `20260501_183314_MiniAgent-azure_gpt-5.4-mini_*`

## Score

| Subset | Correct | Total | Score |
|--------|---------|-------|-------|
| Django | 6 | 25 | 24.0% |
| Sphinx | 11 | 25 | 44.0% |
| **HAL-50** | **17** | **50** | **34.0%** |

Target band: 40–55% (GPT-5-medium HAL-50 = 46%).

## Setup

- Model: `azure/gpt-5.4-mini`, temperature=1.0, parallel_tool_calls=False, drop_params=True
- Agent: MiniAgent (faithful port of mini-SWE-agent), step_limit=250, flat message list
- Infra: Toolkit yul101, n-parallel=20
- All 50 episodes COMPLETED (no infra failures)
- Average steps: 11.3 (min 6, max 20) — agents submit relatively early

## Infrastructure fixes required

1. **Temperature 0.0 rejected by GPT-5**: `litellm.UnsupportedParamsError` — added `drop_params=True` to `LLMConfig` and changed recipe to `temperature=1.0`.
2. **--retry reuses old episode_config.json**: Old configs with wrong temperature persisted. Fix: always start fresh runs (don't use --retry across config changes).

## Gap analysis

| Factor | Upstream | Ours | Impact |
|--------|----------|------|--------|
| Temperature | 0.0 | 1.0 (forced) | Est. -6-15pp |
| parallel_tool_calls | True | False | Minor |
| Model tier | GPT-5-mini tier | gpt-5.4-mini | Unknown |

**Primary gap cause**: temperature=1.0 (GPT-5 API doesn't support 0.0). Upstream mini-SWE-agent used GPT-4 era models where temperature=0 is supported.

## Passed tasks (17)

Django: django-11848, 11951, 11964, 12050, 12155, 12193
Sphinx: sphinx-10323, 10435, 10466, 7757, 8269, 8475, 8721, 9281, 9320, 9367, 9698

## Next steps

1. **Run with Sonnet** (`anthropic/claude-sonnet-4-6`, temperature=0) — should be closer to upstream
2. **Enable parallel_tool_calls=True** for gpt-5 models — upstream uses this
3. **Bash timeout**: Current 120s may be too short for some test suites; upstream used 600s
