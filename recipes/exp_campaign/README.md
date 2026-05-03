# Experiment Campaign — Tool × Model Sweep on Azure

Started 2026-05-02. Sweep across two observation/action tools and the top-priority
GPT models on Azure OpenAI, measured on WAA (Windows Agent Arena) and OSWorld.

## Tools under test

| Tool | Observations | Actions | Notes |
|------|--------------|---------|-------|
| **Tool 1** | Screenshot + a11y element table | `run_pyautogui(code)` | Current default. Element table provides explicit (x,y,w,h) for grounding; agent emits Python/pyautogui. |
| **Tool 2** | Screenshot only (no a11y) | 13 discrete primitives via `computer_13` | Lower priority. Pure pixel-grounding with discrete `click`/`typing`/`hotkey`/etc. Different system prompt — see `_prompts.py`. |

Both tools share `observe_after_action=True` and `max_steps=100`.

## Prompts (per-benchmark, per-tool)

Prompts in `_prompts.py` are mirrored verbatim from each benchmark's existing azure
recipe except the Actions block, which swaps to the 13-action surface for Tool 2:

| Constant | Mirrors | Diff |
|----------|---------|------|
| `WAA_TOOL1_AXTREE_PYAUTOGUI` | `WAA_SYSTEM_PROMPT` in `recipes/waa/azure_haiku.py` | none — exact match |
| `WAA_TOOL2_SCREENSHOT_13ACTIONS` | same WAA framing | drops a11y/element-table line; 13 discrete actions instead of pyautogui |
| `OSWORLD_TOOL1_AXTREE_PYAUTOGUI` | `OSWORLD_SYSTEM_PROMPT_PYAUTOGUI_AXTREE` in `recipes/osworld/eval_azure_osworld.py` | none — exact match |
| `OSWORLD_TOOL2_SCREENSHOT_13ACTIONS` | same OSWorld framing | drops element-table line; 13 discrete actions instead of pyautogui |

## Models under test (priority order)

1. `azure/gpt-5.4-mini`  ← starting here
2. `azure/gpt-5.4`
3. (lower priority, deferred) Sonnet 4.6, Opus 4.7 (after cost+caching review), Gpt-5.4-nano, Haiku

## Benchmarks

| Benchmark | Task count | Subset env var | Notes |
|-----------|-----------:|----------------|-------|
| **WAA**   | 152 | (full corpus) | Windows 11, image `waa-windows-vm-kusha-lo`. Has fast-fail VM health gate (`_health_gate_or_die` bails dead VMs in ~6s so Ray retries on a fresh VM). |
| **OSWorld** | 360 (`test_nogdrive`) | `OSWORLD_SUBSET=test_nogdrive` (default), `test_small` (39), `test_all` | Ubuntu, image `osworld-ubuntu-vm-generalized`. **No fast-fail health gate yet** — adds setup-budget burn on bad VMs. Filters out gdrive-credentialed tasks automatically. |

## Infra settings (shared)

- Resource group `ui_assist`, storage `cubeexpvhd`, vnet `vnet-westus2`, nsg `osworld-nsg`
- `n_cpus=50` parallelism (validated through prior runs)
- `Experiment.max_retries=3`, `RETRIABLE_STATUSES = {FAILED, CANCELLED, STALE}`
- Pre-warm Azure CLI token in driver before Ray spawn (avoids cold-start auth storm)
- Retry+backoff on `_CachedCliCredential.{get_token, get_token_info}` (6 attempts, 0/1/2/4/8/16s + jitter)
- Pre-run `cleanup_orphaned_resources()`; pre-run `cleanup_stale(60s)` if `WAA_CLEAN_START=1`

## How runs are driven

I drive the runs myself, one recipe at a time:

1. Launch the recipe via `nohup` + `caffeinate` so the Mac stays awake
2. Monitor via the 15-min `/loop` — DNS, VM count, az failures, stale heartbeats
3. When a recipe finishes (or hits a sticky failure mode I can't recover from in
   place), kill cleanly, update the README's results table, move to the next
4. After all 8 in Pass 1, do Pass 2 with the appropriate `*_RESUME_DIR` env var
   on each recipe to retry FAILED/STALE/CANCELLED tasks

Per-recipe launch template (substitute `<recipe>`):

```bash
mkdir -p /tmp/cube-harness-logs/campaign
LOG=/tmp/cube-harness-logs/campaign/<recipe>.log
nohup .venv/bin/python -u recipes/exp_campaign/<recipe>.py > "$LOG" 2>&1 &
EVAL_PID=$!
caffeinate -d -i -s -w $EVAL_PID &  # auto-exits when eval exits
```

Per-recipe resume template:

```bash
# WAA
WAA_RESUME_DIR=<output_dir> nohup .venv/bin/python -u recipes/exp_campaign/<recipe>.py > "$LOG" 2>&1 &
# OSWorld
OSWORLD_RESUME_DIR=<output_dir> nohup .venv/bin/python -u recipes/exp_campaign/<recipe>.py > "$LOG" 2>&1 &
```

Resilience: if a recipe is degenerating into a known failure mode (e.g. sustained
az credential storm, stuck workers >5min), I kill, apply the appropriate fix
(zombie+retry status patch, az retry+backoff, editable install drift), and
resume. I won't let an obviously-broken eval bleed for hours.

## Run order (WAA all models first, then OSWorld)

1. `waa_axtree_pyautogui_gpt54mini.py`     — Tool 1 × gpt-5.4-mini
2. `waa_axtree_pyautogui_gpt54.py`         — Tool 1 × gpt-5.4
3. `waa_screenshot_13actions_gpt54mini.py` — Tool 2 × gpt-5.4-mini
4. `waa_screenshot_13actions_gpt54.py`     — Tool 2 × gpt-5.4
5. `waa_axtree_pyautogui_gpt54nano.py`     — Tool 1 × gpt-5.4-nano
6. `waa_screenshot_13actions_gpt54nano.py` — Tool 2 × gpt-5.4-nano
7. `waa_axtree_pyautogui_haiku.py`         — Tool 1 × Claude Haiku 4.5
8. `waa_screenshot_13actions_haiku.py`     — Tool 2 × Claude Haiku 4.5
9. `osworld_axtree_pyautogui_gpt54mini.py`     — Tool 1 × gpt-5.4-mini
10. `osworld_axtree_pyautogui_gpt54.py`        — Tool 1 × gpt-5.4
11. `osworld_screenshot_13actions_gpt54mini.py` — Tool 2 × gpt-5.4-mini
12. `osworld_screenshot_13actions_gpt54.py`     — Tool 2 × gpt-5.4

## Results — Pass 1 (fresh)

| # | Recipe | Status | Output dir | Done | Pass | Pass% | Notes |
|---|--------|--------|------------|-----:|-----:|------:|-------|
| 1 | `waa_axtree_pyautogui_gpt54mini` | DONE (12:19→19:49, ~7h30m) | `20260502_121934_..._waa-cube_8c80a1f1` | 152 | 35 | **23.0%** | Avg reward 0.368 (partial credit). 112 COMPLETED + 40 MAX_STEPS + 0 FAILED. 19 tasks auto-retried (round 1 → retry round 1, no further rounds needed). Best apps: vs code 45%, clock 50%, settings 50%, notepad 50%. Worst: chrome/edge 0% (5 tasks), paint 0% (2). Long-tail wins took up to 84 steps. **Pre-launch fix**: `WAATaskConfig.load_task_execution_info(self.task_id)` was calling stale cube API; fixed `cubes/.../waa_cube/benchmark.py:88` → `self.load_task_execution_info()`. |
| 2 | `waa_axtree_pyautogui_gpt54` | DONE (19:55→22:10, ~2h15m) | `20260502_195519_..._waa-cube_678dbc98` | 152 | 51 | **33.6%** | Avg reward 0.400 (partial credit). **All 152 COMPLETED, 0 MAX_STEPS, 0 FAILED** (vs r1's 40 MAX_STEPS) — gpt-5.4 always commits to done()/fail() early. 10 cancelled tasks (step_timeout) retried in round 2. **~3.5× faster than r1** (avg 19.9 steps/task vs 47): pass median 5 steps (vs 10), fail median 6 (vs 18). +10pp pass rate over r1, +0.06 reward. |
| 3 | `waa_screenshot_13actions_gpt54mini` | DONE (04:11→06:54, ~2h43m) | `20260503_041146_..._waa-cube_7fe28396` | 152 | 32 | **21.1%** | Avg reward 0.243 (lower than r1's 0.368 — Tool 2 commits harder, fewer partial-credit tasks). 134 COMPLETED + 15 MAX_STEPS + 3 FAILED (ContentPolicy non-retryable). 3 auto-retry rounds (round 1: 58 tasks retried, rounds 2/3: 3 each). Survived a mid-run DNS outage via retries. First `computer_13` eval — no Genny tool-surface mismatch surfaced. |
| 4 | `waa_screenshot_13actions_gpt54` | DONE (08:49→11:26, ~2h37m) | `20260503_084927_..._waa-cube_dd65263a` | 152 | 37 | **24.3%** | Avg reward 0.303. 127 COMPLETED + 22 MAX_STEPS + 3 FAILED (ContentPolicy non-retryable). 3 retry rounds (r1:38→3, r2:3→3, r3:3→3) recovered all transient failures from a mid-run DNS outage (APIConn/ServiceReq/CredUnavail). Tool 2 × gpt-5.4 lands between r2 (Tool 1 × gpt-5.4 = 33.6%) and r3 (Tool 2 × gpt-5.4-mini = 21.1%) — Tool 2 imposes ~10pp penalty on gpt-5.4. |
| 5 | `waa_axtree_pyautogui_gpt54nano` | RESUMING (PID 28334, since 15:12) | `20260503_115926_..._waa-cube_6c3052fd` | — | — | — | Tool 1 / gpt-5.4-nano. **First launch killed at 0:21 elapsed (78.6% UnsupportedParamsError on `tool_choice`); patched all recipes 5-12 with `LITELLM_DROP_PARAMS=true`. Second launch ran ~3h, hit subnet exhaustion (172.16.0.0/24 SubnetIsFull) at 116/152 done with 16 pass + 24 HttpResponseError + 16 APIConnectionError, then laptop went to sleep killing the eval.** Resuming with `WAA_RESUME_DIR` to recover the 82 incomplete tasks (40 FAILED + 34 RUNNING + 2 QUEUED + 6 retriable). |
| 6 | `waa_screenshot_13actions_gpt54nano` | not started | — | — | — | — | |
| 7 | `waa_axtree_pyautogui_haiku` | not started | — | — | — | — | |
| 8 | `waa_screenshot_13actions_haiku` | not started | — | — | — | — | |
| 9 | `osworld_axtree_pyautogui_gpt54mini` | not started | — | — | — | — | |
| 10 | `osworld_axtree_pyautogui_gpt54` | not started | — | — | — | — | |
| 11 | `osworld_screenshot_13actions_gpt54mini` | not started | — | — | — | — | |
| 12 | `osworld_screenshot_13actions_gpt54` | not started | — | — | — | — | |

Pass% = COMPLETED-with-reward=1 / (COMPLETED + MAX_STEPS_REACHED + FAILED).

## Results — Pass 2 (resume)

| # | Recipe | Status | Done | Pass | Pass% | Δ from Pass 1 |
|---|--------|--------|-----:|-----:|------:|---------------|
| (will be filled in once Pass 1 finishes and we kick off `--resume`) |

## Open questions / pending items

- **Model deployment names**: Recipes use `azure/gpt-5.4-mini` and `azure/gpt-5.4`. The prior run used `azure/gpt-5-mini`. If the actual Azure deployment names are `gpt-5-mini` / `gpt-5`, edit `MODEL_NAME` in each recipe (1 line each).
- **VM-level health gate parity for OSWorld**: WAA bails dead VMs in ~6s; OSWorld currently waits the full setup_controller budget. If OSWorld's first-pass failure rate is high, port `_health_gate_or_die` style logic to `osworld_cube/task.py`.
- **Tool 2 first run**: `computer_13` action space hasn't been exercised end-to-end with Genny in our setup before. If issues surface (e.g. Genny tool surface mismatch), fix on first eval.
