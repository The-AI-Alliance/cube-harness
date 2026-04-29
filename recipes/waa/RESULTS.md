# WAA Eval Results — 2026-04-27 → 04-29

Full 152-task WindowsAgentArena corpus on the LO-enabled image
(`waa-windows-vm-kusha-lo`), Genny agent with `use_som=False` unless noted,
n_cpus=10, max_steps=100, native tool calling.

## Summary

| Model | Wins | Win rate | Avg reward | Empty-action rate | Hit step cap | Wall time | LLM cost |
|---|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | 55/152 | **36.2%** | 0.363 | ~0% | 2% | 7h54m | $80.71 |
| Claude Haiku 4.5 + SoM | 9/24† | **37.5%** | – | – | – | _running_ | – |
| GPT-5-mini (Azure) | 44/152 | **28.9%** | 0.288 | 0% | 30% | 17h + resume | $38.58 + ~$0.20 |
| GPT-4o (Azure) | 7/151 | **4.6%** | 0.046 | 22% | – | 3h | $19.71 |

\* GPT-5-mini original run killed with 3 unfinalized; resumed in
`eval_azure_waa_kusha_gpt5mini_resume.py` (max_steps=50, n_cpus=3). Final
result: ep8 lost (41 steps, reward 0), ep9 won (2 steps, reward 1.0 — chrome
CDP cleared on retry), ep150 partial (2 steps, reward 0.70). Net +2 wins.

† Haiku SoM in progress (started 2026-04-29 10:15, output `20260429_101553_*`).
Numbers are interim.

Paper Table 4 reference (OneOCR + ✓UIA, no Navi grounding pipeline — closest
to our Genny+axtree setup): GPT-4o-mini 7.3%, GPT-4o 13.3%. Our GPT-4o number
is well below paper because the paper's "Navi" agent uses text-block parsing
(extracts ` ```python ``` ` blocks from text), not native tool calling, so it
sidesteps GPT-4o's tool-call-adherence weakness.

## Per-model notes

### Claude Haiku 4.5 (axtree)
- Output dir: `20260427_143250_genny_azure_kusha_haiku_full_waa-cube`
- Original full run had 2 chrome-CDP setup failures (ep9, ep84). After fixing
  the port-allocation race in `cube/infra_utils.free_port`, both passed on
  retry — ep9 won (reward 1.0 in 4 steps), ep84 lost. Net +1 win.
- Avg agent steps per ep: 27. 2% hit step cap.

### Claude Haiku 4.5 + SoM (axtree replaced by tagged screenshot + element
table without coords)
- Output dir: `20260429_101553_genny_azure_kusha_haiku_som_full_waa-cube`
- Status: running (started 2026-04-29 10:15)
- SoM smoke validated end-to-end (1 task, 2 steps, reward 1.0) on
  `eval_azure_waa_kusha_haiku_som_smoke.py`.

### GPT-4o (Azure `gpt-4o-2024-11-20`)
- Output dir: `20260428_134855_genny_azure_kusha_gpt4o_full_waa-cube`
- 22% of agent steps were empty (model returned text without tool calls →
  agent treated as stop). This is the dominant failure mode.
- Tried `tool_choice="required"` briefly but reverted — risk of step-cap loops
  on stuck states without offsetting benefit at this stage.

### GPT-5-mini (Azure `gpt-5-mini`)
- Output dir: `20260428_171223_genny_azure_kusha_gpt5mini_full_waa-cube`
- 0% empty actions across 800+ agent steps — tool-call adherence is solid,
  unlike GPT-4o.
- Avg 47 agent steps per ep (vs Haiku's 27); **30% of episodes hit the
  100-step cap.** This is the wall-time killer — running long.
- Recommendation for future GPT-5-mini evals: lower `max_steps` to 50 to cut
  wall time roughly in half with minimal win-rate impact (the 30% that capped
  at 100 mostly already lost).

## Infra bugs found and fixed during the session

1. **Port-allocation race in `cube/infra_utils.free_port`** — concurrent VM
   launches got duplicate host tunnel ports → SSH tunnels misrouted → setup
   API calls hit the wrong VM, manifesting as 502s. Fixed in
   `cube-standard/src/cube/infra_utils.py` with a process-wide lock and
   reserved-port set.
2. **`WAABenchmark.use_som=True` was a no-op** — field existed but never
   propagated to `WAATaskConfig` or `WAATask`. Fixed in
   `cubes/windows-agent-arena-cube/src/waa_cube/benchmark.py` (3 lines).
3. **`/launch` endpoint typo in probe code** — used `/launch` instead of
   `/setup/launch`; only affected one probe, not the eval. Fixed in
   `recipes/waa/probe_chrome_cdp.py`.

## Known unresolved issues

- **`/setup/upload` 502 tail at n_cpus=20** — appears to be a per-VM
  born-broken issue (not port collision); persists for 60+ seconds even with
  5-retry backoff. Occurs at ~28% rate at n_cpus=20, ~1% at n_cpus=10. Likely
  Defender or Azure-side flakiness during VM warm-up. Workaround for now:
  stay at n_cpus=10. Diagnostic probe needed to characterize.
- **Chrome CDP "ECONNREFUSED on first launch"** — observed once at n=20;
  tunnel established but Chrome not yet listening internally. Currently
  retries 30× × 5s = 150s; with that budget it usually clears.
- **GPT-5-mini hung worker** — ep150 in our run stuck at step 65 for >20min.
  Cause unclear (LLM timeout? Ray worker death?). Process-level retry would
  bound this.

## Recipe inventory

- `eval_azure_waa_kusha_haiku_full.py` — Haiku 4.5 full corpus
- `eval_azure_waa_kusha_haiku_som_full.py` — Haiku 4.5 + SoM full corpus
- `eval_azure_waa_kusha_haiku_som_smoke.py` — 1-task SoM smoke
- `eval_azure_waa_kusha_haiku_retry_failed.py` — resume specific failed eps
- `eval_azure_waa_kusha_gpt4o_full.py` — GPT-4o full corpus
- `eval_azure_waa_kusha_gpt5mini_full.py` — GPT-5-mini full corpus
- `eval_azure_waa_kusha_lo_smoke.py` / `_seq.py` — LO image smoke tests
- `probe_*.py` — diagnostic probes (chrome CDP, upload concurrency, parallel
  VMs, setup-upload race)
