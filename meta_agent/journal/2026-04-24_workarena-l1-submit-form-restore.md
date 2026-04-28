# 2026-04-24 — WorkArena L1: submit_form restore + hints wiring

## Context

Investigating a performance regression and iterating on create tasks with gpt-5.4.

## Root cause of create task failures

`submit_form` was removed from `BrowsergymTool` in commit `fc7d4d6` (ExtraWebActionsTool refactor) and not migrated. The function called `gsftSubmit(btn, form, 'sysverb_insert')` in the gsft_main iframe — three arguments. Clicking the visible Submit button uses `sysverb_insert_and_stay` which regenerates system-managed fields like the record Number. WorkArena validates the pre-seeded Number (e.g. `CHG0000013`) so the mismatch causes failure.

Interim attempt: hint the agent to call `js_eval('gsftSubmit()', frame='gsft_main')`. Failed because `gsftSubmit()` with zero args passes `undefined` as the action name → ServiceNow error "Unable to find UI Action with name 'undefined'".

Fix: restored `submit_form` as a `@tool_action` on `ExtraWebActionsTool` (web_actions.py). The action calls `gsftSubmit(btn, form, action)` where `action='sysverb_insert'` by default. Docstring frames this as a general pattern applicable to any SPA where the visible Submit button has side effects.

## Hints wiring

`GennyConfig` has `task_clarification` (per-task goal clarification) but the recipe was not passing `WORKARENA_TASK_PRECISION`. Fixed recipe to pass both:
- `task_hints=WORKARENA_TASK_HINTS`
- `task_clarification=WORKARENA_TASK_PRECISION`

## Results (gpt-5.4, 1 seed, with hints)

| Run | Score | Notes |
|-----|-------|-------|
| nohints | 22/33 = 67% | submit_form missing |
| hints run 1 | 27/33 = 82% | gsftSubmit() 0-arg bug |
| hints run 2 | 28/33 = 85% | submit_form restored |

All 5 create tasks pass (create-change-request, create-hardware-asset, create-incident, create-user, create-problem).

## Remaining failures (5/33)

- `sort-asset-list`, `sort-change-request-list` — stochastic near 40-step limit; passed in previous run with same seed
- `knowledge-base-search` — agent browses but never calls `send_message` with the answer
- `multi-chart-value-retrieval` — reads wrong value from the chart
- `order-loaner-laptop` — empty failure message (ServiceNow state/seed issue, not reproducible)

## Files changed

- `src/cube_harness/tools/web_actions.py` — restored `submit_form` action
- `meta_agent/workarena_hints.py` — fixed `_CREATE_PRECISION` (submit_form, not gsftSubmit()), added js_eval hidden-field hint
- `meta_agent/recipes/workarena_l1_full.py` — import + pass `WORKARENA_TASK_PRECISION` as `task_clarification`
