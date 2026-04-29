# WorkArena L1 — Code Fixes Ablation (Hints vs No-Hints)

**Date**: 2026-04-12
**Branch**: `feat/meta-agent`
**Base commit**: `2c6b07f` (fix: increase max_steps to 40) + uncommitted changes
**Benchmark**: WorkArena L1 (33 tasks, 1 seed)
**Model**: azure/gpt-5.4
**Objective**: Measure impact of code fixes alone vs code fixes + hints

## Context
Previous sessions accumulated hints to get sort/filter/create tasks passing. This session
routes fixes to their correct homes (tool docstrings, scaffolding, AXTree config) and measures
how much the code fixes alone achieve without any hints.

## Code fixes applied (both runs)
1. **Action error propagation**: bridge `_last_info["action_error"]` into `_last_obs["last_action_error"]` — agent now sees timeout errors
2. **AXTree enrichment**: `flatten_axtree_to_str(with_clickable=True, ignored_properties=(...))` — agent sees which elements are clickable/readonly/focusable
3. **send_message filtering**: hidden for non-chat tasks via `_TASKS_REQUIRING_CHAT`
4. **js_eval removed** from `_SUPPORTED_ACTION_NAMES`
5. **Tool docstrings**: browser_type (autocomplete warning), browser_click (combobox pattern), submit_form (final action), send_message (answer only)

## Runs

### Run 7 — No hints (code fixes only)
- Config: gpt-5.4, max_steps=40, 4 workers, task_precision={}, task_hints={}

**Result: 19/33 = 58%**

### Run 8 — With hints (code fixes + precision + hints)
- Config: gpt-5.4, max_steps=40, 4 workers, task_precision=WORKARENA_TASK_PRECISION, task_hints=WORKARENA_TASK_HINTS

**Result: 32/33 = 97%**

### Breakdown

| Category | No hints | With hints | Hints needed? |
|----------|----------|------------|---------------|
| order (8) | 100% | 100% | No |
| create (5) | 100% | 100% | No — docstrings sufficient |
| all-menu (1) | 100% | 100% | No |
| impersonation (1) | 100% | 100% | No |
| knowledge (1) | 100% | 100% | No |
| chart (4) | 25% | 75% | Yes — answer format (task precision) |
| sort (6) | 0% | 100% | Yes — "use filter UI" (task precision) |
| filter (6) | 17% | 100% | Yes — combobox bid+1 (hint) |

## Findings
- Code fixes alone: 39% -> **58%** (+19pp)
- Code fixes + hints: 39% -> **97%** (+58pp)
- **Create tasks solved without hints** — docstrings + AXTree clickable annotations enough
- Sort/filter still need hints: sort is task precision (goal doesn't say to use filter UI), filter is interaction pattern (combobox bid+1)
- Chart needs task precision (answer format not in goal)
- single-chart-min-max passed without hints but failed with — seed variance

## Next steps
- Fix sort upstream: WorkArena task goal should specify the filter UI method
- Fix filter: investigate if AXTree readonly annotation makes combobox input obviously non-clickable
- Fix chart: WorkArena goal should specify expected answer format
