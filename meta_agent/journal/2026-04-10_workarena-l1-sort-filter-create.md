# WorkArena L1 — Sort, Filter, Create Task Investigation

**Date**: 2026-04-10 to 2026-04-11
**Branch**: `feat/meta-agent` (on top of PR #279)
**Base commit**: `14bad2a` (feat: add agent_hints.py)
**Benchmark**: WorkArena L1 (targeted subsets + full)
**Model**: azure/gpt-5-mini, azure/gpt-5.4
**Objective**: Fix 0% pass rate on create/sort/filter tasks

## Context
After chat wiring, order/chart/knowledge tasks work. Create (0/5), sort (0/6), filter (0/6) still fail.

## Create tasks — submit_form() development

**Root cause**: Visible Submit button uses `sysverb_insert_and_stay` (navigates away). WorkArena's `gsftSubmit` hook only patches `window.gsftSubmit` in gsft_main.

**Fix evolution**:
1. Simple `gsftSubmit(null, null)` — TypeError (null form)
2. Find `#sysverb_insert` button — threw because `btn.form` is null
3. Find `<form>`, set `form.sys_action` — old_gsftSubmit still threw
4. **Final**: JS try/catch wrapper. localStorage write happens before throw.

New actions: `submit_form()`, `keyboard_type_into(bid, text)`.
Result: create 0/5 -> **5/5** with hints.

## Sort tasks — filter UI discovery

**Root cause**: Column header clicks don't update `sysparm_query` URL param (which the verifier checks). Must use filter panel.

**Combobox pattern**: Custom combobox has input (times out) + adjacent button (works). Agent retried same BID 15+ times.

Hint iterations: v1 "use filter UI" -> v2 "click+type" -> v3 "click BUTTON not INPUT" -> v4 "NEVER retry same BID twice" (breakthrough).
Result: sort 0/6 -> **5/6** with hints.

## Filter tasks
Same combobox pattern. max_steps 25->40 (4+ conditions need ~22 turns).
Result: filter 0/6 -> **2-3/6** with hints.

## Other fixes
- Azure ContentPolicyViolationError: append validation content (not prepend) in episode.py
- subset_from_list not filtering get_task_configs: added allowed_ids check

## Runs

### Full L1 with gpt-5.4 (v4)
- Config: max_steps=40, all hints, submit_form + keyboard_type_into
- Launched as background process, results pending at session end
