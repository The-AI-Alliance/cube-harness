# 2026-04-25 — Create Tasks Fixed (change-request + hardware-asset)

## Status: BOTH PASS ✅

Experiment `20260425_072247_genny_workarena-l1-hints-gpt-5.4`:
- create-change-request: reward=1.0 ✅
- create-hardware-asset: reward=1.0 ✅

## What was fixed

### create-hardware-asset (seed=700)
Fixed in previous session. The TAB PROTOCOL hint in `_CREATE_HARDWARE_ASSET_HINT` resolved
the infinite General↔Financial tab cycling loop. Confirmed again this session.

### create-change-request (seed=106)

**Problem 1 — Number field (CHG0035285 vs CHG0000013)**
The React Submit button sends `sysverb_insert_and_stay` instead of `sysverb_insert`,
causing an auto-generated non-sequential Number. Fixed two ways:
1. Patch 3 in `GenericNewRecordTask.setup_goal()`: `page.route()` intercept replaces
   `sysverb_insert_and_stay` → `sysverb_insert` in all POST bodies.
2. Number field hint: `_CREATE_HINT` tells agent to fill Number field first with `fill()`.
In this run the agent filled CHG0000013 first (hint effective), and Number validated correctly.

**Problem 2 — Close notes not filled**
The Change Request form has a "Closure Information" tab containing both:
- Close code (combobox)
- Close notes (textbox)

The agent correctly clicked the tab and filled Close code, but then navigated BACK to
the Planning tab (act021 = click a565) before filling Close notes, then submitted.

Fix: added `_CREATE_CHANGE_REQUEST_HINT` with explicit warning:
> After clicking Closure Information tab, fill BOTH Close code AND Close notes before
> leaving that tab.

## Key observations

- Genny rolling summary loses track of "what remains to fill" when jumping between tabs
- The cheat test (`test_workarena_cheat.py react-submit`) confirmed: clicking Submit
  with no fields filled produces 0 POST bodies (React validates client-side before submitting)
- Patch 3 (page.route intercept) is correct but can't be verified by empty-form diagnostic
- Navigation timeout fix: added `page.set_default_navigation_timeout(90000)` to diagnostic

## Code changes

- `WorkArena/src/browsergym/workarena/tasks/form.py`: Patch 3 in `setup_goal()` + teardown cleanup
- `cube-harness/meta_agent/workarena_hints.py`:
  - `_CREATE_HINT`: fill Number field first
  - `_CREATE_CHANGE_REQUEST_HINT`: warn about Closure Information tab dual-field requirement
  - `_CREATE_HARDWARE_ASSET_HINT`: tab protocol + loop detection

## Next steps

Run full L1 baseline (33 tasks) to get updated accuracy after all create task fixes.
Previous baseline was 39% (13/33) with gpt-5.4-mini.
