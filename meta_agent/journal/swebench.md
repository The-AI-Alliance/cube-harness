# SWE-bench Meta-Agent Journal

Tracking iterative improvements to the SWE-bench agent harness.
Baseline: gpt-5.4, no hints, 36% (104/288) on full verified set.
Working model for iterations: gpt-5.4-mini (faster/cheaper signal).

> **Debug-only entries** are marked `[debug]` — not for the PR.
> **Harness changes** (tool.py, recipe, hints) go in the PR.

---

## Iteration 0 — Baseline (2026-04-29)

**Model**: gpt-5.4, no hints  
**Tasks**: 288 (full verified)  
**Result**: 104/288 = 36.1%  
**Setup**: render_last_n_obs=2, max_steps=30, no str_replace tool  

**Failure patterns observed in traces**:
- Agents use `sed` in bash for edits — fragile, silently edits wrong locations
- Pass/pass false confidence: agent sees unrelated tests pass, calls final_step without fixing the bug
- Agents set timeout=600-1200s on individual bash commands (test suites, mostly harmless but slow)
- `git` misused as a test runner (git stash/apply loops)

**Not yet addressed**.

---

## Iteration 1 — str_replace + hints + obs window (2026-04-30)

**Model**: gpt-5.4-mini, with hints  
**Tasks**: 20 (targeted: 11 with hints/failed, 6 no hints/failed, 3 previously passed)  
**Result**: 0/20 — REGRESSION  

**Changes made**:
- Added `str_replace` tool to SWEBenchTool
- Capped bash timeout at 300s
- System prompt: "verify the specific failing test before final_step"
- render_last_n_obs: 2 → 3
- Hints: generalized (removed exact solutions)

**Root causes of regression**:
1. `render_last_n_obs=3` + smaller context → gpt-5.4-mini confused, burned steps without final_step
2. "verify the specific failing test" too vague — agent doesn't know which test; spun without calling final_step
3. `str_replace` being re-applied in loops: new_str embedding old pattern → "Already found" every time
4. 300s timeout cap: potentially blocked Django/astropy test suites (though all hit step limit, not timeout)
5. gpt-5.4-mini with hints: additional context may overwhelm smaller model

**Flask trace analysis** [debug]:
- Agent used str_replace at T04 ✓ — tool IS being adopted
- Applied str_replace 4 times (T04, T10, T17, T21) due to loop bug
- Ran full test suite 5 times (T14, T18, T23, T25, T28, T29) — pass_to_pass only
- Never called final_step — all 30 steps consumed

---

## Iteration 2 — Fix regression, tune (2026-04-30)

**Model**: gpt-5.4-mini, with hints  
**Tasks**: 2 sanity first, then 20  

**Changes from iteration 1**:
- `str_replace`: idempotency check — if old_str not found AND new_str already present → "Already applied"
- Timeout cap: 300s → 600s
- System prompt: replace "specific failing test" with "find and run a targeted test (single class or function)"
- render_last_n_obs: 3 → 2 (revert)
- Hints: added `Verify with: conda run -n testbed ...` to each hint

**Hypothesis**: Idempotency fix + better test guidance should stop the spin loop. render_last_n_obs=2 revert should recover the 3 previously-passing tasks.

**Result**: 0/20 — all MAX_STEPS_REACHED (20 task run), 0/2 sanity run

**Diagnosis**:
- 16/20 tasks DID make file edits — agents are exploring and editing
- 2/20 made zero file edits (matplotlib-14623, requests-1142) — likely gpt-5.4-mini couldn't parse the problem
- 0/20 called `final_step` — all hit max steps limit
- Evaluation runs on whatever file state exists when max_steps is reached (no final_step needed)
- Fix quality: conceptually right direction but wrong implementation (requests-1921: added session_setting None loop but test still fails)
- django-10914 agent MODIFIED TEST FILES despite explicit instruction not to
- str_replace loop bug still present — 3-5 edit passes per file even after idempotency fix (recursive pattern fix wasn't in yet)

**Root cause**: gpt-5.4-mini too weak for these tasks. Fixes are directionally correct but implementation wrong.
Secondary: need recursive-pattern str_replace fix and loop detection.

---

## Iteration 3 — gpt-5.4 + fixes (2026-04-30)

**Model**: gpt-5.4, with hints
**Tasks**: 20 (same as iter 2)

**Changes from iteration 2**:
- `str_replace`: added recursive-pattern guard (detect when old_str is prefix of new_str)
- Genny: added loop detection — injects [LOOP WARNING] when same action repeated in episode
- Model: gpt-5.4-mini → gpt-5.4

**Hypothesis**: gpt-5.4 should produce correct fixes for most of these tasks. Loop detection prevents spinning.
With hints + gpt-5.4, expecting 5-10/20 (25-50%).

**Result**: TBD

---
