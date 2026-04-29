# WorkArena L1 Baseline

**Date**: 2026-04-08
**Branch**: `meta-agent-iter-1-use-autocomplete` -> `feat/meta-agent`
**Base commit**: `56f8c0c` (feat: meta-agent scaffolding)
**Benchmark**: WorkArena L1 (33 tasks, 1 seed)
**Model**: azure/gpt-5-mini
**Objective**: Establish WorkArena L1 baseline with BrowsergymTool

## Runs

### Run 0 — Debug (2 tasks)
- `keyboard_press` NameError: HighLevelActionSet default subsets don't include `"coord"`.
- **Fix**: `HighLevelActionSet(subsets=["chat", "infeas", "bid", "nav", "tab", "coord"])`.
- Result: 2/2 after fix.

### Run 1 — Full L1, sequential
- Ray crashed: `KeyError: 'workarena.servicenow.create-problem'` — ClassVar `task_metadata` not available in Ray workers.
- **Workaround**: Embed `task_class_path` in `WorkArenaTaskConfig`.
- Fell back to sequential (no Ray).

**Result: 13/33 = 39%**

| Category | Result | Notes |
|----------|--------|-------|
| order | 8/9 (89%) | Reliable |
| create | 0/5 | Submit button navigates away without saving |
| sort | 0/6 | Column headers don't update sysparm_query |
| filter | 0/6 | Stuck on combobox inputs that time out |
| chart | 0/4 | No send_msg_to_user action available |
| knowledge | 0/1 | Same — no chat action |
| impersonation | 1/1 | Lucky |
| all-menu | 1/1 | Passed |

## Bugs found
1. `keyboard_press` not in default HighLevelActionSet subsets
2. Ray ClassVar isolation for task_metadata
3. `send_msg_to_user` not exposed on BidBrowserActionSpace branch
4. `Ctrl+a` vs `Control+a` — Playwright expects `Control`
