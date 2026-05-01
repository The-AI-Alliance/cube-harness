# WorkArena L1 — Chat Wiring and Send Message

**Date**: 2026-04-09
**Branch**: `refactor/bgym-thin-wrapper` (PR #272) -> `fix/workarena-chat-wiring` (PR #279)
**Base commit**: PR #272 head
**Benchmark**: WorkArena L1 (33 tasks + targeted 13-task subset)
**Model**: azure/gpt-5-mini
**Objective**: Wire send_message through ChatTool, fix chart/knowledge tasks

## Context
Baseline was 39%. Chart/knowledge tasks (6 total) were structurally impossible — no send_msg_to_user action. Switched to PR #272 which exposes it via HighLevelActionSet.

## Runs

### Run 2 — Targeted subset (13 tasks)
- Fixed: `filter_actions` for bgym native names, `_build_action_schemas` type pop, subset filtering
- Added `send_msg_to_user` description override ("answer only")
- Added `_task_needs_send_msg` filter to hide chat for non-QA tasks

**Result: 8/13 = 62%** (chart 3/4, knowledge 1/1, order 3/3)

### Run 3 — Full L1, description-only approach
- Removed `_task_needs_send_msg` hack in favor of docstring guidance only.

**Result: 3/33 = 9%** — massive regression. Description-based guidance insufficient.

**Lesson**: Structural filtering (hiding the action) is necessary. The agent ignores docstring instructions and uses send_msg_to_user as a thinking channel.

### Run 3b — Full L1, with all fixes restored
**Result: 17/33 = 51.5%**

## Changes made
- PR #272: filter_actions, action schema fix, send_msg exposed
- PR #279: ChatTool wiring, validate cache, on_send_message callback
- cube-standard PR #97: report_infeasible action on ChatTool

## Findings
- Description-only approach fails — agent needs structural guardrails
- `requires_chat_answer` must be computed at benchmark setup, not at runtime
- Double validate() call per step is a perf issue (cached in PR #279)
