# 2026-04-29 — Genny windowed-obs bug (SWEBench)

## Tasks
psf__requests-1142, pallets__flask-5014 (debug tasks)

## What the agent saw
LLM prompt at turn 3 (second act) had only 4 messages:
`[system][user:goal][assistant:COT-summary][user:react_prompt]`.
The windowed observation section was completely absent despite
`render_last_n_obs=2`. The agent repeated the same `ls` command every
step because it never saw any bash output.

## Hypothesis
`_windowed_history()` in `genny.py` strips leading `tool`-role messages
from each obs group "for structural validity". For browser envs a
`user` (screenshot) message follows, so only tool results are dropped.
For SWEBench the obs IS only tool messages — stripping left the group
empty, and the section header was also suppressed (`if windowed:` was
False).

## Intervention
Read step-2 file from a failed run: confirmed `Observation.contents =
[TextContent(tool_call_id='call_4Abc...', data='total 84 ...')]`.
`Observation.to_llm_messages()` converts `TextContent(tool_call_id=...)`
to `{"role":"tool", ...}`. All messages in the obs group are tool-role →
`start = len(group)` → `group[start:]` = `[]`.

## Fix
Added a third branch in `_windowed_history()`: when `start == len(group)
> 0` (entire group is tool messages), concatenate content and emit a
single `{"role":"user", ...}` message. Orphaned `tool` messages cause
API 400 errors anyway; re-wrapping as user avoids both problems.

Files changed:
- `src/cube_harness/agents/genny.py` — `_windowed_history()` third branch
- `tests/test_genny.py` — `test_all_tool_obs_rewrapped_as_user`

Commit: `fix(genny): re-wrap all-tool obs groups as user messages`

## Result
| Task | Before | After |
|------|--------|-------|
| psf__requests-1142 | 0.0 (MAX_STEPS, agent looped) | **1.0** |
| pallets__flask-5014 | 0.0 (MAX_STEPS, agent looped) | **1.0** |

Model: gpt-5.4-mini, no hints, sequential debug run.

## Control set
All 721 harness unit tests pass. 75 genny tests pass (including 2 prior
windowed-history tests for the browser-style strip case).
