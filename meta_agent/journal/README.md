# Meta-Agent Journal

Each file is one debugging session. A session is a multi-step investigation: it
typically includes several experiments, hypothesis tests, code changes, and re-runs.
A session has a scope (which benchmark/tasks), an objective (what we're trying to
fix/improve), and a record of every run and finding along the way.

## File naming

`YYYY-MM-DD_<benchmark>-<scope>.md` — e.g. `2026-04-08_workarena-l1-baseline.md`

Multiple sessions on the same day get a suffix: `2026-04-12_workarena-l1-axtree-b.md`

## Session template

```markdown
# <Title>

**Date**: YYYY-MM-DD
**Branch**: feat/meta-agent
**Base commit**: <short hash> (<one-line description>)
**Benchmark**: workarena L1 / miniwob / webarena
**Model**: azure/gpt-5.4
**Objective**: <what we're trying to fix or learn>

## Context
<Brief setup — what's already known, what changed since last session>

## Runs
### Run N — <label>
- **Config**: hints=on/off, max_steps=40, n_workers=4
- **Result**: X/Y (Z%)
- **Breakdown**: (per-category table if useful)
- **Key observations**: ...

## Findings
<What we learned — root causes, confirmed/rejected hypotheses>

## Changes made
<Files changed and why — link to specific fixes>

## Next steps
<What to try next session>
```

## Guidelines

- **Track codebase version**: always note the base commit hash and branch. If you
  make changes mid-session, note which run used which state.
- **Be concise**: the journal is for replication and context, not a transcript.
  Link to conversation transcripts for full detail.
- **One session, one scope**: if you switch benchmarks or objectives, start a new file.
  The scope can be revised mid-session if discoveries lead elsewhere — just note the pivot.
