# Update OpenSpec

Audit `openspec/specs/` against the current source code and update specs that have drifted.

## When to use

Run this after any PR that changes a public contract — new method signatures, removed fields,
changed invariants, new layers. Also useful as a periodic drift check.

## Instructions

For each layer listed below, read the source file(s) and the corresponding spec, then identify drift:

| Layer | Source | Spec |
|-------|--------|------|
| Core types | `src/cube_harness/core.py` | `openspec/specs/core/spec.md` |
| Agent | `src/cube_harness/agent.py` | `openspec/specs/agent/spec.md` |
| Tool (OTel wrapper) | `src/cube_harness/tool.py` | `openspec/specs/tool/spec.md` |
| LLM | `src/cube_harness/llm.py` | `openspec/specs/llm/spec.md` |
| Episode | `src/cube_harness/episode.py` | `openspec/specs/episode/spec.md` |
| Experiment + runners | `src/cube_harness/experiment.py`, `src/cube_harness/exp_runner.py` | `openspec/specs/experiment/spec.md` |
| Storage | `src/cube_harness/storage.py`, `src/cube_harness/summary.py` | `openspec/specs/storage/spec.md` |
| Metrics / OTel | `src/cube_harness/metrics/` | `openspec/specs/metrics/spec.md` |
| XRay / Analyze | `src/cube_harness/analyze/` | `openspec/specs/analyze/spec.md` |
| MCP server | `src/cube_harness/mcp/` | `openspec/specs/mcp/spec.md` |

Focus on layers relevant to the current change, or all layers for a full audit.

## What counts as drift

- Public method added, removed, or renamed
- Required field added or removed from a config/model
- Invariant no longer holds (or a new one was added)
- A "Gotcha" the spec doesn't mention but the code exhibits
- A constraint in the spec that no longer matches the implementation

## Decision rule

**Minor drift** (1–3 spec lines, no contract change): edit the spec directly.

**Substantive drift** (new capability, breaking change, or multiple invariants affected):
create `openspec/changes/<name>/` with `proposal.md` + `deltas.md` (ADDED / MODIFIED / REMOVED
sections in target-state language) before editing the spec. This signals to the team that
a real contract change is in flight.

## Output

Report per-layer: `OK`, `UPDATED`, or `CHANGE PROPOSED`. Show diffs for every change made.
If no drift found, say so — don't open a PR for a no-op.
