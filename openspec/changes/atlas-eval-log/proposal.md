# RFC: Atlas EvalLog — Structured Evaluation Records

**Status:** DRAFT  
**PR:** [#297](https://github.com/The-AI-Alliance/cube-harness/pull/297)  
**Author:** Alexandre Lacoste  
**Date:** 2026-04-27  
**Related:** [EEE alignment doc](../../../../atlas/docs/agent_eval_record_proposal.md), PR #315 (episode-status)

---

## Problem

cube-harness produces trajectories, but has no standard output format for the
community matrix **M[agent, task] = reward** that the ATLAS project needs.

Three concrete gaps:

1. **No stable agent identity.** Two runs of the same agent config can't be matched
   without re-reading the full config. ATLAS needs a stable row key that is cheap to
   index.

2. **Action schemas are ephemeral.** The action set given to the agent at runtime is
   never persisted. Post-hoc analysis (what tools did this agent have on this task?)
   requires re-instantiating the task — expensive and sometimes impossible.

3. **No standardized episode export.** Downstream tools (ATLAS server, leaderboards,
   other frameworks) each extract reward/cost/metadata differently from our trajectory
   files. A single JSONL line per episode, readable without cube-harness, solves this.

---

## Scope

- New module `eval_log.py` with five public classes: `UsageSummary`, `AgentInfo`,
  `TaskInfo`, `TaskEvalRecord`, `EvalLog`.
- `Episode._run_loop` stores `action_schemas` in `trajectory.metadata` so eval records
  can be built post-hoc without re-instantiating tasks.
- `Experiment.export_eval_log()` batch-exports one `TaskEvalRecord` per trajectory to
  `<output_dir>/eval_log.jsonl`.
- 42 unit tests, no regressions.

Fields are named to mirror EEE destinations (see forward-compatibility map) so adapting
to Elron's schema review is a small rename-and-nest, not a redesign.

---

## Schema (v1)

All records are plain JSON, one per line (JSONL). No cube-harness dependency to read.

### `TaskEvalRecord`

```jsonc
{
  // ── Task ──────────────────────────────────────────────────────────────────
  "task": {
    "benchmark_id": "workarena-l1",          // → EEE evaluation_name (slugified)
    "benchmark_name": "WorkArena-L1",
    "benchmark_version": "1.2.0",            // → EEE metadata.benchmark_version
    "benchmark_description": "...",
    "benchmark_authors": ["..."],
    "benchmark_tags": ["gui", "enterprise"],  // → EEE metadata.benchmark_tags

    "task_id": "workarena.create_incident",  // → EEE sample_id
    "task_version_hash": "9c2f...",          // SHA-256 of TaskConfig JSON → EEE sample_hash
    "seed": 42,                              // → EEE eval_conditions.seed (Elron's ext)
    "split": "test",                         // → EEE metadata.split

    "abstract_description": "...",           // broad category description
    "first_observation_text": "...",         // actual text the agent saw at runtime
                                             // distinct from EEE input.raw (static spec)
                                             // primary field for ATLAS task embedding
    "recommended_max_steps": 15,
    "extra_info": {}                         // TaskMetadata.extra_info passthrough
  },

  // ── Agent ─────────────────────────────────────────────────────────────────
  "agent": {
    // Identity
    "agent_id": "<sha256>",                  // SHA-256(sorted config JSON) — stable row key
    "config_type": "ReactAgentConfig",       // → EEE agent_system (Elron's ext)
    "config": {...},                         // full serialized agent config
    "llm_model": "gpt-4o-2024-11-20",       // → EEE agent_system.models[0].name

    // Capabilities (episode-specific: same agent gets different tools per task)
    "tools": [...],                          // full schemas in litellm format
    "tool_names": ["browser_click", ...],    // → EEE agent_system.tools

    // Runtime environment
    "framework_version": "0.4.2",
    "dependency_versions": {                 // → EEE agent_system.dependency_versions
      "cube-harness": "0.4.2",
      "litellm": "1.65.0"
    },

    // Git provenance → EEE agent_system.git_*
    "git_commit": "abc123",
    "git_remote_url": "https://github.com/org/repo/tree/abc123",
    "git_is_dirty": false,

    // LLM embedding warm-start (ATLAS cold-start)
    "description": "ReAct agent, gpt-4o, browser tools, no memory"
  },

  // ── Outcome ───────────────────────────────────────────────────────────────
  "success": true,                           // reward > 0
  "reward": 1.0,                             // → EEE evaluation.score (v2 rename)
  "reward_breakdown": {                      // → EEE evaluation.breakdown (v2 nest)
    "done": true,
    "form_submitted": true
  },
  "error_type": null,                        // exception class name if any step raised

  // ── Trajectory summary ────────────────────────────────────────────────────
  "n_steps": 25,                             // total steps (agent + env)
  "n_agent_steps": 12,                       // agent decision turns
  "n_env_steps": 13,                         // environment executions
  "wall_time_s": 47.3,
  "usage": {                                 // → EEE session_accounting (Elron's ext)
    "prompt_tokens": 12400,
    "completion_tokens": 840,
    "total_tokens": 13240,
    "cached_tokens": 9800,
    "cache_creation_tokens": 0,
    "total_cost_usd": 0.034,
    "n_llm_calls": 12
  },

  // ── Provenance ────────────────────────────────────────────────────────────
  "run_id": "my_exp_workarena.create_incident_ep0",
  "trajectory_id": "workarena.create_incident_ep0",
  "timestamp": 1745000000.0,
  "framework_version": "0.4.2",

  // ── Declaration (ATLAS MNAR correction) ──────────────────────────────────
  // Required for ATLAS submissions. Feeds a propensity model that downweights
  // cherry-picked runs in the matrix factorization (Missing Not At Random).
  "declaration": {
    "motivation": "capability_probe",        // "capability_probe" | "leaderboard"
                                             // | "training_data" | "debugging"
    "task_selection_method": "random",       // "random" | "difficulty_stratified"
                                             // | "domain_filtered" | "cherry_picked"
    "compute_budget": "full_benchmark"       // "full_benchmark" | "partial" | "targeted"
  }
}
```

### Why `declaration` exists

The ATLAS matrix M[agent, task] is extremely sparse: labs don't evaluate uniformly at
random. They cherry-pick strong agents on hard benchmarks, hide results where agents
fail, and gravitate toward popular benchmarks. If the matrix factorization trains
naively on this biased data, recovered difficulty estimates reflect *what people chose
to run*, not true task difficulty (Missing Not At Random / MNAR).

`declaration` fields let submitters self-report their selection intent. ATLAS feeds
these into a propensity model: `P(observed | agent, task, declaration)`. Observations
from cherry-picked runs are downweighted via Inverse Propensity Scoring. The anchor
set (tasks every submitter must run) handles the mandatory portion cleanly; declaration
corrects the free-choice portion.

### Why no `eval_function_ref`

`input.reference` (EEE) already carries the privileged ground truth. The evaluation
function pointer is derivable from `benchmark_name + task_id`, goes stale on refactors,
and is not used by any ATLAS component — the feasibility judge operates from trajectories
and `input.reference`, not from reading eval source code.

---

## Implementation

### Files changed

| File | Change |
|------|--------|
| `src/cube_harness/eval_log.py` | **NEW** — `UsageSummary`, `AgentInfo`, `TaskInfo`, `TaskEvalRecord`, `EvalLog`; remove `eval_function_ref`; add `declaration` |
| `src/cube_harness/episode.py` | Store `action_schemas` in `trajectory.metadata` via `extra_metadata` parameter |
| `src/cube_harness/experiment.py` | Add `export_eval_log(output_path, git_cwd) -> EvalLog` |
| `tests/test_eval_log.py` | **NEW** — 42 unit tests |

### `episode.py` delta

One addition to `_run_loop`, immediately after `agent = self.config.agent_config.make(action_set)`:

```python
# Persisted in metadata so eval records can be built post-hoc without
# re-instantiating the task (action set is only available at task.reset() time).
action_schemas = [a.as_dict() for a in action_set]
```

And in the `trajectory.metadata` dict:

```python
metadata={
    "task_id": task_id,
    "agent_name": agent_name,
    "action_schemas": action_schemas,    # ← added
    **env_output.info,
},
```

### `experiment.py` delta

New public method on `Experiment`:

```python
def export_eval_log(
    self,
    output_path: Path | None = None,
    git_cwd: str | None = None,
) -> EvalLog
```

Iterates `storage.list_trajectory_ids()`, builds one `TaskEvalRecord` per trajectory,
writes `<output_dir>/eval_log.jsonl`. No task re-instantiation; all data comes from
persisted trajectories and episode configs.

### Output location

```
<output_dir>/
├── experiment_config.json
├── experiment_summary.json
├── eval_log.jsonl          ← one JSON line per episode
└── episodes/
    └── <trajectory_id>/
        ├── episode_config.json
        └── ...
```

### Interaction with upcoming PRs

- **PR #315 (episode-status):** status files live inside `episodes/<id>/`. No conflict.
  `export_eval_log` reads status indirectly via `list_trajectory_ids()` — if #315 lands
  first, we can filter on `COMPLETED` status instead of scanning trajectories.
- **PR #314 (env-cube credentials):** no overlap.

---

## EEE Forward-Compatibility Map

| v1 field | v2 EEE destination | Notes |
|---|---|---|
| `task.benchmark_id` / `task.benchmark_name` | `evaluation_name` | slugified name |
| `task.task_id` | `sample_id` | open question: semantic vs numeric key |
| `task.task_version_hash` | `sample_hash` | scope difference (full config vs input hash) |
| `task.seed` | `eval_conditions.seed` (Elron's ext) | |
| `task.split` / `task.benchmark_version` / `task.benchmark_tags` | `metadata` (string escape hatch) | |
| `task.first_observation_text` | new top-level field | needs EEE schema extension |
| `agent.*` | `agent_system.*` (Elron's ext) | nested + renamed |
| `reward` | `evaluation.score` | rename |
| `reward_breakdown` | `evaluation.breakdown` | nest inside `evaluation` |
| `usage.*` | `session_accounting.*` (Elron's ext) | renamed fields |
| `declaration` | new top-level field | needs EEE schema extension |

---

## Open Questions (deferred to Elron)

1. **`sample_id` semantics.** EEE often uses a numeric row index (`gsm8k_0001`).
   CUBE tasks use semantic names (`workarena.create_incident`). Should `sample_id`
   carry the semantic name, or should we add a separate `task_id` and keep `sample_id`
   numeric?

2. **`sample_hash` scope.** EEE defines it as `hash(input.raw + input.reference)`.
   ATLAS wants to detect any drift in the task config (env setup, reward function, not
   just the prompt). Proposal: keep `sample_hash` as EEE defines it; ATLAS submits a
   separate `task_version_hash` inside `metadata` or as an extension.

3. **`metadata` string constraint.** The EEE escape hatch is `string → string`, which
   forces `seed` into a string. Worth proposing EEE relax `metadata` to `string → any`
   for agentic use cases?

4. **`agent_system` at instance vs aggregate level.** Elron's PR adds `agent_system` to
   `evaluation_results[]` in the aggregate schema. For ATLAS, per-episode records must
   be self-contained — the agent description needs to be at the instance level. Separate
   EEE PR needed?
