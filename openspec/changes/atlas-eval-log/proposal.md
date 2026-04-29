# RFC: Atlas EvalLog — Two-Level Structured Evaluation Records

**Status:** DRAFT  
**PR:** [#297](https://github.com/The-AI-Alliance/cube-harness/pull/297)  
**Author:** Alexandre Lacoste  
**Date:** 2026-04-28

---

## Problem

cube-harness produces trajectories, but has no standard output format for the community
matrix **M[agent, task] = reward** that the ATLAS project needs.

Three concrete gaps:

1. **No stable agent identity.** Two runs of the same agent config can't be matched
   without re-reading the full config. ATLAS needs a stable row key that is cheap to index.

2. **Action schemas are ephemeral.** The action set given to the agent at runtime is
   never persisted. Post-hoc analysis (what tools did this agent have on this task?)
   requires re-instantiating the task — expensive and sometimes impossible.

3. **No standardized episode export.** Downstream tools (ATLAS server, leaderboards,
   other frameworks) each extract reward/cost/metadata differently from trajectory files.

---

## Scope

- New module `eval_log.py` with eight public classes.
- `Episode.run` persists `action_schemas` in `trajectory.metadata` so per-episode
  tool lists can be read post-hoc without re-instantiating tasks.
- `Experiment.export_eval_log()` writes two files per experiment:
  - `experiment_record.json` — once per experiment (agent, benchmark metadata, git provenance)
  - `eval_log.jsonl` — one JSON line per episode (outcome, usage, trajectory summary)
- 48 tests, no regressions.

---

## Design: Two-Level Schema

The schema mirrors the [Every Eval Ever (EEE)](https://github.com/evaleval/every_eval_ever)
two-level structure: an aggregate record (experiment-level) and per-instance records
(episode-level), linked by a stable FK.

This split eliminates redundancy: agent description and benchmark metadata appear once per
experiment rather than once per episode. For a 500-task benchmark, this is a 500× reduction
in agent config serialization.

### Why two files instead of one envelope JSONL

- `eval_log.jsonl` is streamable: records can be appended immediately after each episode
  without holding the full experiment in memory.
- `experiment_record.json` is indexable: ATLAS and leaderboards can read agent/benchmark
  metadata without scanning all episode lines.
- Consistent with EEE's two-level schema.

### MNAR correction via `benchmark_subset`

The ATLAS matrix M[agent, task] is Missing Not At Random (MNAR): labs cherry-pick which
(agent, task) pairs they run, gravitating toward benchmarks where their agent is strong.
If matrix factorization trains on this biased data naively, recovered difficulty estimates
reflect selection bias rather than true task difficulty.

Rather than asking submitters to self-report intent (high friction, garbage-prone),
`BenchmarkSubset` is derived automatically from the benchmark object:

- `name` — `benchmark_metadata.name`, which includes any subset suffix applied via
  `subset_from_glob` (e.g., `"WorkArena_[level=l1]"`) or `subset_from_list`.
- `n_tasks` — `len(benchmark.task_metadata)` — the denominator for completion rate.
  ATLAS can infer which fraction of the benchmark was run without user input.

This gives ATLAS the structural information it needs for propensity scoring (what subset
was run and how large it was) without requiring submitters to fill in subjective fields.

---

## Schema

### `experiment_record.json`

```jsonc
{
  "experiment_id": "a1b2c3d4e5f6a7b8",        // SHA-256(exp_name + output_dir)[:16]
  "experiment_name": "workarena_l1_gpt4o",
  "timestamp": 1745000000.0,
  "framework_version": "0.4.2",
  "agent": {
    "agent_id": "<sha256>",                    // SHA-256(sorted config JSON)
    "config_type": "ReactAgentConfig",
    "config": {...},
    "llm_model": "gpt-4o-2024-11-20",
    "framework_version": "0.4.2",
    "dependency_versions": {"cube-harness": "0.4.2", "litellm": "1.65.0"},
    "git_commit": "abc123",
    "git_remote_url": "https://github.com/org/repo/tree/abc123",
    "git_is_dirty": false,
    "description": null                        // optional free-form prose for embedding
  },
  "benchmark_name": "WorkArena_[level=l1]",
  "benchmark_version": "1.2.0",
  "benchmark_subset": {
    "name": "WorkArena_[level=l1]",
    "n_tasks": 33,
    "filter": null                             // glob pattern if subset_from_glob was used
  },
  "judge_config": null                         // populated if a post-hoc judge was run
}
```

### `eval_log.jsonl` (one line per episode)

```jsonc
{
  "experiment_id": "a1b2c3d4e5f6a7b8",        // FK to experiment_record.json
  "task_id": "workarena.create_incident",
  "task_version_hash": "9c2f...",             // SHA-256 of TaskConfig JSON
  "seed": 42,
  "split": "test",
  "task_description": "...",                  // TaskMetadata.abstract_description
  "tool_names": ["browser_click", ...],       // episode-specific; varies per task
  "success": true,
  "reward": 1.0,
  "error_type": null,
  "n_steps": 25,
  "n_agent_steps": 12,
  "n_env_steps": 13,
  "wall_time_s": 47.3,
  "usage": {
    "prompt_tokens": 12400,
    "completion_tokens": 840,
    "total_tokens": 13240,
    "cached_tokens": 9800,
    "cache_creation_tokens": 0,
    "total_cost_usd": 0.034,
    "n_llm_calls": 12
  },
  "trajectory_id": "workarena.create_incident_ep0",
  "timestamp": 1745000100.0,
  "verifier": null,                           // optional: ref URL + source code
  "judge_output": null                        // optional: difficulty, feasible, root cause
}
```

---

## Implementation

### Files changed

| File | Change |
|------|--------|
| `src/cube_harness/eval_log.py` | **NEW** — 8 public classes |
| `src/cube_harness/episode.py` | Persist `action_schemas` in `trajectory.metadata` |
| `src/cube_harness/experiment.py` | Replace `export_eval_log` with two-file output |
| `tests/test_eval_log.py` | **NEW** — 48 unit + integration tests |

### `episode.py` delta

In `Episode.run`, immediately after the action set is resolved:

```python
extra_metadata = {"action_schemas": [a.as_dict() for a in action_set]}
return self._run_loop(setup_fn, step_fn, close_fn, agent, extra_metadata=extra_metadata)
```

`_run_loop` merges `extra_metadata` into `trajectory.metadata`. Action schemas are only
available at `task.reset()` time; persisting them here enables post-hoc reconstruction
of `EpisodeRecord.tool_names` without re-instantiating the task.

### Output location

```
<output_dir>/
├── experiment_config.json
├── experiment_summary.json
├── experiment_record.json      ← NEW: one JSON object, written by export_eval_log()
├── eval_log.jsonl              ← NEW: one JSON line per episode
└── episodes/
    └── <trajectory_id>/
        ├── episode_config.json
        └── ...
```

### Interaction with upcoming PRs

- **PR #315 (episode-status):** no conflict; status files live inside `episodes/<id>/`.
- **PR #314 (env-cube credentials):** no overlap.
