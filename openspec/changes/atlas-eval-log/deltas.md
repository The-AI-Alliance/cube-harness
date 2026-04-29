# Deltas — Atlas EvalLog

Changes against current specs in `openspec/specs/`.

---

## `episode/spec.md`

### MODIFIED — `Episode.run`

`Episode.run` collects action schemas immediately after the action set is resolved and
passes them via the `extra_metadata` parameter:

```python
extra_metadata = {"action_schemas": [a.as_dict() for a in action_set]}
return self._run_loop(setup_fn, step_fn, close_fn, agent, extra_metadata=extra_metadata)
```

`_run_loop` accepts `extra_metadata: dict | None = None` and merges it into
`trajectory.metadata`:

```python
metadata={
    "task_id": task_id,
    "agent_name": agent_name,
    **(extra_metadata or {}),    # injects action_schemas
    **env_output.info,
},
```

**Why:** action schemas are only available at `task.reset()` time. Persisting them in
metadata lets `export_eval_log()` reconstruct `EpisodeRecord.tool_names` post-hoc without
re-instantiating the task or environment.

---

## `experiment/spec.md`

### MODIFIED — `Experiment.export_eval_log`

Signature change: `output_path: Path | None` → `output_dir: Path | None` (now writes
structured records instead of a single flat JSONL).

```python
def export_eval_log(
    self,
    output_dir: Path | None = None,
    git_cwd: str | None = None,
) -> EvalLog
```

- Builds one `ExperimentRecord` (agent info, benchmark metadata, git provenance).
- Iterates `storage.list_trajectory_ids()`.
- Builds one `EpisodeRecord` per trajectory (reads action schemas from
  `trajectory.metadata["action_schemas"]`).
- Resolves `task_config` from the episode config index for `task_version_hash` and `seed`.
- Writes to `output_dir` or `self.output_dir` by default:
  - `experiment_record.json` — one JSON object
  - `episodes/<trajectory_id>/episode_record.json` — one per episode
- Returns the in-memory `EvalLog`.

No task re-instantiation; all data comes from persisted files.

---

## `storage/spec.md`

### ADDED — output files

```
<output_dir>/
├── experiment_record.json                    ← ADDED: ExperimentRecord (once per experiment)
└── episodes/
    └── <trajectory_id>/
        └── episode_record.json               ← ADDED: one per completed episode
```

Written by `Experiment.export_eval_log()`. Not written automatically during runs;
must be called explicitly post-experiment. For ATLAS submission, call
`eval_log.to_jsonl(path)` to assemble a flat JSONL from the per-trajectory records.

---

## New module — `eval_log/spec.md`

### ADDED — `src/cube_harness/eval_log.py`

Eight public classes:

| Class | Purpose |
|---|---|
| `UsageSummary` | Aggregated LLM token/cost stats across an episode |
| `AgentInfo` | Agent descriptor: agent_id, config, dependency versions, git provenance |
| `BenchmarkSubset` | Benchmark subset descriptor for MNAR propensity correction |
| `JudgeConfig` | Configuration of the LLM judge (optional) |
| `JudgeOutput` | Per-episode judge assessment: difficulty, feasibility, failure root cause |
| `Verifier` | Task verifier reference: GitHub URL + source code |
| `ExperimentRecord` | Experiment-level record → `experiment_record.json` |
| `EpisodeRecord` | Episode-level record → `episodes/<id>/episode_record.json` |
| `EvalLog` | Two-level container: `save(output_dir)` / `load(output_dir)` / `to_jsonl(path)` |

Key invariants:

- `AgentInfo.agent_id` — SHA-256 of sorted serialized agent config JSON. Stable across
  runs with the same config. Primary matrix row key for ATLAS.
- `AgentInfo` has no `tools` field — tools are episode-specific (same agent gets different
  action sets on different tasks). See `EpisodeRecord.tool_names`.
- `ExperimentRecord.experiment_id` — SHA-256(experiment_name + output_dir)[:16]. Stable
  within a run; unique across different output directories.
- `EpisodeRecord.experiment_id` — FK linking back to `ExperimentRecord`. Consistent across
  all episode records in an experiment.
- `EpisodeRecord.task_version_hash` — SHA-256 of `TaskConfig.model_dump_json()`. Detects
  silent benchmark drift across submissions.
- `EpisodeRecord.tool_names` — read from `trajectory.metadata["action_schemas"]` at export
  time. Empty list for trajectories produced before this field was added.
- `BenchmarkSubset` — automatically derived from the benchmark object; no user input required.
- Per-trajectory storage: retried episodes overwrite stale records naturally since the new trajectory occupies the same directory.
- `to_jsonl(path)`: submission helper that assembles a flat JSONL from all per-trajectory records. Output is one JSON object per line, no envelope, no cube-harness dependency to read.
