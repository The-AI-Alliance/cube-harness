# Deltas — Atlas EvalLog

Changes against current specs in `openspec/specs/`.

---

## `episode/spec.md`

### MODIFIED — `Episode.run` and `Episode._run_loop`

`Episode.run` collects action schemas immediately after the agent is constructed and
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
    **(extra_metadata or {}),    # ← injects action_schemas
    **env_output.info,
},
```

**Why:** action schemas are only available at `task.reset()` time. Persisting them in
metadata lets `export_eval_log()` reconstruct `AgentInfo.tools` post-hoc without
re-instantiating the task or environment.

---

## `experiment/spec.md`

### ADDED — `Experiment.export_eval_log`

```python
def export_eval_log(
    self,
    output_path: Path | None = None,
    git_cwd: str | None = None,
) -> EvalLog
```

- Iterates `storage.list_trajectory_ids()`.
- Builds one `TaskEvalRecord` per trajectory (reads action schemas from
  `trajectory.metadata["action_schemas"]`).
- Resolves `task_config` from the episode config index for `task_version_hash` and `seed`.
- Writes to `output_path` or `<output_dir>/eval_log.jsonl` by default.
- Returns the in-memory `EvalLog`.

No task re-instantiation; all data comes from persisted files.

---

## `storage/spec.md`

### ADDED — output file

```
<output_dir>/
├── eval_log.jsonl    ← ADDED: one JSON line per completed episode
```

Written by `Experiment.export_eval_log()`. Not written automatically during runs;
must be called explicitly post-experiment.

---

## New module — `eval_log/spec.md`

### ADDED — `src/cube_harness/eval_log.py`

Five public classes:

| Class | Purpose |
|---|---|
| `UsageSummary` | Aggregated LLM token/cost stats across an episode |
| `AgentInfo` | Agent descriptor: agent_id, config, tools, git provenance, dependency versions |
| `TaskInfo` | Task descriptor: benchmark metadata, task_id, hash, seed, split, first_observation_text |
| `TaskEvalRecord` | Complete episode record (task + agent + outcome + usage + declaration) |
| `EvalLog` | JSONL collection with `save_jsonl` / `load_jsonl` / `append_record` |

Key invariants:

- `AgentInfo.agent_id` — SHA-256 of sorted serialized agent config JSON. Stable across
  runs with the same config. Primary matrix row key for ATLAS.
- `TaskInfo.task_version_hash` — SHA-256 of `TaskConfig.model_dump_json()`. Changes when
  the task config changes, even if `task_id` stays the same. Detects silent benchmark drift.
- `TaskInfo.first_observation_text` — extracted from the first `EnvironmentOutput` in the
  trajectory (what the agent actually saw, not the static eval spec).
- `TaskEvalRecord.declaration` — MNAR bias correction fields. Optional; required for ATLAS
  community submissions.
- JSONL format: one JSON object per line, no envelope, no cube-harness dependency to read.
