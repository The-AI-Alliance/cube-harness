# Deltas: Trajectory Judge

Applies to: `openspec/specs/analyze/spec.md` (primary), `openspec/specs/storage/spec.md` (minor)

---

## ADDED — `cube_harness/analyze/judge.py`

### Public types

- `Outcome` — enum: `success`, `success_lucky`, `failure`, `indeterminate`
- `BlameCategory` — enum: `task_unclear`, `model_capability`, `tool_or_env`, `scaffolding`, `eval_brittle`, `submission_format`, `none`
- `Confidence` — enum: `high`, `medium`, `low`
- `EvidenceItem` — `TypedBaseModel(step: int, quote: str)`
- `JudgeOutput` — `TypedBaseModel` with fields: `outcome`, `primary_blame`, `primary_blame_confidence`, `other_blames`, `summary`, `evidence`, `hypothesis`, `hypothesis_confidence`, `analysis`

### Public functions

- `judge_episode(trajectory, *, agent_config, task_description, related_trajectories=None, model, codebase_files=None) -> JudgeOutput`
  — Invokes Claude Code via Python API on a single trajectory.

- `judge_experiment(output_dir, *, model, n_parallel, overwrite) -> dict[str, JudgeOutput]`
  — Batch judges all episodes in an output directory. Writes `judge_output.json` per episode. Skips already-judged episodes unless `overwrite=True`.

### Invariants

- `primary_blame` must be a single `BlameCategory` value; `none` is valid and required for successful episodes.
- `other_blames` must not repeat `primary_blame`.
- `evidence` must be non-empty when `primary_blame != "none"`.
- `JudgeOutput` is serializable to JSON via `model_dump()`.

---

## MODIFIED — `openspec/specs/storage/spec.md`

### `EpisodeRecord.judge_output`

The `judge_output` field (type `JudgeOutput | None`, currently `null` in the atlas-eval-log schema) is now formally populated by `judge_experiment()`.

- Written by `judge_experiment()` into `episodes/<trajectory_id>/episode_record.json`.
- Written by a separate pass after the episode completes; never written during the episode loop.
- `null` until a judge pass has been run.

---

## ADDED — `openspec/specs/analyze/spec.md` (new section)

### Judge CLI

Module `cube_harness.analyze.judge` is also a CLI entry point:

```
python -m cube_harness.analyze.judge <output_dir> [--model MODEL] [--summary] [--overwrite]
```

- `--summary` prints aggregated blame distribution and outcome counts to stdout.
- `--overwrite` re-judges episodes that already have a `judge_output`.
- Default model: `claude-opus-4-7`.

---

## NOT CHANGED

- `Trajectory`, `TrajectoryStep`, `AgentOutput` — no structural changes.
- `Episode`, `Experiment`, `exp_runner` — judge is post-hoc; episode loop is unmodified.
- `Storage` protocol — `judge_output.json` is a sidecar file written by the judge, not a Storage protocol method.
- Any benchmark or agent contracts.
