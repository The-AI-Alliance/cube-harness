# Analyze — XRay Viewer

**Module:** `cube_harness.analyze`

## Purpose

Gradio-based web UI for exploring experiment outputs. Browse agents → tasks → seeds,
step through trajectories, inspect observations (screenshots, AXTree, HTML, reward),
view agent reasoning, and compare runs across experiments.

## Public API

### Entry point
```bash
make xray                          # Makefile target
# or
uv run python -m cube_harness.analyze.xray --results-dir <path>
```

### `XRayState` (dataclass)
Holds all mutable viewer state. Captured by Gradio handler closures. Not a
serializable model — it's UI-only state that lives for the duration of a viewer
session.

Key fields:
- `trajectories: list[Trajectory]` — currently loaded set
- `current_trajectory`, `step` — navigation cursor
- `_storages: list[FileStorage]` — one per loaded experiment dir
- `_traj_storages: list[FileStorage]` — index-aligned with trajectories
- `_exp_tags` — timestamp tag per storage (for disambiguation)
- `_bg_loading_done` / `_bg_gen` — background loading coordination

### `inspect_results` (`cube_harness.analyze.inspect_results`)
CLI-style inspection helpers used by the viewer and exported for ad-hoc scripts.

### `xray_utils` (`cube_harness.analyze.xray_utils`)
Formatting and data-extraction helpers (HTML rendering, trace fragments, step
summaries).

## UI model

A "UI step" is one environment observation paired with the agent action that
follows it. Navigation moves between environment steps. For UI step N:
- Shows the Nth `EnvironmentOutput` (screenshot, axtree, reward, etc.)
- Shows the `AgentOutput` that immediately follows it (actions, LLM call, thoughts)

## Invariants

1. Read-only — the viewer never writes to experiment dirs.
2. Handles V2 (episodes/) and V1 (jsonl) layouts via `FileStorage`.
3. Background loading: a worker thread populates `trajectories` incrementally;
   stale threads self-abort by comparing `_bg_gen`.
4. Displays `_missing=True` stub trajectories (planned but never ran) distinctly.
5. Injects `_failure_text` from `failure.txt` into metadata when a trajectory has
   no `end_time` — so failed episodes show their stack trace in the UI.

## Gotchas

- Gradio state is per-tab. Closing and reopening the browser resets the view; the
  server keeps running.
- Large trajectories (thousands of steps) are loaded lazily — switching trajectories
  may have noticeable latency on first open.
- The viewer caches step deserialization in-memory per session; very long sessions
  with many open trajectories can grow memory use.
