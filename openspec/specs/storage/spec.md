# Storage

**Module:** `cube_harness.storage`, `cube_harness.summary`

## Purpose

Persist trajectories, episode configs, and experiment summaries to disk. The
`Storage` Protocol defines the contract; `FileStorage` is the default implementation.
Supports two on-disk layouts: **V2** (current, per-episode directories with compressed
step files) and **V1** (legacy JSONL, read-only fallback).

## Public API

### Protocol
```python
class Storage(Protocol):
    def save_trajectory(self, trajectory: Trajectory, allow_overwrite: bool = False) -> None
    def save_step(self, step: TrajectoryStep, trajectory_id: str, step_num: int) -> None
    def save_episode_config(self, episode_config: EpisodeConfig) -> None
    def update_experiment_summary(self, trajectory: Trajectory) -> None
```

Custom backends (cloud storage, DB) must implement all four.

### `FileStorage`
Writes V2 only. Reads V2 + V1.

```python
class FileStorage:
    def __init__(self, output_dir: str | Path)

    # Writes (V2)
    def save_trajectory(traj, allow_overwrite=False)
    def save_step(step, trajectory_id, step_num)
    def save_episode_config(ep_config)
    def save_failure(trajectory_id: str, stack_trace: str)      # failure.txt
    def update_experiment_summary(trajectory)                   # experiment_summary.json (flock-protected)

    # Reads (V2 + V1 fallback)
    def load_trajectory(trajectory_id) -> Trajectory
    def load_step(trajectory_id, step_index) -> TrajectoryStep
    def load_trajectory_metadata(trajectory_id) -> Trajectory   # steps=[]
    def load_all_trajectory_metadata() -> list[Trajectory]
    def load_all_trajectories(exp_dir=None) -> list[Trajectory]
    def list_trajectory_ids() -> list[str]
    def list_trajectory_ids_with_mtime() -> dict[str, float]
    def load_missing_trajectory_stubs() -> list[Trajectory]     # stubs with metadata._missing=True
    def list_episode_configs() -> list[Path]
    def load_episode_config(config_path) -> EpisodeConfig

    # Logs
    def get_log_path(trajectory_id) -> Path
    def has_logs(trajectory_id) -> bool
    def load_logs(trajectory_id) -> str
```

## On-disk layouts

### V2 (current, written by all runs)
```
<output_dir>/
├── experiment_config.json
├── experiment_summary.json           # aggregated stats; flock-protected
└── episodes/
    └── <trajectory_id>/              # f"{task_id}_ep{episode_id}"
        ├── episode_config.json
        ├── episode.metadata.json     # Trajectory without steps
        ├── steps/
        │   ├── 000_obs.msgpack.zst   # msgpack + zstd(level=3)
        │   └── 001_act.msgpack.zst
        ├── episode_summary.jsonl     # SummaryProcessor output
        ├── failure.txt               # on crash before success
        └── logs/<trajectory_id>.log
```

Archived episodes (overwritten) are renamed to `<id>.archived_<ts>/`.

### V1 (legacy, read-only)
```
<output_dir>/
├── <trajectory_id>.metadata.json
├── <trajectory_id>.jsonl             # one step per line
├── llm_calls/
│   └── <step_id>_<llm_call_id>.json  # resolved on load
└── trajectories/                      # alt location for V1 layout
    └── ...
```

## Invariants

1. Writes are always V2; V1 is read-only.
2. Saving a trajectory that already exists raises `FileExistsError` unless
   `allow_overwrite=True` (triggers archive).
3. Step files: `{nnn:03d}_{obs|act}.msgpack.zst`. Suffix = `obs` for
   `EnvironmentOutput`, `act` for `AgentOutput`.
4. `update_experiment_summary` uses `fcntl.LOCK_EX` — concurrent episodes across
   workers serialize on the summary update. Stats are accumulated incrementally.
5. `load_*` methods resolve V2 first; fall back to V1 if not found.
6. V1 LLM call references are resolved at load time — referenced files must exist
   or `FileNotFoundError` is raised.

## `SummaryProcessor` (`cube_harness.summary`)

Writes `episode_summary.jsonl` (one line per step) and updates
`experiment_summary.json` on episode completion. The summary captures aggregate
stats without requiring loading full trajectories.

### `ExperimentSummary`
```python
n_episodes: int
n_completed: int
n_errored: int
total_reward: float
avg_reward: float
total_prompt_tokens: int
total_completion_tokens: int
total_cost: float
updated_at: str          # ISO-8601 UTC
```

## Gotchas

- msgpack+zstd is fast but opaque to `cat`/`jq`. Use `load_step()` or the XRay viewer
  to inspect steps. The V1 JSONL format is human-readable but no longer written.
- `experiment_summary.lock` persists after the lock is released (flock metadata).
  Safe to leave; delete manually if you want a clean dir.
- V1 support will be dropped in a future release (see DEPRECATED.md). Callers that
  depend on writing JSONL must migrate now.
- `load_all_trajectories()` reads every step of every episode — expensive for large
  experiments. Prefer `load_all_trajectory_metadata()` + on-demand `load_trajectory()`.
