# Deltas — Episode Status File

Applies to: `openspec/specs/episode/spec.md`, `openspec/specs/experiment/spec.md`,
`openspec/specs/storage/spec.md`

---

## episode/spec.md

### ADDED — `EpisodeStatus` dataclass (new module: `cube_harness/episode_status.py`)

`EpisodeStatus` lives in its own module to avoid circular imports between `episode.py`
and `storage.py`. Both import from `episode_status.py`.

```python
Status = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "STALE", "MAX_STEPS_REACHED"]

IN_FLIGHT_STATUSES  = frozenset({"QUEUED", "RUNNING"})
TERMINAL_STATUSES   = frozenset({"COMPLETED", "FAILED", "CANCELLED", "STALE", "MAX_STEPS_REACHED"})
RETRIABLE_STATUSES  = frozenset({"FAILED", "CANCELLED", "STALE"})

@dataclass
class EpisodeStatus:
    status: Status
    task_id: str
    episode_id: int
    started_at: float
    ended_at: float | None = None
    last_heartbeat_at: float | None = None
    current_step: int = 0
    reward: float | None = None
    had_step_errors: bool = False
    error_type: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    extra: dict = field(default_factory=dict)
```

Status meanings:

| Status             | Written by                     | Meaning                                                          |
|--------------------|--------------------------------|------------------------------------------------------------------|
| `QUEUED`           | driver (`_pre_claim`)          | Submitted to Ray; worker hasn't started yet                      |
| `RUNNING`          | worker (`_open_status`)        | Worker is actively executing the episode                         |
| `COMPLETED`        | worker (`finally`)             | Loop reached natural end (`done=True` or agent gave up)          |
| `MAX_STEPS_REACHED`| worker (`finally`)             | Loop exhausted `max_steps` without `done=True`; not retriable    |
| `FAILED`           | worker (`finally`)             | Unhandled exception propagated out of `_run_loop`                |
| `CANCELLED`        | driver (`_kill_stale_workers`) | Step heartbeat went stale; driver force-killed the worker        |
| `STALE`            | `sweep_stale_statuses` (driver)| Worker died without writing a terminal status                    |

`MAX_STEPS_REACHED` is terminal but **not** retriable — the agent legitimately ran out of
its step budget; retrying from a fresh initial state would just truncate again.

### ADDED — `next_retry_count(prior: EpisodeStatus | None) -> int`

Helper in `episode_status.py`:

- `prior is None` → `0` (original attempt)
- `prior.status in IN_FLIGHT_STATUSES` → `prior.retry_count` (idempotent re-pre-claim)
- `prior.status in TERMINAL_STATUSES` → `prior.retry_count + 1`

### MODIFIED — `Episode._run_loop` lifecycle

Before (no status file):

- Trajectory written incrementally; no terminal signal on crash.

After:

1. `Episode._open_status(trajectory_id)` is called at the top of `_run_loop`.
   - Reads any prior `status.json`.
   - If prior status is terminal and `allow_overwrite=True`, archives the old episode directory.
   - Writes `status=RUNNING` with `last_heartbeat_at=now` and `current_step=0`.
   - This covers stuck `setup_fn` (env reset, container boot).
2. At the start of each turn (before `agent.step()` and `step_fn()`):
   - Updates `last_heartbeat_at` and `current_step` in-place and writes to `status.json`.
3. In `finally`: sets `ended_at` and `last_heartbeat_at` to `now`, then writes terminal status:
   - `COMPLETED` — loop exited normally (`done=True` or agent returned no actions).
   - `MAX_STEPS_REACHED` — `turns >= max_steps` and `done` was never `True`.
   - `FAILED` — unhandled exception; also sets `error_type` and `error_message`.

### INVARIANT

`status.json` (`QUEUED` or `RUNNING`) is always written before any step file. A
missing `status.json` in an episode directory with a config means the worker died
before entering `_run_loop` — treat as retriable.

---

## experiment/spec.md

### REMOVED — `retry_failed: bool` field on `Experiment`

The `retry_failed` flag and the 2×2 `resume` × `retry_failed` selection table are
removed. Retry behaviour is now fully controlled by `resume=True` combined with the
status file: retriable statuses (`FAILED`, `CANCELLED`, `STALE`) are picked up
automatically when `resume=True`, so a separate flag is redundant.

### REMOVED — `_is_trajectory_successful`, `_load_successful_trajectory_ids`, `_load_started_trajectory_ids`, `_find_episodes_to_relaunch`

Replaced by status-driven selection via `storage.list_episode_statuses()`.
No trajectory deserialisation in the retry decision path.

### MODIFIED — Resume semantics

`Experiment.get_episodes_to_run()` is status-driven and keyed off `resume`:

| `resume` | Episodes returned |
|----------|-------------------|
| `False`  | All episodes from scratch |
| `True`   | Episodes with no `status.json` (never started), plus retriable statuses (`FAILED`, `STALE`, `CANCELLED`) with `retry_count < max_retries` |

`RUNNING` / `QUEUED` (in-flight) are never returned. `COMPLETED` and
`MAX_STEPS_REACHED` (terminal, non-retriable) are skipped.

When `resume=True`, `sweep_stale_statuses` runs first so orphaned `RUNNING`/`QUEUED`
entries become `STALE` and are eligible for retry.

### ADDED — `sweep_stale_statuses` (in `experiment.py`)

Marks in-flight episodes whose worker is dead as `STALE`:

- `RUNNING` with `last_heartbeat_at` older than `step_timeout_s + cancel_grace_s` → `STALE`
- `QUEUED` with `started_at` older than `orphan_threshold_s` (default 1 h) → `STALE`

Run at:

- The start of `get_episodes_to_run()` when `resume=True`
- After `ray.shutdown()` in `_run_with_ray_impl`

### ADDED — `max_retries: int = 3` field on `Experiment`

Controls how many times a failed episode is retried. Episodes at `retry_count >= max_retries`
are excluded from future runs.

### ADDED — `max_retry_rounds: int = 3` parameter on `run_with_ray` and `run_sequentially`

After each round, the runner calls `_has_retriable_episodes(exp)`. If any exist and
`round_num < max_retry_rounds`, it runs another round with `resume=True` on the same
`output_dir`. The loop repeats until no retriable episodes remain or the round budget
is exhausted. The final `ExpResult` aggregates trajectories and failures across all
rounds.

Pass `max_retry_rounds=0` to disable auto-retry entirely.

---

## storage/spec.md

### ADDED — `Storage` Protocol methods

```python
def write_episode_status(self, trajectory_id: str, status: EpisodeStatus) -> None: ...
    # Atomic: write to .tmp sibling, then os.replace()

def read_episode_status(self, trajectory_id: str) -> EpisodeStatus | None: ...
    # Returns None if status.json does not exist or is corrupt

def archive_episode(self, trajectory_id: str) -> None: ...
    # Renames the episode directory to <id>.archived_<ts>/ preserving per-attempt history
```

### ADDED — `FileStorage` implementation

- `write_episode_status`: delegates to `EpisodeStatus.write(path)`, which writes
  atomically via a `.tmp` sibling + `os.replace()`.
- `read_episode_status`: delegates to `EpisodeStatus.read(path)`; returns `None` on
  missing or corrupt file.
- `list_episode_statuses() -> dict[str, EpisodeStatus]`: returns `{trajectory_id: status}`
  for every non-archived episode directory that has a `status.json`. Used by the
  retry loop and `sweep_stale_statuses`.

Status files live at `episodes/{trajectory_id}/status.json`.

---

## exp_runner.py (not a spec file — implementation notes)

### Pre-claim (`_pre_claim`)

Before submitting episodes to Ray, the driver calls `_pre_claim(storage, episode)`:

- Reads any prior `status.json`.
- If prior is terminal, calls `storage.archive_episode(traj_id)` (public `Storage` Protocol method) to preserve per-attempt history.
- Writes `status=QUEUED` with `last_heartbeat_at=None` and `retry_count` from `next_retry_count(prior)`.
- Only called for episodes that will actually run — in `debug_limit` mode, episodes beyond the limit are not pre-claimed.

A concurrent runner sees `QUEUED` / `RUNNING` and skips those episodes.

### Step-timeout kill (`_kill_stale_workers`)

In the driver poll loop, after each `ray.wait()` batch:

- Reads `status.json` for each in-progress ref.
- Skips refs in `QUEUED` state (no heartbeat to check yet).
- If `last_heartbeat_at` age exceeds `step_timeout_s + cancel_grace_s`: calls
  `ray.cancel(ref, force=True)`, re-reads status (to avoid clobbering a race-won
  terminal write), and writes `status=CANCELLED` with `error_type="StepTimeout"`.

### End-of-run STALE sweep

After `ray.shutdown()` in `_run_with_ray_impl`, `sweep_stale_statuses` is called.
Any `RUNNING` entries whose worker was just killed by `ray.shutdown()` (and therefore
never wrote a terminal status) are swept to `STALE` so the next retry round can pick
them up.
