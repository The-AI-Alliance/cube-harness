# RFC: Episode Status File

**Version:** 0.2 — updated after design review 2026-04-27

## Problem

The current resume/retry logic in `experiment.py` determines whether an episode needs
re-running by loading and scanning its full trajectory for `StepError` entries. This
has three failure modes:

**1. Silent crash — dead workers leave no terminal signal.**
If a Ray worker is killed (OOM, `ray.cancel(force=True)`, machine failure), no terminal
status is ever written. The trajectory may be partial or missing. The current code
catches the load failure silently and drops the episode — it neither retries nor counts
it as done.

**2. False positive on recovered errors.**
A `StepError` recorded mid-episode (a tool error that the harness caught and turned into
an error observation, then continued) causes `_is_trajectory_successful` to return
`False`, triggering a retry even though the episode completed normally.

**3. Status requires reading the payload.**
To check whether an episode is done, the entire trajectory must be deserialized. At
scale (thousands of episodes per experiment) this is expensive and fragile — a single
corrupt step file breaks the check for that episode.

**4. Concurrent runner collision.**
If two callers invoke `run_with_ray` on the same output directory simultaneously (e.g.,
a stalled run and a resume attempt), both see the same unstarted episodes and submit
them to Ray twice. There is no mechanism to detect or prevent this.

## Proposed solution

Write a small `status.json` file in each episode directory. Status is the ground truth
for retry decisions. The trajectory is data; `status.json` is control.

---

### Episode statuses

| Status | Written by | Meaning |
|---|---|---|
| `RUNNING` | driver (pre-claim) then worker | Episode is queued or actively executing; skip during resume/retry |
| `COMPLETED` | worker in `finally` | Loop ran to natural end (done or max_steps) |
| `FAILED` | worker in `finally` | Unhandled exception propagated out of the run loop |
| `CANCELLED` | `exp_runner` after `ray.cancel(force=True)` | Explicitly timed out by the harness |
| `STALE` | STALE sweep | Worker was killed; heartbeat went cold; episode will never finish |

**Core invariant:** `RUNNING` with a fresh heartbeat = skip (episode is alive or queued).
`RUNNING` with a stale heartbeat = dead worker. `COMPLETED` = done, never retry.
Everything else (`FAILED`, `STALE`, `CANCELLED`) = eligible for retry if
`retry_count < max_retries`.

`STALE` vs `FAILED`: `FAILED` means the worker lived long enough to catch an exception.
`STALE` means it was killed silently. Both are retried, but the distinction aids
diagnosis: many `STALE` = infrastructure problem (OOM, cluster failure); many `FAILED`
= application bug.

---

### `status.json` schema

```json
{
  "status": "FAILED",
  "task_id": "workarena.servicenow.create-incident",
  "episode_id": 3,
  "started_at": 1745000000.0,
  "ended_at": 1745000042.0,
  "last_heartbeat_at": 1745000040.0,
  "reward": null,
  "had_step_errors": true,
  "error_type": "ConnectionError",
  "error_message": "Environment container failed to start on port 8080",
  "retry_count": 1
}
```

**Fields:**
- `status` — one of the five values above
- `started_at` — set by the driver on pre-claim (see below); overwritten by worker at actual start
- `ended_at` — wall-clock timestamp; `None` until a terminal status is written
- `last_heartbeat_at` — updated by the worker's background thread every 30 s while `RUNNING`; `None` if the worker never started
- `reward` — final reward; `None` until `COMPLETED`
- `had_step_errors` — `True` if any step recorded a `StepError`; informational only, does not affect retry
- `error_type` / `error_message` — exception class name and message for the failure; `None` on success. Enables fast diagnosis without opening log files.
- `retry_count` — how many times this specific episode has been retried; capped at `max_retries`

---

### Pre-claim: preventing concurrent runner collision

Before submitting episodes to Ray, the driver writes `RUNNING` for every episode it
is about to launch. This "pre-claim" step converts all targeted episodes from
"unstarted" (no `status.json`) to `RUNNING` before any other process can see them.

A second runner calling `get_episodes_to_run(resume=True)` or
`get_episodes_to_run(retry_failed=True)` will see `RUNNING` episodes with a fresh
`started_at` and skip them — no collision.

When the Ray worker actually picks up the episode, `_run_loop` overwrites the pre-claim
with a new `RUNNING` entry (same `retry_count`, updated `started_at`) and starts the
heartbeat thread.

Pre-claim only applies to `run_with_ray`. The sequential runner (`run_sequentially`)
has no concurrency to protect against; the worker writes `RUNNING` directly.

---

### Heartbeat: distinguishing alive workers from dead ones

A background thread in the worker's `_run_loop` writes `last_heartbeat_at` to
`status.json` every **30 seconds** while the episode is executing.

**STALE sweep** is the complementary mechanism on the driver side. It scans all episode
directories for `RUNNING` entries with a stale `last_heartbeat_at` and rewrites them
as `STALE`. It is called:

1. **Before every `get_episodes_to_run()` call with `resume=True` or `retry_failed=True`** — cleans up orphaned `RUNNING` from a previous run that crashed without calling `ray.shutdown()`.
2. **After `ray.shutdown()`** — handles the normal case where workers are killed at experiment completion.

Staleness threshold: `now - (last_heartbeat_at or started_at) > 2 * HEARTBEAT_INTERVAL`
(default: `2 * 30s = 60s`). If `last_heartbeat_at` is `None` (worker never started),
the threshold is applied to `started_at` instead. Because the driver pre-claims with
`started_at = now`, a freshly claimed but not-yet-started episode is always within the
threshold, so a concurrent STALE sweep will never incorrectly evict it.

Note: episode timeout (`episode_timeout` in `run_with_ray`, default 3600 s) handles
hung episodes. The heartbeat mechanism handles OOM / external kills that bypass the
timeout path.

---

### Retry logic

An episode is **eligible for retry** if `retry_count < max_retries` (default: **3**) and any of:
1. `status.json` is missing (worker died before the driver could pre-claim, or before the worker could write `RUNNING`)
2. `status == STALE`
3. `status == FAILED`
4. `status == CANCELLED`

An episode is **skipped** (treated as currently executing) if:
- `status == RUNNING` with `(last_heartbeat_at or started_at)` within the staleness threshold

An episode is **done** (never retry) if:
- `status == COMPLETED`

`COMPLETED` with `reward == 0` is a legitimate agent failure, not a technical failure.
Retrying it wastes quota.

---

### `Experiment.get_episodes_to_run()` — new flag semantics

| `resume` | `retry_failed` | Episodes returned |
|---|---|---|
| False | False | All episodes from scratch |
| True | False | Episodes with missing `status.json` (never started) |
| False | True | Episodes with `status IN (FAILED, STALE, CANCELLED)` **or** missing `status.json`, with `retry_count < max_retries` |
| True | True | Union: all missing **+** all retriable failures |

A new field on `Experiment`:
```python
max_retries: int = 3
```

Replaces the trajectory-scan path:
- `_is_trajectory_successful` → **deleted**
- `_load_successful_trajectory_ids` → **deleted**
- `_load_started_trajectory_ids` → **replaced** by `_read_episode_status_map`
- `_find_episodes_to_relaunch` → **absorbed** into `get_episodes_to_run`

---

### Automatic retry loop in `exp_runner`

Both `run_with_ray` and `run_sequentially` gain a `max_retry_rounds` parameter
(default: **3**). After each round, the runner checks for retriable episodes. If any
exist and `rounds_done < max_retry_rounds`, it runs them with `retry_failed=True`.
The loop stops when no retriable episodes remain or the round budget is exhausted.

```python
def run_with_ray(
    exp: Experiment,
    ...,
    max_retry_rounds: int = 3,
) -> ExpResult:
```

The final `ExpResult` aggregates trajectories and failures across all rounds.

Existing recipes that call `run_with_ray(exp)` get automatic retry out of the box
(`max_retry_rounds=3`). Pass `max_retry_rounds=0` to opt out.

---

### Storage changes

Two new methods on `Storage` Protocol and `FileStorage`:

```python
def write_episode_status(self, trajectory_id: str, status: EpisodeStatus) -> None
    # Atomic: write to .tmp then os.replace()

def read_episode_status(self, trajectory_id: str) -> EpisodeStatus | None
    # Returns None if status.json does not exist
```

`EpisodeStatus` is a plain Python `@dataclass` (not Pydantic) defined in a new
`cube_harness/episode_status.py` module. Both `episode.py` and `storage.py` import
from it; this avoids the circular-import problem that would arise if it lived in
`episode.py`.

---

### Error logging

The full stack trace is always written to the episode's log file (existing behaviour
via `redirect_output_to_log`). `error_type` and `error_message` in `status.json`
provide the first line of diagnosis without opening logs. No other changes to logging.

---

## Alternatives considered

**Ray dashboard query.** Rejected: couples status to Ray infrastructure, broken for
sequential runs.

**PID file.** Rejected: PIDs are recycled; unreliable across multi-node clusters.

**Trajectory existence as sentinel.** Current approach. Rejected: requires full
deserialization; silently drops crashed episodes.

**Experiment-level lock file.** Would prevent concurrent runners at the experiment
level but gives no per-episode visibility. Replaced by the pre-claim mechanism, which
is more surgical: episodes claimed by an active run are individually locked.

## Scope

Touches: `episode.py`, `episode_status.py` (new), `experiment.py`, `exp_runner.py`,
`storage.py` — and their specs.

Does **not** change: trajectory format, `Trajectory` model, existing step files,
XRay viewer.

## Open questions (resolved)

| # | Question | Decision |
|---|---|---|
| 1 | `max_retries` default? | **3** |
| 2 | `max_retry_rounds` default? | **3** |
| 3 | Heartbeat interval / staleness threshold? | **30 s / 60 s** |
| 4 | Should `CANCELLED` be retried? | **Yes** — treated same as `FAILED` |
| 5 | Should missing `status.json` be retried by `retry_failed=True`? | **Yes** |
