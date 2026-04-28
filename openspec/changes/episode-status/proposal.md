# RFC: Episode Status File

**Version:** 0.3 — heartbeat & timeout consolidation, 2026-04-27

## Problem

Resume / retry logic in `experiment.py` decides what to re-run by deserialising every
trajectory and scanning for `StepError`. This has four failure modes:

1. **Silent crash.** A Ray worker killed (OOM, `ray.cancel(force=True)`, machine
   failure) writes no terminal status. The trajectory is partial or missing. The
   current code catches the load failure silently and drops the episode — neither
   retried nor counted as done.

2. **False positive on recovered errors.** A `StepError` written mid-episode (an env
   error the harness caught and continued past) flips `_is_trajectory_successful` to
   `False`, triggering a needless retry of an episode that completed normally.

3. **Status requires reading the payload.** Checking whether N episodes are done
   means deserialising N trajectories. At benchmark scale this is slow, and a single
   corrupt step file breaks the check for that episode.

4. **Concurrent runner collision.** Two `run_with_ray` invocations on the same
   `output_dir` see the same unstarted episodes and both submit them to Ray. Nothing
   in the system detects or prevents this.

## Proposed solution

A small `status.json` per episode directory drives all retry decisions. The
trajectory is data; `status.json` is control.

The mechanism has three pieces:

1. **`status.json` per episode**, with a step-boundary heartbeat written from the
   worker's main thread.
2. **Driver poll loop** reads `status.json` from the filesystem instead of querying
   the Ray dashboard for elapsed time. Step-timeout is the only kill trigger.
3. **STALE sweep** cleans up orphaned `RUNNING` entries from crashed drivers.

> **Load-bearing assumption**: `output_dir` is on a filesystem visible to both the
> driver and every Ray worker (NFS, shared object store, single host). This is
> already required for trajectory writes; this RFC just relies on it more pervasively.

---

### Episode statuses

| Status | Written by | Meaning |
|---|---|---|
| `RUNNING` | driver (pre-claim) → worker | Episode queued or actively executing |
| `COMPLETED` | worker (`finally`) | Loop reached natural end (done or `max_steps`) |
| `FAILED` | worker (`finally`) | Unhandled exception propagated out of `_run_loop` |
| `CANCELLED` | driver after `ray.cancel(force=True)` | Step heartbeat went stale → driver killed it |
| `STALE` | STALE sweep (driver) | Worker died without writing terminal status |

**Retry-eligibility ground truth:**
- `RUNNING` with fresh heartbeat → **skip** (alive or queued).
- `RUNNING` with stale heartbeat → **retry** (worker is dead, will be swept to `STALE`).
- `COMPLETED` → **never retry** (`reward == 0` is a legitimate agent failure, not a
  technical one).
- `FAILED` / `STALE` / `CANCELLED` / missing → **retry**, gated by `retry_count <
  max_retries`.

---

### Step-boundary heartbeat

Inside `Episode._run_loop`, the worker writes `last_heartbeat_at` and `current_step`
to `status.json` at:

1. The very top of `_run_loop` (covers stuck `setup_fn` — env reset, container boot).
2. The start of each turn, before `agent.step()` and before `step_fn()`.

That is the entire mechanism. **No background thread. No child process. No asyncio
interaction.** Just a file write in the worker's main thread at natural sync points.

**Why this works where threads and subprocesses didn't:**
- Ray workers run inside an asyncio event loop. A daemon thread doing I/O while the
  main thread is in an async call deadlocks or gets silently dropped on shutdown.
- Ray workers are daemon processes; Python disallows non-daemon children of daemons.

The step-boundary write avoids both because it doesn't run *concurrently* with the
episode — it runs *between* steps, on the same thread that's executing the episode.
It's not "obvious in hindsight"; it's a different topology.

---

### Pre-claim: race protection on submission

Before submitting episodes to Ray, the driver writes `RUNNING` for every episode it
intends to launch:

```python
storage.write_episode_status(traj_id, EpisodeStatus(
    status="RUNNING",
    started_at=now,
    last_heartbeat_at=None,    # worker hasn't started; queued in Ray
    retry_count=incremented_from_previous,
    ...,
))
```

A concurrent runner that opens the same `output_dir` then sees `RUNNING` and skips:
- `resume=True` skips because it only returns missing-`status.json`.
- `retry_failed=True` skips because `RUNNING` ∉ `{FAILED, STALE, CANCELLED}` and the
  heartbeat-staleness check screens out the rest.

When the worker actually picks up the episode, `_run_loop` overwrites the pre-claim
with `started_at = now()` and starts writing heartbeats.

> **Limitation, accepted for v1.** Two drivers starting within the same ~second can
> both call `get_episodes_to_run()` before either pre-claims, then race-write
> overlapping pre-claims. The harness can't fully prevent this without an
> experiment-level lock. Document and revisit if it becomes a real problem.

---

### `last_heartbeat_at = None` is two semantics in one absent field

The field is `None` between pre-claim (driver) and the first heartbeat (worker), i.e.
while the episode is **queued in Ray** but not yet picked up. It's also `None` if the
worker died after pre-claim but before reaching `_run_loop` (rare).

Two distinct policies follow:

| Caller | Behaviour for `last_heartbeat_at = None` |
|---|---|
| Driver poll (mid-run) | **Skip** — assume it's queued, don't kill |
| STALE sweep (start of next run, or after `ray.shutdown`) | Mark `STALE` if `now − started_at > orphan_threshold_s` (default 1 h — generous Ray-queue allowance) |

This pins down a behaviour that would otherwise live in folklore.

---

### Driver kill via filesystem read, not Ray dashboard

Replace `_get_running_elapsed_s` (which calls `ray.util.state.api.list_tasks` on the
Ray dashboard) with a filesystem read:

```python
for ref in episodes_in_progress:
    traj_id = ref_to_traj_id[ref]
    status = storage.read_episode_status(traj_id)
    if status is None or status.last_heartbeat_at is None:
        continue   # not started yet — let Ray handle queueing

    age = now - status.last_heartbeat_at
    if age > step_timeout_s + cancel_grace_s:
        ray.cancel(ref, force=True)
        status.status = "CANCELLED"
        status.ended_at = now
        status.error_type = "StepTimeout"
        status.error_message = f"Step {status.current_step} exceeded {step_timeout_s:.0f}s"
        storage.write_episode_status(traj_id, status)
        episodes_in_progress.remove(ref)
```

Wins:
- **No Ray-dashboard dependency.** `from ray.util.state.api import list_tasks`
  deleted. Past us has been bitten by this API drifting between Ray versions.
- **Per-step granularity.** Detects "stuck on step 14 for 30 min" instead of waiting
  for total elapsed to exceed an episode-wide cap.
- **Always writes a terminal status.** No more zombie `RUNNING` after a force kill.

`episode_timeout` is **removed**. Total-time is bounded by `max_steps × step_timeout_s`
(with `max_steps=50`, `step_timeout=30 min` → 25 h ceiling — adequate; nothing in the
repo runs near it).

---

### `status.json` schema

```json
{
  "status": "FAILED",
  "task_id": "workarena.create-incident",
  "episode_id": 3,
  "started_at": 1745000000.0,
  "ended_at": 1745000042.0,
  "last_heartbeat_at": 1745000040.0,
  "current_step": 14,
  "reward": null,
  "had_step_errors": true,
  "error_type": "ConnectionError",
  "error_message": "Container failed to bind on port 8080",
  "retry_count": 1
}
```

| Field | Type | Set by | Notes |
|---|---|---|---|
| `status` | enum | both | Lifecycle state |
| `task_id` | str | both | For grouping / listing |
| `episode_id` | int | both | For grouping / listing |
| `started_at` | float | driver (pre-claim), then worker (actual start) | Wall clock |
| `ended_at` | float\|None | worker / driver | Wall clock at terminal status |
| `last_heartbeat_at` | float\|None | worker | `None` until first turn begins |
| `current_step` | int | worker | `0` during `setup_fn`; `1..` once turn loop starts |
| `reward` | float\|None | worker | `None` until `COMPLETED` |
| `had_step_errors` | bool | worker | Informational; does not affect retry |
| `error_type` | str\|None | worker / driver | Exception class for failures |
| `error_message` | str\|None | worker / driver | Short message — first-line diagnosis |
| `retry_count` | int | driver | See semantics below |

### `retry_count` semantics

- `retry_count = 0` → original attempt.
- `retry_count = N` → the N-th retry.
- Episode is retried iff `retry_count < max_retries`.
- `max_retries = 3` ⇒ attempts at `retry_count` 0, 1, 2, 3 ⇒ **4 total attempts**.

The driver computes the new value during pre-claim: it reads the existing
`status.json` (if any), increments `retry_count` by one if a previous attempt exists,
then writes `RUNNING`.

---

### `Experiment.get_episodes_to_run()` semantics

A new field:

```python
class Experiment(TypedBaseModel):
    ...
    max_retries: int = 3
```

| `resume` | `retry_failed` | Episodes returned |
|---|---|---|
| F | F | All episodes from scratch |
| T | F | Missing `status.json` (never started) |
| F | T | `status IN (FAILED, STALE, CANCELLED)` **or** missing, with `retry_count < max_retries` |
| T | T | Union of the above two |

`RUNNING` (with fresh heartbeat) is **never** included — by either flag.

The deletion list in `experiment.py`:
- `_is_trajectory_successful` — superseded by COMPLETED status.
- `_load_successful_trajectory_ids` — full trajectory scans no longer needed.
- `_load_started_trajectory_ids` — replaced by `_read_episode_status_map`.
- `_find_episodes_to_relaunch` — folded into `get_episodes_to_run`.

---

### STALE sweep

Marks `RUNNING` → `STALE` for episodes that meet:

- `last_heartbeat_at` set, and `now − last_heartbeat_at > step_timeout_s + cancel_grace_s`, **or**
- `last_heartbeat_at == None`, and `now − started_at > orphan_threshold_s` (default 1 h).

Run at:
- The **start** of `get_episodes_to_run()` whenever `resume=True` or `retry_failed=True`
  — cleans up after a previous driver that crashed without `ray.shutdown()`.
- **After** `ray.shutdown()` in `_run_with_ray_impl` — handles the normal end-of-run
  case where workers are killed when the cluster shuts down.

---

### Auto-retry loop

`run_with_ray` and `run_sequentially` gain `max_retry_rounds: int = 3`. After each
round, the runner queries for retriable episodes; if any exist and we haven't hit
the round budget, it re-runs them with `retry_failed=True` on the same `output_dir`.

```python
def run_with_ray(
    exp: Experiment,
    *,
    n_cpus: int = 4,
    ray_poll_timeout: float = 2.0,
    step_timeout_s: float = 1800.0,        # 30 min — kill if a single step hangs
    cancel_grace_s: float = 120.0,         # buffer over step_timeout
    orphan_threshold_s: float = 3600.0,    # 1 h — for never-started pre-claims
    max_retry_rounds: int = 3,             # post-run retry sweeps
    ...,
) -> ExpResult:
```

Existing recipes that call `run_with_ray(exp)` get auto-retry out of the box. Pass
`max_retry_rounds=0` to opt out.

The final `ExpResult` aggregates trajectories and failures across all rounds.

---

### Sequential mode

`run_sequentially` shares `max_retry_rounds`. It does **not** enforce
`step_timeout_s` (no driver poll loop, no external killer for the in-process
worker). Heartbeats are still written for status visibility; pre-claim is skipped
(no concurrency to defend against). A hung step in sequential mode requires a
human Ctrl-C — acceptable, since sequential is the debug path. A future RFC can
add a `signal.alarm`-based per-step timeout if it becomes needed.

---

### Storage Protocol additions

```python
class Storage(Protocol):
    ...
    def write_episode_status(self, trajectory_id: str, status: EpisodeStatus) -> None: ...
    def read_episode_status(self, trajectory_id: str) -> EpisodeStatus | None: ...
```

`FileStorage` implements them with atomic write (`.tmp` sibling + `os.replace()`).
Status lives at `episodes/{trajectory_id}/status.json`.

---

### `EpisodeStatus` lives in a new module

A plain `@dataclass` in `cube_harness/episode_status.py`. Imported by both
`episode.py` and `storage.py`. Avoids the circular import that would arise from
defining it in `episode.py`.

---

### Error logging

Stack traces continue to land in the per-episode log file via the existing
`redirect_output_to_log` plumbing. `error_type` and `error_message` in
`status.json` are the **first-line** diagnosis — surfaced in summaries / dashboards
without forcing a log open.

For driver-written `CANCELLED`, the driver populates `error_type = "StepTimeout"`
and a message like `"Step 14 exceeded 1800s"`.

---

## Alternatives considered

**Background-thread heartbeat.** Rejected: Ray-worker / asyncio interactions made
this unstable in the past — see the "Why this works" section.

**Child-process heartbeat.** Rejected: Ray workers are daemon processes; Python
forbids non-daemon children of daemons.

**Ray-dashboard query (current behaviour).** Rejected: requires the dashboard to be
reachable (sometimes it isn't), couples retry to Ray-state-API versioning.

**PID file.** Rejected: PIDs recycle; unreliable on multi-node clusters.

**Trajectory existence as sentinel (current behaviour).** Rejected: requires full
deserialisation; silently drops crashed episodes.

**Experiment-level lock file.** Discussed for v1, deferred. Pre-claim narrows the
race window; a lock would close it. Add when needed.

---

## Scope

**Touches:** `episode.py`, `episode_status.py` (new), `experiment.py`, `exp_runner.py`,
`storage.py`, plus their specs and tests.

**Does not change:** trajectory format, `Trajectory` model, existing step files, XRay
viewer, sequential-mode debug experience.

---

## Resolved questions

| # | Question | Decision |
|---|---|---|
| 1 | `Experiment.max_retries` default? | **3** (4 total attempts) |
| 2 | `max_retry_rounds` default? | **3** |
| 3 | Heartbeat mechanism? | Step-boundary write from worker's main thread |
| 4 | `step_timeout_s` default? | **1800s (30 min)** |
| 5 | `cancel_grace_s` default? | **120s (2 min)** |
| 6 | `orphan_threshold_s` default? | **3600s (1 h)** |
| 7 | Drop `episode_timeout`? | **Yes** |
| 8 | Drop `list_tasks` (Ray dashboard) dependency? | **Yes** |
| 9 | `CANCELLED` retried? | **Yes**, treated like `FAILED` |
| 10 | Missing `status.json` retried by `retry_failed=True`? | **Yes** |
| 11 | Sequential-mode step-timeout enforcement? | **Out of scope for v1** |
| 12 | Concurrent-driver hard lock? | **Out of scope for v1** (pre-claim narrows; document the residual race) |
