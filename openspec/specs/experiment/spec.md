# Experiment

**Module:** `cube_harness.experiment`, `cube_harness.exp_runner`

## Purpose

An `Experiment` pairs an `AgentConfig` with a `Benchmark` and produces one `Episode`
per task. The runners in `exp_runner.py` execute those episodes sequentially or in
parallel via Ray, with OpenTelemetry benchmark-level spans wrapping the run.

## Public API

### `Experiment` (serializable)
```python
class Experiment(TypedBaseModel):
    name: str
    output_dir: Path
    agent_config: AgentConfig
    benchmark: Benchmark             # cube.benchmark.Benchmark
    resume: bool = False
    retry_failed: bool = False
    max_steps: int = MAX_STEPS

    @property
    def config(self) -> dict          # model_dump(serialize_as_any=True)

    def get_episodes_to_run(self) -> list[Episode]
    def save_config(self) -> None     # writes experiment_config.json
    @classmethod
    def load_config(cls, path: str) -> Experiment

    def print_stats(self, results: ExpResult) -> None
```

### Resume / retry semantics
| `resume` | `retry_failed` | Episodes returned |
|----------|----------------|-------------------|
| False    | False          | All episodes created from scratch |
| True     | False          | Unstarted episodes only (configs exist, no trajectory) |
| False    | True           | Failed episodes only (trajectory exists but not successful); `allow_overwrite=True` |
| True     | True           | Unstarted ∪ failed |

Successful = `trajectory.last_env_step().done and no step.output.error`.

### `ExpResult`
```python
class ExpResult(TypedBaseModel):
    exp_id: str
    tasks_num: int
    config: dict = {}
    trajectories: dict[str, Trajectory] = {}    # task_id → Trajectory
    failures: dict[str, str] = {}               # task_id → error message
```

### Runners

#### `run_sequentially(exp, debug_limit=None, otlp_endpoint=None, model=None, agent_name=None) -> ExpResult`
Runs episodes one at a time. `debug_limit` caps the list (useful for `make debug`).

#### `run_with_ray(exp, n_cpus=4, ray_poll_timeout=2.0, episode_timeout=3600.0, otlp_endpoint=None, model=None, agent_name=None) -> ExpResult`
Parallel via Ray `@remote`. Initializes Ray with dashboard enabled. `episode_timeout`
(seconds) cancels stuck episodes — graceful for `_CANCEL_GRACE_PERIOD_S=60`, force
after.

Both runners:
1. Enter `tracer.benchmark(exp.name)` span
2. `exp.save_config()`
3. `exp.benchmark.setup()`
4. Call the internal `_run_*_impl`
5. `finally: exp.benchmark.close()` + `tracer.shutdown()`

Per-episode stdout/stderr is redirected to `<output_dir>/episodes/<traj_id>/logs/`
via `redirect_output_to_log` from `episode_logs.py`.

## Invariants

1. `Experiment` is itself a `TypedBaseModel` — JSON-serializable. Holds a live
   `Benchmark` though, which is not guaranteed picklable; don't ship `Experiment`
   across processes except via `save_config()`/`load_config()`.
2. `save_config()` is called before every run. `experiment_config.json` is
   authoritative for resume/retry workflows.
3. `benchmark.setup()` and `benchmark.close()` wrap every run. Resource cleanup is
   guaranteed on exceptions.
4. Ray disrupts signal handling (Ctrl+C) — known limitation, no fix yet.

## Contracts for implementers

- New runners should preserve the setup/close/tracer wrapping so benchmarks that
  create L2 resources don't leak.
- Telemetry metadata (model, agent_name) flows from runner args into the tracer
  span attributes — populate these when writing custom runners.
- Resume requires `episode_config.json` to exist — writes always go through
  `save_episode_config()` before the episode runs (see episode spec).

## Gotchas

- `run_with_ray` sets `dashboard_host="0.0.0.0"` — the Ray dashboard is exposed on
  all interfaces when running locally. Fine on a workstation; consider in multi-user
  environments.
- Ray workers inherit `env_vars` from `get_trace_env_vars()` — if you need extra env
  vars in workers, they must be added explicitly (not automatic from the driver).
- Episode timeouts use the Ray dashboard API (`list_tasks`). If the dashboard is
  unreachable, timeouts are silently skipped that cycle — logs a debug message.
- Cancellation is best-effort: graceful cancel waits `_CANCEL_GRACE_PERIOD_S` seconds,
  then force-kills. An episode that hangs in a C extension may not respect cancel.
