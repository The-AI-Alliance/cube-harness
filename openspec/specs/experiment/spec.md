# Experiment

**Module:** `cube_harness.experiment`, `cube_harness.exp_runner`

## Purpose

An `Experiment` pairs an `AgentConfig` with a `BenchmarkConfig` and produces one
`Episode` per task. The runners in `exp_runner.py` instantiate the live
`Benchmark` via `benchmark_config.make(infra)` and execute episodes sequentially
or in parallel via Ray, with OpenTelemetry benchmark-level spans wrapping the run.

## Public API

### `Experiment` (serializable)
```python
class Experiment(TypedBaseModel):
    name: str
    output_dir: Path
    agent_config: AgentConfig
    benchmark_config: BenchmarkConfig    # cube.benchmark.BenchmarkConfig
    infra: InfraConfig | None = None     # forwarded to benchmark_config.make(infra)
    resume: bool = False
    max_steps: int = MAX_STEPS
    max_retries: int = 3                 # per-episode retry cap

    @property
    def config(self) -> dict             # model_dump(serialize_as_any=True)

    def get_episodes_to_run(
        self,
        benchmark: Benchmark | None = None,
        *,
        step_timeout_s: float = 1800.0,
        cancel_grace_s: float = 120.0,
        orphan_threshold_s: float = 3600.0,
    ) -> list[Episode]
    def save_config(self) -> None        # writes experiment_config.json
    @classmethod
    def load_config(cls, path: str) -> Experiment

    def print_stats(self, results: ExpResult) -> None
```

`benchmark_config` is the serialisable side; `Benchmark` is no longer Pydantic
in cube-standard, so it cannot live as a field on `Experiment`. The runners
build the live `Benchmark` for the duration of a run via
`with exp.benchmark_config.make(exp.infra) as benchmark:` and pass it to
`get_episodes_to_run` so episodes pick up its `_runtime_context` and
`config.container_backend`. Tests that only enumerate episodes without running
may omit `benchmark`; the resulting episodes carry no `runtime_context` and no
`container_backend`.

### Resume / retry semantics

| `resume` | Episodes returned                                                                                                                         |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `False`  | All episodes from scratch                                                                                                                 |
| `True`   | Episodes with no `status.json` (never started), plus retriable statuses (`FAILED`, `CANCELLED`, `STALE`) with `retry_count < max_retries` |

`RUNNING` / `QUEUED` (in-flight) are never returned. `COMPLETED` and
`MAX_STEPS_REACHED` (terminal, non-retriable) are always skipped.

When `resume=True`, `sweep_stale_statuses` runs first so orphaned `RUNNING`/`QUEUED`
entries become `STALE` and are eligible for retry.

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
3. `with exp.benchmark_config.make(exp.infra) as benchmark:` — `make` provisions
   declared resources, instantiates the runtime pair, and calls `setup()`. The
   context manager guarantees `close()` on exit.
4. Call the internal `_run_*_impl` with the live `benchmark`
5. `tracer.shutdown()`

Per-episode stdout/stderr is redirected to `<output_dir>/episodes/<traj_id>/logs/`
via `redirect_output_to_log` from `episode_logs.py`.

## Invariants

1. `Experiment` is itself a `TypedBaseModel` — JSON-serializable. It holds a
   `BenchmarkConfig` (Pydantic, picklable), not a live `Benchmark`. Live
   benchmarks are constructed by the runner via `benchmark_config.make(infra)`
   inside a context manager.
2. `save_config()` is called before every run. `experiment_config.json` is
   authoritative for resume/retry workflows.
3. The runner's `with benchmark_config.make(infra) as benchmark:` block wraps
   every run. `make()` calls `setup()` internally, and the context manager calls
   `close()` on exit — resource cleanup is guaranteed on exceptions.
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
