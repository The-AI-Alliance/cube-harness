# RFC: BenchmarkConfig, CompositeBenchmark, and BenchmarkPool

**Status**: Proposal — no code yet  
**Replaces**: `docs/BENCHMARK_POOL_DESIGN.md` (PR #283)

---

## Problem

Three separate but related problems that share a root cause:

1. **`subset_from_list` is fragile.** It uses `copy.deepcopy(self)`, which crashes when a benchmark holds OS-level private attrs (subprocess handles, file descriptors, thread locks) created during `setup()`. Workarounds (`__deepcopy__` overrides) have already appeared in the codebase.

2. **Benchmarks can't be serialized or composed.** There is no way to save a configured benchmark suite to JSON and reload it. A suite combining WorkArena + MiniWob + OSWorld can't be persisted without custom code.

3. **Scaling to multiple server instances requires ugly workarounds.** WorkArena and WebArena degrade under load. To distribute tasks across N servers today you have to manually manage N benchmark objects and write your own dispatch logic.

All three problems share the same root cause: **`Benchmark` mixes config (serializable data) with runtime state (OS handles, connections)**.

---

## Proposed Solution

Split `Benchmark` into two layers, mirroring the existing `TaskConfig` / `Task` pattern:

```
BenchmarkConfig   →  make(infra_config)  →  Benchmark
  (pure data)                               (runtime state)
  serializable                              not serializable
  subset_from_list()                        _setup() / close()
  composable                                get_task_configs()
```

`CompositeBenchmark` and `BenchmarkPool` then become natural compositions of `BenchmarkConfig`.

---

## Part 1 — BenchmarkConfig

`BenchmarkConfig` is the serializable description of a benchmark: what tasks exist, how they're parameterized, what resources they need. It holds no runtime state.

`subset_from_list` becomes trivially safe — `BenchmarkConfig` is a pure Pydantic model with no subprocess handles, so `model_copy` just works.

`make(infra_config)` is the single point where a config becomes a live `Benchmark`. It provisions required resources idempotently, then calls `_setup()`. A `Benchmark` is always born ready — there is no state where it exists but hasn't been initialized. This eliminates the current footgun of constructing a benchmark and forgetting to call `.setup()`.

**`task_metadata` must be available at config construction time**, without calling `make()`. For most benchmarks this is already true (static JSON loaded at import time). For benchmarks with dynamic task lists (WorkArena), the right pattern is a developer script that runs once and ships `task_metadata.json` with the package — which is exactly what Nic's uniformization PRs (#276, #278, #292) do.

### What serialization enables

```python
# Configure once, save to disk, reload anywhere
config = WorkArenaBenchmarkConfig().named_subset("l1")
config.model_dump_json()  # → fully serializable JSON

# Multi-benchmark suite — also serializable
suite = CompositeBenchmarkConfig([
    WorkArenaBenchmarkConfig().named_subset("l1"),
    MiniWobBenchmarkConfig(),
])
suite.model_dump_json()  # → share with a colleague, store as experiment artifact, replay on CI
```

---

## Part 2 — CompositeBenchmark

### Problem

There is no way to run WorkArena + MiniWob + OSWorld in a single experiment today. Each benchmark requires its own `Experiment` and there is no unified result.

### Design

`CompositeBenchmarkConfig` holds a list of `BenchmarkConfig` instances (which can themselves be `BenchmarkPool` configs). Because each element is a `BenchmarkConfig`, the whole composite is serializable.

`make()` instantiates each sub-benchmark and returns a `CompositeBenchmark` that routes each task to its source benchmark's runtime context.

### Usage

```python
suite = CompositeBenchmarkConfig([
    BenchmarkPoolConfig(config=WorkArenaBenchmarkConfig().named_subset("l1"), n_servers=3),
    MiniWobBenchmarkConfig(),
    OSWorldBenchmarkConfig(),
])

benchmark = suite.make(azure_infra)
exp = Experiment(name="multi_bench", benchmark=benchmark, ...)
run_with_ray(exp, n_cpus=32)
```

---

## Part 3 — BenchmarkPool

### Problem

WorkArena (ServiceNow) and WebArena support limited concurrency per server (~7 agents). At RL scale you need N server instances. Today there is no clean way to distribute tasks across them.

### Design

`BenchmarkPool` is a concrete class that takes a `BenchmarkConfig` and `n_servers`. It calls `config.make(infra)` N times, provisioning N independent L2 resources (e.g. one ServiceNow instance per slot). `BenchmarkConfig.make()` is what makes this clean — calling it N times naturally yields N independent servers.

```python
pool = BenchmarkPool(
    config=WorkArenaBenchmarkConfig().named_subset("l1"),
    n_servers=3,
)
exp = Experiment(name="workarena-l1", benchmark=pool.make(azure_infra), ...)
run_with_ray(exp, n_cpus=21)    # 3 servers × 7 agents each
```

`BenchmarkPool` is not a `BenchmarkConfig` subclass — it is a thin execution wrapper. The config stays infra-agnostic; `n_servers` is an execution decision made at construction time.

### Load balancing via Ray Actor

**Server assignment must happen at task execution time**, not at episode creation time. If contexts are baked into episodes upfront, all workers end up on the same server.

A Ray Actor acts as a cross-process semaphore. The main process dispatches all N tasks to Ray immediately (non-blocking — Ray's scheduler is unaffected). Each worker blocks on `actor.acquire()` just before running its task, receives the `RuntimeContext` dict for a free server slot, runs the task, then calls `actor.release()`.

```
Main process:  dispatch all N tasks at once (non-blocking)
                        ↓
Ray workers:   actor.acquire() → RuntimeContext dict → run task → actor.release()
```

`Benchmark` instances stay in the main process and are never serialized to workers. Only `RuntimeContext` dicts (plain dicts) cross the process boundary — which is all `task_config.make(runtime_context=ctx)` needs. The actor handle is serializable by Ray design, so workers can reach it from any process.

---

## What Changes Where

| Layer | Change |
|---|---|
| cube-standard | New `BenchmarkConfig` base class; `subset_from_list` moves there; `Benchmark` loses it |
| cube-harness | `BenchmarkPool`, `CompositeBenchmark`; `run_with_ray` passes actor handle to workers |
| Per-cube | Rename `XxxBenchmark` → `XxxBenchmarkConfig`, add `make()`, slim runtime class |

Per-cube migration is mechanical for all benchmarks whose `task_metadata` is already a shipped JSON (WorkArena after #292, MiniWob, SWE-bench after #276, Terminal-Bench after #278). Nic's uniformization PRs are the direct prerequisite.

---

## Open Questions

1. **`BenchmarkConfig` in cube-standard or cube-harness?**  
   It belongs in cube-standard (same layer as `TaskConfig`). `BenchmarkPool` and `CompositeBenchmark` are harness concerns.

2. **Backwards compatibility for `Benchmark.setup()`**  
   `setup()` becomes an implementation detail called automatically by `make()`. Recipes that call `benchmark.setup()` explicitly need to migrate. Transition: deprecate for 1–2 releases before removal.

3. **`InfraConfig` in `make()` vs stored on the config**  
   Passing at `make()` is cleaner — the config describes *what*, the infra describes *where*.

4. **How does `run_with_ray` detect a pool?**  
   Two options: (a) check `isinstance(benchmark, BenchmarkPool)` and pass the actor handle explicitly to `run_episode`, or (b) a hook on `Benchmark` that lets the pool wrap the Ray function transparently. Option (b) is cleaner but adds abstraction.

5. **Per-sub-benchmark `n_servers` in `CompositeBenchmark`**  
   Out of scope for now. Likely resolved by a future `InfraConfig` design where infra describes available resources per benchmark type.
