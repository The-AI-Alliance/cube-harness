# RFC: Benchmark Scaling — Three-Part Proposal

**Status**: Analysis only — not yet an RFC  
**Date**: 2026-04-14  

This document covers three related but independent improvements to the benchmark layer:

- **Part 1 — BenchmarkConfig / make(infra)**: Split `Benchmark` into pure-data config and runtime object (mirrors the existing `TaskConfig` / `Task` pattern).
- **Part 2 — CompositeBenchmark**: Combine multiple heterogeneous benchmarks into a single logical benchmark for multi-benchmark experiments.
- **Part 3 — BenchmarkPool**: Replicate a single benchmark across K equivalent server instances for parallel task execution.

---

# Part 1 — BenchmarkConfig / make(infra)

**Context**: Proposed split of `Benchmark` into pure-data `BenchmarkConfig` and runtime `Benchmark`, aligned with the existing `TaskConfig` / `Task` pattern.

---

## Proposed Pattern

```python
# Current
benchmark = WorkArenaBenchmark(level="l1", default_tool_config=tc)
subset = benchmark.subset_from_list(["task-a"])   # needs deepcopy workarounds
benchmark.setup()                                  # provisions infra + builds task_metadata

# Proposed
config = WorkArenaBenchmarkConfig(level="l1", default_tool_config=tc)
subset = config.subset_from_list(["task-a"])       # trivial dict filter, no deepcopy
benchmark = subset.make(infra_config)              # provision (idempotent) + setup
```

**Key contracts:**
- `BenchmarkConfig` — pure data, serializable, holds `task_metadata` at construction time, no runtime state
- `BenchmarkConfig.subset_from_list()` — trivial `dict` filter, no copying of runtime objects
- `BenchmarkConfig.make(infra_config)` — provisions L1 resources idempotently, calls `_setup()`, returns a live `Benchmark`
- `Benchmark` — holds runtime state only: server processes, connections, `_runtime_context`

This mirrors the existing `TaskConfig.make(runtime_context, container_backend) → Task` pattern.

---

## Section 1: cube-standard Changes

### 1.1 New `BenchmarkConfig` base class

The main addition. Currently `Benchmark` is both config and runtime. We split it:

```python
class BenchmarkConfig(TypedBaseModel, ABC):
    benchmark_metadata: ClassVar[BenchmarkMetadata]
    task_metadata: ClassVar[dict[str, TaskMetadata]]  # loaded at import time via __init_subclass__
    task_config_class: ClassVar[type[TaskConfig]]

    resources: list[ResourceConfig] = Field(default_factory=list)
    default_tool_config: ToolConfig | None = None
    seed_generator: AbstractSeedGenerator | None = None

    def subset_from_list(self, tasks: list[str]) -> "BenchmarkConfig":
        """Trivial filter — no deepcopy, no workarounds, BenchmarkConfig is pure data."""
        new = self.model_copy()
        subset = {tid: tm for tid, tm in self.task_metadata.items() if tid in set(tasks)}
        object.__setattr__(new, "task_metadata", subset)
        return new

    @abstractmethod
    def make(self, infra_config: InfraConfig | None = None) -> "Benchmark":
        """Provision resources idempotently, then return a live Benchmark."""
        ...
```

`subset_from_list` becomes trivially safe: `BenchmarkConfig` has no subprocess handles, no file descriptors, no thread locks — all data fields, all safely copyable.

### 1.2 `Benchmark` base class (runtime only)

`Benchmark` keeps `_setup()`, `close()`, `get_task_configs()`, and `_runtime_context`. It no longer needs `subset_from_list` or `model_copy` tricks. The class becomes leaner.

### 1.3 `InfraConfig` integration in `make()`

`InfraConfig` already exists in `cube-standard/src/cube/resource.py` with `provision()`, `launch()`, `cleanup()`. The `make()` template would look like:

```python
def make(self, infra_config: InfraConfig | None = None) -> "Benchmark":
    if infra_config is not None:
        for resource in self.resources:
            if infra_config.provision_status(resource) == "needs_provisioning":
                infra_config.provision(resource)
    benchmark = self._instantiate()   # subclass creates the runtime Benchmark
    benchmark.setup()
    return benchmark
```

### 1.4 `task_metadata` contract

The contract becomes explicit: **task_metadata must be available at config construction time**, without calling `setup()` or `make()`. For static benchmarks (JSON files), this is already true via `__init_subclass__` auto-loading. For dynamic benchmarks, the metadata must be populated in `__init__` or via `install()`.

`install()` is the agreed escape hatch for the case where a developer needs to run a one-time computation (call an external library, download from HuggingFace) and cache the result as `task_metadata.json`. Once installed, subsequent config instantiations load from JSON — no network call, no library invocation.

### 1.5 Effort: cube-standard

| Change | Effort |
|---|---|
| Define `BenchmarkConfig` base class | Medium — careful API design, backward compat |
| Slim down `Benchmark` (remove subset logic) | Small |
| Update `Experiment` to accept `BenchmarkConfig` | Small |
| Deprecate `Benchmark.setup()` as public entrypoint | Small + docs |

**Total: ~3–5 days** (design-heavy, implementation is small)

---

## Section 2: cube-harness Changes

### 2.1 `Experiment` / `exp_runner`

Currently `Experiment` takes a fully-setup `Benchmark`. Under the new pattern it would take a `BenchmarkConfig` and call `make(infra_config)` internally, or the user calls `make()` before passing it in. The second is more explicit — recipes stay readable.

```python
# Recipe change:
config = WorkArenaBenchmarkConfig(level="l1", default_tool_config=tc)
config.subset_from_list(task_ids)
benchmark = config.make()           # or config.make(AzureInfraConfig(...))

exp = Experiment(benchmark=benchmark, ...)
run_with_ray(exp)
```

Effort: **Small** — `Experiment` just stops calling `benchmark.setup()` (if it ever did — currently it's the recipe's job).

---

## Section 3: General Patterns

### Pattern 1: Static metadata (JSON file shipped with package)

**Examples**: MiniWob (today), SWE-bench Verified (after uniformization-swebench), Terminal-Bench (after uniformization-terminalbench)

```
install()  →  calls external API / HuggingFace once  →  writes task_metadata.json
             (developer runs once, result committed to repo or cached locally)

BenchmarkConfig.__init_subclass__  →  loads task_metadata.json automatically
```

No changes needed at config construction time. `make()` can call `_setup()` (start server, etc.) without touching metadata.

**Effort**: Mechanical — rename, add `make()` method.

### Pattern 2: Seed-generator pattern (Nic's task-metadata PR for WorkArena)

**Examples**: WorkArena (after task-metadata PR)

`task_metadata` holds all task types (loaded from JSON via `install()`). Seeds are not stored in `task_metadata` — a `SeedGenerator` lazily computes them from BrowserGym on first use.

```
install()  →  calls get_all_tasks_agents once  →  writes task_metadata.json

BenchmarkConfig.__init_subclass__  →  loads task_metadata.json
BenchmarkConfig.make()  →  instantiates SeedGenerator  →  sets up benchmark
```

This is the correct pattern: BrowserGym is only imported when `make()` is called (or `install()`), not at config construction. `subset_from_list` works purely on the pre-loaded JSON.

**Effort**: Already done in Nic's task-metadata PR — WorkArena is essentially there.

### Pattern 3: External-provisioned infra (L2 resources)

**Examples**: WebArena-Verified (Docker service), OSWorld (VM)

```python
config = WebArenaBenchmarkConfig(default_tool_config=tc)
benchmark = config.make(AzureInfraConfig())   # provisions Docker service, runs _setup()
```

`make(infra_config)` provisions the L2 resource (calls `infra.launch(DockerServiceConfig(...))`), then hands the `ResourceHandle` to `_setup()` via `_runtime_context`. Task configs receive the live endpoint.

**Effort**: Medium — requires wiring `InfraConfig` through `make()` cleanly. Webarena PR #260 already does this pattern, but manually in `_setup()` rather than in `make()`.

### Pattern 4: Self-hosted server (L2, no external infra)

**Examples**: MiniWob (local HTTP server)

`make(infra_config=None)` — starts the subprocess in `_setup()`, no `InfraConfig` needed. The `infra_config` parameter is optional.

**Effort**: Trivial.

---

## Section 4: Per-Cube Analysis

### 4.1 WorkArena

**Current state**: `_setup()` calls `get_all_tasks_agents()` to build `task_metadata` and `_task_tuples` dynamically.

**After Nic's task-metadata PR**: `task_metadata.json` is shipped, loaded via `__init_subclass__`. `_setup()` only wires up the `SeedGenerator`. Seeds come from `get_all_tasks_agents()` lazily via `WorkArenaSeedGenerator._ensure_loaded()`.

**Remaining gap**: `_setup()` still has the early-return guard (`if "task_metadata" in self.__dict__: return`) which breaks with `model_copy`. This is fixed by the separate `fix/webagent-papercuts` commit.

**With the proposed pattern**: `WorkArenaBenchmarkConfig.make()` wires `SeedGenerator` and calls `_setup()`. `subset_from_list` becomes a pure JSON filter.

**Effort**: **Trivial** once task-metadata PR merges — rename + add `make()`.

---

### 4.2 MiniWob

**Current state**: `task_metadata` loaded from `miniwob_tasks.json` statically. `_setup()` starts an HTTP server (subprocess).

**Remaining issue**: `_server_process` (PrivateAttr) breaks `model_copy` in `subset_from_list` → fixed by cube-standard#101 (`model_copy` skips PrivateAttrs) + removal of `__deepcopy__` workaround.

**With the proposed pattern**: `MiniWobBenchmarkConfig.make()` calls `_setup()` which starts the server. Config is pure JSON + port number. `subset_from_list` is trivial.

**Effort**: **Trivial** — already clean, just rename + add `make()`.

---

### 4.3 SWE-bench Verified

**Current state (before uniformization-swebench)**: `_setup()` downloads HuggingFace dataset and builds `task_metadata`.

**After uniformization-swebench PR**: `task_metadata.json` is shipped. `install()` populates per-task execution cache from HuggingFace. `_setup()` is a no-op comment.

**With the proposed pattern**: Already aligned. `SWEBenchVerifiedBenchmarkConfig.make()` is trivial — no shared infra to provision.

**Effort**: **Trivial** after uniformization PR merges.

---

### 4.4 Terminal-Bench

**Current state (before uniformization-terminalbench)**: Same pattern as SWE-bench — HuggingFace load in `_setup()`.

**After uniformization-terminalbench PR**: Same solution as SWE-bench — `task_metadata.json` shipped, `install()` for heavy data.

**Effort**: **Trivial** after uniformization PR merges.

---

### 4.5 WebArena-Verified

**Current state**: `task_metadata` loaded lazily in `model_post_init()` via `WebArenaVerified().get_tasks()`. `_setup()` pings URLs to verify services are reachable.

**PR #260 (fix/webarena-v3)**: Wires `InfraConfig` auto-provisioning — `_setup()` calls `infra.launch(DockerServiceConfig(...))` and maps endpoints to `wav_config.environments`.

**Remaining gap for proposed pattern**: `task_metadata` must be in a JSON file (or computed at `__init__` from a static source). Currently it requires calling the WebArena library. Needs an `install()` that writes `task_metadata.json`.

**Effort**: **Medium** — needs `install()` to generate `task_metadata.json` from WebArena library. Then same rename + `make()` pattern.

---

### 4.6 OSWorld

**Current state**: `task_metadata` loaded from OSWorld repo JSON files in `_setup()`. `_setup()` calls `self.install()` (clone repo, provision VM image). `infra: InfraConfig` already a field on the benchmark — task configs already receive it.

**With the proposed pattern**: Almost there. Move task metadata loading to `install()` + `__init_subclass__` auto-load. `make(infra_config)` provisions the VM image (L1) and returns the benchmark.

**Effort**: **Small** — OSWorld is already the closest to the proposed pattern.

---

### 4.7 SWE-bench Live

**Current state**: Similar to SWE-bench Verified but hits live GitHub issues. Task list changes over time.

**Consideration**: `task_metadata` cannot be fully static here (issues open/close). Two options:
1. `install()` snapshots the current task list to `task_metadata.json` (reproducible, but dated)
2. `__init__` fetches live task list (always fresh, but slow and network-dependent)

Option 1 is correct for reproducibility. The config specifies a snapshot version.

**Effort**: **Medium** — design decision needed around staleness, then mechanical.

---

### 4.8 Arithmetic

**Current state**: `task_metadata` hardcoded as a ClassVar. No `_setup()` logic. Simplest benchmark.

**With the proposed pattern**: Already aligned. `make()` is a one-liner.

**Effort**: **Trivial**.

---

## Summary

| Cube | task_metadata source | Runtime state | Infra deps | Effort |
|---|---|---|---|---|
| WorkArena | JSON (after task-metadata PR) | SeedGenerator (lazy) | None | Trivial |
| MiniWob | JSON (static) | subprocess (HTTP server) | None | Trivial |
| SWE-bench Verified | JSON (after uniformization PR) | None | Per-task Docker | Trivial |
| Terminal-Bench | JSON (after uniformization PR) | None | Per-task Docker | Trivial |
| Arithmetic | ClassVar hardcoded | None | None | Trivial |
| OSWorld | JSON (from install()) | None | L1 VM image + L3 VMs | Small |
| WebArena-Verified | Needs install() | None | L2 Docker service | Medium |
| SWE-bench Live | Needs design decision | None | Per-task Docker | Medium |

**Bottom line**: With Nic's uniformization PRs merged, the majority of the work is mechanical — rename `Benchmark` → `BenchmarkConfig`, add `make()`, done. The two non-trivial cases are WebArena-Verified (needs `install()`) and SWE-bench Live (design question).

The cube-standard change (defining the `BenchmarkConfig` base class + updating `Experiment`) is the only design-heavy piece. Once that's done, each cube is an independent 1–2 day mechanical task.

**The deepcopy / `__deepcopy__` complexity disappears entirely** once `BenchmarkConfig` is pure data — `subset_from_list` becomes a trivial dict filter, and no benchmark ever needs to override copy behavior again.

---

# Part 2 — CompositeBenchmark

## Motivation

Running a single `Experiment` across multiple benchmarks today requires either running separate experiments or manually merging task lists. `CompositeBenchmark` provides a first-class way to combine heterogeneous benchmarks into one logical unit.

**Example use case**: a single evaluation run that covers WorkArena L1, MiniWob, and SWE-bench Verified simultaneously — one `Experiment`, one results directory, one summary.

## Design

`CompositeBenchmark` wraps a list of heterogeneous `Benchmark` instances and presents their task metadata as a unified collection. Each task config retains a reference to its source benchmark so `task_config.make()` still receives the correct `runtime_context`.

Key contracts:
- `task_metadata` is the union of all sub-benchmark task metadata (task IDs must be globally unique or namespaced)
- `_setup()` calls `setup()` on each sub-benchmark in order
- `close()` calls `close()` on each sub-benchmark
- `get_task_configs()` yields task configs from all sub-benchmarks, each carrying the right `runtime_context`
- `subset_from_list()` works as usual — the composite is transparent to `Experiment`

## Effort

Medium — the main complexity is namespacing task IDs to avoid collisions across benchmarks, and ensuring each task config routes to the right sub-benchmark at execution time.

---

# Part 3 — BenchmarkPool

## Motivation

Some benchmarks only support one concurrent task per server instance (e.g. WorkArena: each task modifies shared ServiceNow state). To run N tasks in parallel you need K equivalent server instances. `BenchmarkPool` manages this transparently.

## Design

`BenchmarkPool` wraps K identical `Benchmark` instances (one per server). It exposes the same `Benchmark` interface so `Experiment` and `run_with_ray` require minimal changes.

The core scheduling problem: K slots, N workers, tasks must be assigned to a free slot at execution time — not at dispatch time (static pre-assignment wastes idle slots when task durations vary).

**Solution**: a Ray actor acting as a cross-process semaphore. The main process dispatches all N tasks to Ray immediately (non-blocking). Each Ray worker blocks on `actor.acquire()` until a slot is free, runs the task against that server, then calls `actor.release()`. The main process stays free to monitor progress, handle timeouts, and process completions — Ray's scheduler continues to operate normally.

```
Main process:  dispatch all N tasks at once (non-blocking)
                        ↓
Ray workers:   block on actor.acquire() → run task → actor.release()
```

The actor holds the list of `RuntimeContext` dicts (one per slot) — plain dicts that cross process boundaries cleanly. The `Benchmark` instances themselves stay in the main process and are never serialized.

## Effort

Small — the actor is a ~20-line async semaphore; the change to `run_with_ray` is a handful of lines.
