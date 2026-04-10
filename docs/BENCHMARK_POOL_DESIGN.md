# Scalable Parallel Evaluation: BenchmarkPool, CompositeBenchmark & ResourcePool

## Problem

Benchmarks like WorkArena (ServiceNow) and WebArena are backed by external servers that
degrade under load (~7 concurrent agents max). OSWorld tasks require fresh VMs that take
~60s to launch. At RL scale (100s–1000s of parallel rollouts), both patterns become
bottlenecks.

We need three capabilities:
1. **Load distribution** — spread tasks across a pool of identical server instances
2. **Benchmark composition** — run tasks from multiple different benchmarks in a single experiment
3. **Resource pooling** — pre-warm VMs/containers and recycle them between tasks

---

## Design Principle

> **cube-standard is untouched.** All pooling, composition, and resource recycling
> logic lives in cube-harness. The existing `Benchmark`, `ResourceConfig`, `InfraConfig`,
> and `TaskConfig` interfaces are sufficient.

---

## Prior Art

| System | Pattern | Pooling? | Pre-warming? |
|---|---|---|---|
| OSWorld (VMware) | Registry file: free/occupied VMs, snapshot revert | Simple registry | Snapshot = warm state |
| OSWorld (AWS) | Destroy-and-recreate EC2 from AMI per task | No | Image cache only |
| SWE-bench | Ephemeral Docker containers from cached images | No | Docker layer cache |
| SWE-bench (Modal) | Serverless fan-out, platform-managed scaling | Platform-managed | Platform-managed |
| EnvPool | C++ thread pool, fixed homogeneous env array | Fixed pool | Created at init |
| Sample Factory | Persistent worker processes with shared memory | Persistent workers | Yes |

**Gap**: None of these implement a pre-warmed heterogeneous resource pool with
cross-machine coordination. The closest is Sample Factory's persistent-worker model.

---

## Architecture Overview

Three new harness-side abstractions, all implementing or wrapping the existing `Benchmark` interface:

```
┌─────────────────────────────────────────────────────┐
│ CompositeBenchmark(Benchmark)                       │
│                                                     │
│  ┌─────────────────────┐  ┌──────────────────────┐  │
│  │ BenchmarkPool       │  │ SingleBenchmark       │  │
│  │  ├─ WorkArena(A)    │  │  └─ MiniWob           │  │
│  │  ├─ WorkArena(B)    │  │                       │  │
│  │  └─ WorkArena(C)    │  │                       │  │
│  └─────────────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────┘
         ↓ get_task_configs()
   Experiment(benchmark=composite)
         ↓
   run_with_ray(exp, n_cpus=...)
```

| Abstraction | Purpose |
|---|---|
| `BenchmarkPool` | N copies of the **same** benchmark type, different server instances. Round-robin task assignment for load distribution. |
| `CompositeBenchmark` | N **different** benchmark types combined into one evaluation suite. Each sub-benchmark manages its own resources. |
| `ResourcePool` | Pre-warmed VM/container handles for task-scoped resources. Recycles handles between tasks via snapshot revert. |

---

## 1. BenchmarkPool — Load Distribution

Wraps N benchmark instances (same type, different servers) behind a single `Benchmark` interface.

```python
class BenchmarkPool(CubeBenchmark):
    benchmarks: list[CubeBenchmark]

    def _setup(self) -> None:
        for b in self.benchmarks:
            b.setup()

    def close(self) -> None:
        for b in self.benchmarks:
            b.close()

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        # All sub-benchmarks have the same task list; only instance differs.
        # Round-robin assignment ensures even load.
        configs = list(self.benchmarks[0].get_task_configs())
        n = len(self.benchmarks)
        for i, tc in enumerate(configs):
            self._assignments[tc.task_id] = i % n
            yield tc

    def _runtime_context_for(self, task_config: TaskConfig) -> RuntimeContext:
        """Return the runtime context for the sub-benchmark assigned to this task."""
        idx = self._assignments[task_config.task_id]
        return self.benchmarks[idx]._runtime_context
```

### Wiring: per-task runtime context

Today, `Experiment._create_all_episodes()` passes a single `self.benchmark._runtime_context`
to every episode ([experiment.py:89](src/cube_harness/experiment.py#L89)). With a pool, each task needs
the context of its assigned sub-benchmark.

**Required change** — small modification to `Experiment._create_all_episodes()`:

```python
for i, tc in enumerate(task_configs):
    if hasattr(self.benchmark, '_runtime_context_for'):
        ctx = self.benchmark._runtime_context_for(tc)
    else:
        ctx = self.benchmark._runtime_context
    episodes.append(Episode(..., runtime_context=ctx, ...))
```

This keeps `Benchmark` (cube-standard) unchanged; only `Experiment` (cube-harness) learns the new protocol.

### Benchmark implementation changes

`WorkArenaBenchmark` (and equivalent classes in other cubes) gain a required `instance` field.
Instance credentials are passed explicitly at construction time.

```python
class SnowInstanceConfig(TypedBaseModel):
    url: str
    user: str
    pwd: str

class WorkArenaBenchmark(CubeBenchmark):
    instance: SnowInstanceConfig

    def _setup(self) -> None:
        self._runtime_context = {"instance": self.instance, "level": self.level, ...}
```

`WorkArenaTaskConfig.make()` reads the instance from `runtime_context`:

```python
def make(self, runtime_context: RuntimeContext, ...) -> WorkArenaTask:
    return WorkArenaTask(..., instance=runtime_context["instance"])
```

`WorkArenaTask.reset()` injects credentials before BrowserGym reads them.
Safe because each Ray worker is a separate OS process:

```python
def reset(self) -> tuple[Observation, dict]:
    os.environ["SNOW_INSTANCE_URL"] = self._instance.url
    os.environ["SNOW_INSTANCE_UNAME"] = self._instance.user
    os.environ["SNOW_INSTANCE_PWD"] = self._instance.pwd
    ...
```

### Usage

```python
instances = [SnowInstanceConfig(url=..., user=..., pwd=...) for _ in range(3)]
pool = BenchmarkPool(
    benchmarks=[WorkArenaBenchmark(instance=inst, level="l1") for inst in instances]
)
exp = Experiment(name="workarena", benchmark=pool, ...)
run_with_ray(exp, n_cpus=21)   # 3 instances × 7 agents each
```

---

## 2. CompositeBenchmark — Multi-Benchmark Evaluation Suites

Combines tasks from multiple different benchmarks into a single experiment.
Each sub-benchmark manages its own resources independently.

```python
class CompositeBenchmark(CubeBenchmark):
    benchmarks: list[CubeBenchmark]

    def _setup(self) -> None:
        for b in self.benchmarks:
            b.setup()

    def close(self) -> None:
        for b in self.benchmarks:
            b.close()

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        for i, b in enumerate(self.benchmarks):
            for tc in b.get_task_configs():
                self._assignments[tc.task_id] = i
                yield tc

    def _runtime_context_for(self, task_config: TaskConfig) -> RuntimeContext:
        idx = self._assignments[task_config.task_id]
        return self.benchmarks[idx]._runtime_context
```

Note: `CompositeBenchmark` and `BenchmarkPool` share the same `_runtime_context_for` protocol.
They could share a base class or the protocol could be formalized as a mixin.

### Usage

```python
composite = CompositeBenchmark(benchmarks=[
    BenchmarkPool(benchmarks=[WorkArenaBenchmark(instance=inst) for inst in snow_instances]),
    MiniWobBenchmark(),
    OSWorldBenchmark(),
])
exp = Experiment(name="web_agent_suite", benchmark=composite, ...)
```

---

## 3. ResourcePool — Pre-Warmed VMs for Task-Scoped Resources

For benchmarks with `scope="task"` resources (OSWorld, WindowsAgentArena), each task currently
pays the full VM launch cost (~60s). A `ResourcePool` pre-launches N VMs and recycles them.

### Current flow (no pool)

```
Task 1: launch VM → setup → run → close VM     (~60s launch overhead)
Task 2: launch VM → setup → run → close VM     (~60s launch overhead)
...
```

### With ResourcePool

```
Pool init:  launch N VMs upfront
Task 1:     acquire handle → setup → run → revert snapshot → release
Task 2:     acquire handle → setup → run → revert snapshot → release
...
Pool close: close all handles
```

### Implementation: Ray Actor

A Ray Actor manages the pool of live handles. Workers acquire/release handles instead of
launching/closing VMs. This follows Sample Factory's persistent-worker pattern.

```python
@ray.remote
class ResourcePool:
    """Pool of pre-launched resource handles with acquire/release semantics."""

    def __init__(self, infra: InfraConfig, resource: ResourceConfig, pool_size: int):
        self._handles: list[ResourceHandle] = [
            infra.launch(resource) for _ in range(pool_size)
        ]
        self._free: deque[int] = deque(range(pool_size))
        self._infra = infra

    def acquire(self) -> tuple[int, str]:
        """Block until a handle is free. Return (slot_id, endpoint)."""
        while not self._free:
            time.sleep(0.1)
        slot = self._free.popleft()
        return slot, self._handles[slot].endpoint

    def release(self, slot: int) -> None:
        """Revert snapshot and return handle to the pool."""
        self._handles[slot].revert_snapshot()
        self._free.append(slot)

    def close_all(self) -> None:
        for h in self._handles:
            h.close()
```

### Integration with OSWorldTask

`OSWorldTask` currently calls `self.infra.launch(resource)` in `reset()` and `handle.close()`
in `close()` ([osworld task.py:108-117](cubes/osworld-cube/src/osworld_cube/task.py#L108-L117)).

With a pool, the task receives a pre-launched handle instead of launching its own:

```python
# In Episode.run() or task setup:
slot, endpoint = ray.get(pool.acquire.remote())
try:
    task._handle = pool_handle_proxy(slot, endpoint)
    task._computer.attach_endpoint(endpoint)
    # ... run episode ...
finally:
    ray.get(pool.release.remote(slot))
```

### Required additions to ResourceHandle

`ResourceHandle` needs a `revert_snapshot()` method for recycling. This could be:
- Added to the base `ResourceHandle` in cube-standard (small change, good default: no-op)
- Or: the pool calls `infra`-specific revert logic directly (no cube-standard change)

---

## 4. RL-Scale Execution (100s–1000s of Parallel Rollouts)

At RL scale, the bottleneck shifts from server load to resource lifecycle overhead.
The goal: maximize samples/second with minimal idle time.

### Persistent Workers

Instead of creating and destroying resources per-task, use long-lived Ray Actors that
each own a resource handle and process tasks from a queue:

```python
@ray.remote
class EnvWorker:
    """Persistent worker that owns a resource handle and recycles between episodes."""

    def __init__(self, infra: InfraConfig, resource: ResourceConfig):
        self.handle = infra.launch(resource)

    def run_episode(self, episode: Episode) -> Trajectory:
        episode.attach_handle(self.handle)
        result = episode.run()
        self.handle.revert_snapshot()  # recycle
        return result

    def close(self) -> None:
        self.handle.close()
```

### Batched Dispatch

`exp_runner.py` currently submits all episodes at once ([exp_runner.py:76](src/cube_harness/exp_runner.py#L76)).
For RL-scale with resource pools, dispatch in batches:

```python
def run_with_pool(exp: Experiment, workers: list[EnvWorker], episodes: list[Episode]):
    pending: dict[ray.ObjectRef, EnvWorker] = {}
    queue = deque(episodes)

    while queue or pending:
        # Fill idle workers
        for worker in idle_workers(workers, pending):
            if queue:
                episode = queue.popleft()
                ref = worker.run_episode.remote(episode)
                pending[ref] = worker

        # Collect finished
        done, _ = ray.wait(list(pending), timeout=1.0)
        for ref in done:
            yield ray.get(ref)
            del pending[ref]
```

### Scope="benchmark" at RL scale (WorkArena, WebArena)

For shared-server benchmarks, `BenchmarkPool` handles load distribution. The persistent
worker pattern still applies — each worker is assigned to a specific server instance:

```
Workers 0-6   → instance A (WorkArena)
Workers 7-13  → instance B (WorkArena)
Workers 14-20 → instance C (WorkArena)
```

Workers pull tasks from a shared queue, filtered to their assigned instance.

### Scope="task" at RL scale (OSWorld)

For per-task-VM benchmarks, `ResourcePool` handles pre-warming and recycling.
Each worker owns a VM and reverts between episodes.

Workers pull tasks from a shared queue. No instance affinity needed since
each worker has its own VM.

---

## Concurrency Model

| Scenario | Mechanism | Concurrency limit |
|---|---|---|
| BenchmarkPool (WorkArena) | Static round-robin + `n_cpus` | `N_instances × max_concurrent_agents` |
| ResourcePool (OSWorld) | Ray Actor semaphore | `pool_size` |
| RL persistent workers | Fixed worker count | `N_workers` |

Static round-robin is sufficient for evaluation. For RL (where task runtimes vary widely),
the persistent-worker model naturally load-balances since workers pull tasks when free.

---

## Summary: What Changes Where

| Repo | File | Change | Size |
|---|---|---|---|
| cube-harness | `src/cube_harness/benchmark_pool.py` | New — `BenchmarkPool` class | S |
| cube-harness | `src/cube_harness/composite_benchmark.py` | New — `CompositeBenchmark` class | S |
| cube-harness | `src/cube_harness/resource_pool.py` | New — `ResourcePool` Ray Actor | M |
| cube-harness | `src/cube_harness/experiment.py` | Support `_runtime_context_for()` protocol | S |
| cube-harness | `src/cube_harness/exp_runner.py` | Add `run_with_pool()` for persistent workers (RL path) | M |
| workarena_cube | `benchmark.py` | Add required `instance: SnowInstanceConfig` field | S |
| workarena_cube | `task.py` | Read instance from `runtime_context`, set env vars | S |
| cube-standard | *(none)* | No changes | — |

---

## Non-Goals

- **Dynamic load balancing**: Static round-robin is sufficient initially. Persistent workers naturally load-balance for RL.
- **Cross-machine resource coordination**: ResourcePool runs within a single Ray cluster. Multi-cluster is out of scope.
- **Auto-scaling**: Pool sizes are fixed at init. Elastic scaling is a future concern.

---

## Open Questions

1. **`revert_snapshot()` on ResourceHandle**: Add to cube-standard as a no-op base method, or keep it infra-specific in cube-harness?
2. **Shared task queue vs. static assignment**: For BenchmarkPool, static round-robin is simple but doesn't account for variable task durations. Worth adding a dynamic queue upfront?
3. **Episode serialization with handles**: Ray workers need handles but `ResourceHandle` is not serializable. Persistent workers (Ray Actors) solve this. Is there a simpler path?
