# SWE cube migration: BenchmarkConfig split + generic types + debug contract

**Status:** Proposed
**Date:** 2026-04-30
**Scope:** `swebench-verified-cube`, `swebench-live-cube`, `terminalbench-cube`
**Branch:** `feat/benchmark-config` (cube-harness)
**Requires:** cube-standard `nico_fix` PR #124 (generic `TaskConfig` / `BenchmarkConfig` / `Benchmark`) — already pinned as the cube-standard rev in `pyproject.toml`
**Related:** cube-standard archived `2026-04-24-benchmark-config`, `2026-04-27-typed-task-execution-info`

---

## Context

Three changes landed in cube-standard that together define the new cube contract:

| cube-standard change | What it introduced |
|-----|----|
| PR #118 — `feat/benchmark-config` | `BenchmarkConfig` / `Benchmark` split. `BenchmarkConfig` is the serialisable registry and entry-point target; owns `install()`, `get_task_configs()`. `Benchmark` is pure runtime. |
| PR #121 — `feat/typed-task-execution-info` | Typed `TaskExecutionInfo` slot replaces `extra_info: dict`; `metadata` field stamped on `TaskConfig` by `get_task_configs()` so workers never import `BenchmarkConfig`. |
| PR #124 — `nico_fix` (open) | `Task`, `TaskConfig`, `BenchmarkConfig`, `Benchmark` made generic via PEP 695. Cubes write `class FooTaskConfig(TaskConfig[FooTaskMetadata]):` — no `# type: ignore` re-annotations. |

All three SWE cubes pre-date these changes and violate the current spec.

---

## Pre-conditions confirmed

| Item | Status |
|------|--------|
| cube-standard `feat/benchmark-config` merged (PR #118, Apr 28) | ✅ |
| cube-standard `nico_fix` pinned in cube-harness `pyproject.toml` (commit `7711c73`) | ✅ |
| `swe_agent_recipe` PR merged — deleted 3 old-style hello_* recipes (commit `40b45513`) | ✅ |
| Infra pattern already correct (`InfraConfig`, `cleanup_stale()`, `launch_task_container()`) | ✅ |
| No unit tests exist for any of the three cubes | ⚠️ gap — addressed here |

---

## Problem inventory (identical across all three cubes)

| # | Violation | All three? |
|---|-----------|:----------:|
| 1 | `Benchmark` class holds ClassVars (`benchmark_metadata`, `task_metadata`, `task_config_class`) — must be on `BenchmarkConfig` | ✗ |
| 2 | `Benchmark` class holds user fields (`infra`, `oracle_mode`, `include_hints`) — must be on `BenchmarkConfig` | ✗ |
| 3 | `install()` / `uninstall()` / `get_task_configs()` on `Benchmark` — must be on `BenchmarkConfig` | ✗ |
| 4 | `get_task_configs()` emits `task_id=tm.id` — must stamp `metadata=tm` | ✗ |
| 5 | `task.py` circular import: `BenchmarkClass.task_metadata[self.task_id]` — eliminated by stamping | ✗ |
| 6 | Entry point names `Benchmark` — must name `BenchmarkConfig` | ✗ |
| 7 | `debug.py` calls `install()` + `setup()` inside `get_debug_benchmark()` — pure factory required | ✗ |
| 8 | `task.py make()` keeps deprecated `container_backend` param | swebench-live + terminalbench pass it to Task; swebench-verified declares but doesn't use it |
| 9 | No unit tests | ✗ |

**Additionally (cube-harness shared):**
- `debug_harness.py` calls `bench.setup()` after `get_debug_benchmark()` — will break once cubes return `BenchmarkConfig` (which has no `.setup()`)

---

## Per-cube differences

| Aspect | swebench-verified | swebench-live | terminalbench |
|--------|:-----------------:|:-------------:|:-------------:|
| User fields on Config | `include_hints`, `oracle_mode`, `infra` | `include_hints`, `oracle_mode`, `infra` | `oracle_mode`, `infra` |
| `container_backend` in `make()` | declared unused — drop signature | fallback path + passed to Task — remove both | fallback path + passed to Task — remove both |
| `ContainerBackend` import in `task.py` | not imported | imported, now unused — remove | imported, now unused — remove |

---

## Migration pattern

### `benchmark.py` — split into Config + Benchmark

```python
class SWEBenchVerifiedBenchmark(Benchmark["SWEBenchVerifiedBenchmarkConfig"]):

    def _setup(self) -> None:
        self.config.infra.cleanup_stale()
        self._runtime_context["infra"] = self.config.infra

    def close(self) -> None:
        pass


class SWEBenchVerifiedBenchmarkConfig(BenchmarkConfig[SWEBenchVerifiedTaskMetadata]):

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(...)
    task_metadata: ClassVar[dict[str, SWEBenchVerifiedTaskMetadata]]
    task_config_class: ClassVar[type[TaskConfig]] = SWEBenchVerifiedTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = SWEBenchVerifiedBenchmark

    include_hints: bool = False
    oracle_mode: bool = False
    infra: InfraConfig = Field(default_factory=LocalInfraConfig)

    @classmethod
    def install(cls) -> None: ...

    @classmethod
    def uninstall(cls) -> None: ...

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        for tm in self.tasks().values():
            yield SWEBenchVerifiedTaskConfig(
                metadata=tm,                   # ← stamp; not task_id=tm.id
                tool_config=self.tool_config,
                seed=None,
                include_hints=self.include_hints,
                oracle_mode=self.oracle_mode,
            )
```

Note: `infra` is accessed via `self.config.infra` inside `_setup()` — `Benchmark` has no fields of its own.

### `task.py` — drop circular import, drop deprecated param

```python
class SWEBenchVerifiedTaskConfig(TaskConfig[SWEBenchVerifiedTaskMetadata]):

    include_hints: bool = False
    oracle_mode: bool = False

    def make(self, runtime_context: RuntimeContext | None = None) -> SWEBenchVerifiedTask:
        if runtime_context is None or "infra" not in runtime_context:
            raise ValueError("requires runtime_context['infra']")

        exec_info = self.load_task_execution_info(self.task_id)
        metadata = self.metadata.model_copy(         # ← self.metadata, no import
            update={"extra_info": {**exec_info,
                                   "include_hints": self.include_hints,
                                   "oracle_mode": self.oracle_mode}}
        )
        return SWEBenchVerifiedTask(
            metadata=metadata,
            tool_config=self.tool_config or SWEBenchToolConfig(),
            runtime_context=runtime_context,
        )
```

### `debug.py` — pure factory

```python
def get_debug_benchmark(infra: InfraConfig | None = None) -> SWEBenchVerifiedBenchmarkConfig:
    return SWEBenchVerifiedBenchmarkConfig(
        infra=infra or LocalInfraConfig(),
        oracle_mode=True,
    ).subset_from_list(list(_TASK_ACTIONS))
```

No `install()`, no `setup()`. The harness owns both.

### `debug_harness.py` — call `install()` + `make()` instead of `setup()`

```python
# integration-tests/cube_integration_tests/debug_harness.py

config = cube_debug_module.get_debug_benchmark(infra=infra)
config.install()
benchmark = config.make()
task_configs = [tc for tc in config.get_task_configs() if tc.task_id == task_id]
tc = task_configs[0]
task = tc.make(runtime_context=benchmark._runtime_context)
```

### `pyproject.toml`

```toml
swebench-verified-cube = "swebench_verified_cube.benchmark:SWEBenchVerifiedBenchmarkConfig"
```

### `__init__.py`

```python
from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmark, SWEBenchVerifiedBenchmarkConfig
```

---

## Files changed

Per cube (×3):

| File | Change |
|------|--------|
| `benchmark.py` | Split into `BenchmarkConfig` + `Benchmark`; use generic types |
| `task.py` | Drop `container_backend`; remove circular import; use `self.metadata` |
| `debug.py` | Pure factory — drop `install()` and `setup()` |
| `__init__.py` | Add `BenchmarkConfig` export |
| `pyproject.toml` | Entry point → `BenchmarkConfig` class |
| `tests/test_benchmark.py` (new) | Unit tests — no Docker required |

Shared:

| File | Change |
|------|--------|
| `recipes/swe_agent_recipe.py` | Update 3 instantiation sites to `BenchmarkConfig` |
| `integration-tests/cube_integration_tests/debug_harness.py` | `bench.setup()` → `config.install()` + `config.make()` |

**Total: 16 files, ~350–400 lines changed.**

---

## Unit test plan

Five Docker-free tests per cube (`tests/test_benchmark.py`):

| Test | What it validates |
|------|-------------------|
| `test_config_roundtrip` | `model_dump()` → `model_validate()` round-trips cleanly |
| `test_task_metadata_loaded` | ClassVar populated at import; correct task count (500 / 1895 / 89) |
| `test_get_task_configs_stamps_metadata` | Every emitted `TaskConfig.metadata.id` matches the task |
| `test_subset_from_list` | Scopes to exactly the requested task IDs |
| `test_debug_benchmark_type` | `get_debug_benchmark()` returns a `BenchmarkConfig` instance |

Existing `integration-tests/test_debug_matrix.py` covers e2e and does not need to change.

---

## Commit plan

```
feat(swebench-verified): migrate to BenchmarkConfig split + generic types
feat(swebench-live): migrate to BenchmarkConfig split + generic types
feat(terminalbench): migrate to BenchmarkConfig split + generic types
test: add unit tests for swebench-verified, swebench-live, terminalbench
fix(recipe): update swe_agent_recipe to BenchmarkConfig classes
fix(debug-harness): call install()+make() instead of setup()
```

See [deltas.md](deltas.md) for the two spec clarifications.
