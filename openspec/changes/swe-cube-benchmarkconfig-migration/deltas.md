# Deltas — SWE cube BenchmarkConfig migration

No cube-harness specs are modified by this change — all spec contracts (`debug_harness.py`
behaviour, recipe patterns) are already correct in the specs; the cubes were simply not
compliant with them.

Two clarifications are surfaced for **cube-standard** specs and tracked there:

---

## Upstream delta 1 — `testing/spec.md`: `get_debug_benchmark()` must be a pure factory

The testing spec correctly states that `get_debug_benchmark()` returns a `BenchmarkConfig`
and that the harness owns `config.install()` / `config.make()`. The **must-not** side
was implicit. Proposed addition to cube-standard:

> `get_debug_benchmark()` **MUST NOT** call `install()`, `setup()`, or any other lifecycle
> method. The function is a pure factory. Side effects here double-install when the harness
> also calls `install()`, and calling `.setup()` on the returned `BenchmarkConfig` will
> raise `AttributeError` (that method lives on `Benchmark`, not `BenchmarkConfig`).

---

## Upstream delta 2 — `benchmark/spec.md`: migration guide

Add a "Migrating from pre-split cubes" subsection to cube-standard's benchmark spec as a
canonical recipe for cube authors. Steps:

1. Rename existing class → `FooBenchmarkConfig(BenchmarkConfig[FooTaskMetadata])`. Move all
   ClassVars, user fields, `install()`, `uninstall()`, `get_task_configs()` here. Add
   `benchmark_class: ClassVar[type[Benchmark]] = FooBenchmark`.
2. Create `FooBenchmark(Benchmark["FooBenchmarkConfig"])` with only `_setup()` and `close()`.
   Access user fields via `self.config.<field>`.
3. Update `get_task_configs()`: stamp `metadata=tm` on each yielded `TaskConfig` (not `task_id=tm.id`).
4. Update `task.py`: replace `BenchmarkClass.task_metadata[self.task_id]` with `self.metadata`.
   Remove the benchmark import and the deprecated `container_backend` parameter from `make()`.
5. Update `debug.py`: return a bare `BenchmarkConfig` — no `install()`, no `setup()`.
6. Update `pyproject.toml` entry point to name `FooBenchmarkConfig`.
7. Update `__init__.py` to export `FooBenchmarkConfig`.
