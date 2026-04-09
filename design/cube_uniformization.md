# Plan: pre-populate task_metadata before benchmark instantiation

## Goal

`task_metadata` should be available on the benchmark class **before** any instance is created — without calling `setup()`. This allows harnesses, UIs, and tooling to inspect tasks without spinning up infrastructure.

### Mechanism (cube-standard #77)

- `install()` / `uninstall()` are now **classmethods** — callable as `MyBenchmark.install()`.
- `install()` is responsible for downloading data **and** saving `task_metadata.json` next to the benchmark module, containing **all tasks from all splits** (no filtering).
- `__init_subclass__` auto-loads `task_metadata.json` at import time if `task_metadata = {}`.
- `setup()` raises a clear `RuntimeError` if `task_metadata` is still empty.
- Splits/filters stay in user-land via `subset_from_list` / `subset_from_glob`.

---

## Cube status

| Cube | Status | Notes |
| ---- | ------ | ----- |
| `arithmetic-cube` | ✅ nothing to do | Static hardcoded dict at class definition |
| `miniwob` | ✅ done | Replaced intermediate `miniwob_tasks.json` with proper `task_metadata.json` (generated via `TaskMetadata.model_dump()`); `task_metadata = {}` placeholder lets `__init_subclass__` auto-load it |
| `webarena-verified` | ✅ done | `install()` calls `WebArenaVerified().get_tasks()` and saves `task_metadata.json`; idempotent (skips if file exists) |
| `swebench-verified-cube` | ✅ done | `install()` downloads HF `test` split, saves JSON; `_setup()` applies filters from pre-loaded metadata |
| `swebench-live-cube` | ✅ done | `install()` downloads all 4 HF splits (verified/full/test/lite), merges 1895 unique tasks with priority `verified>full>test>lite`, logs per-field diffs on conflict; `named_subsets` on `BenchmarkMetadata` maps split names to `subset_from_glob` patterns; recipe uses `.named_subset("lite")` |
| `osworld-cube` | ✅ done | `install()` clones repo (needed at runtime for settings_file paths) + saves `task_metadata.json` with all 4 test sets; `extra_info.test_sets` tracks membership; `named_subsets` for all 4 sets; `_setup()` applies `_fix_config_paths()`; debug tasks moved to `DebugOSWorldBenchmark` in `debug.py` with no-op `install()`/`uninstall()` and correct `num_tasks=2` |
| `terminalbench-cube` | ✅ done | `install()` clones repo, reads task dirs, saves `task_metadata.json` (archive base64-encoded); `_setup()` applies filters + oracle_mode; `Task.reset()` decodes archive (base64 string from JSON, or raw bytes from `_setup()`) |
| `workarena` | ✅ done | `install()` enumerates all task types (L1 + L2/L3 agent superset), marks `in_human_curriculum` per task, saves `task_metadata.json`; `WorkArenaSeedGenerator` wraps `get_all_tasks_agents()` and plugs into `Benchmark.seed_generator`; `_setup()` filters by level + sets seed generator; `named_subsets` for l1/l2/l3; human curriculum via `.subset_from_glob("extra_info.in_human_curriculum", "True")`; debug uses `subset_from_list` on first 2 L1 tasks |

## Order of work

1. `webarena-verified` — simplest: no download, just library call → JSON ✅
2. `swebench-verified-cube` — single HF split, straightforward ✅
3. `swebench-live-cube` — multi-split merge ✅
4. `terminalbench-cube` — install() already does the heavy lifting
5. `osworld-cube` — install() already downloads; needs JSON generation added
6. `workarena` — multi-level merge, most complex

---

## Phase 2: Typed public metadata + per-task execution cache

### Motivation

`task_metadata.json` can be very large (swebench-live ~557 MB) because tasks embed heavy fields (`problem_statement`, `patch`, `test_patch`, etc.) that are only needed at execution time. These fields pollute public introspection and slow down import.

### Design

**1. `task_metadata.json` is a shipped package resource**

Committed to the cube repo, included in the package release. Available at import time with no download required. Contains only lightweight public fields (small enough to load without hesitation).

**2. Typed `TaskMetadata` subclasses replace `extra_info` for public fields**

Each cube that needs cube-specific public fields defines a subclass. `extra_info` starts empty and is reserved exclusively for runtime execution data loaded in `TaskConfig.make()`.

```python
# swebench_verified_cube/task.py
class SWEBenchVerifiedTaskMetadata(TaskMetadata):
    repo: str
    difficulty: str = "unknown"
    splits: list[str] = []
    # NO problem_statement / patch / test_patch here
```

`task_metadata.json` is serialized with `_type` so `__init_subclass__` reconstructs the right subclass. Cubes with no extra public fields skip this entirely.

**3. Per-task execution data in `get_cache_dir(benchmark_name) / "tasks_execution_info"`**

Heavy fields live in one JSON file per task in the user's cube cache dir:
```
~/.cube/swebench-verified-cube/tasks_execution_info/django__django-12345.json
  → {"problem_statement": "...", "patch": "...", "test_patch": "...", ...}
```

`cube-standard` adds two classmethods to `Benchmark`:
```python
@classmethod
def task_execution_cache_dir(cls) -> Path:
    return get_cache_dir(cls.benchmark_metadata.name) / "tasks_execution_info"

@classmethod
def load_task_execution_info(cls, task_id: str) -> dict[str, Any]:
    cache_file = cls.task_execution_cache_dir() / f"{task_id}.json"
    if not cache_file.exists():
        raise RuntimeError(f"No execution data for {task_id!r}. Run `{cls.__name__}.install()`.")
    return json.loads(cache_file.read_text())
```

**4. Cube `TaskConfig.make()` loads execution info lazily**

```python
def make(self, runtime_context=None, container_backend=None):
    metadata = SWEBenchVerifiedBenchmark.task_metadata[self.task_id]
    exec_info = SWEBenchVerifiedBenchmark.load_task_execution_info(self.task_id)
    metadata = metadata.model_copy(update={"extra_info": exec_info})
    return SWEBenchVerifiedTask(metadata=metadata, ...)
```

`Task.reset()` and `Task.evaluate()` access `self.metadata.extra_info` as before — no changes to task logic. Works correctly in Ray workers because it reads from disk.

> **Ray worker note:** `_setup()` modifications to `task_metadata` (e.g. `oracle_mode=True`) are invisible to Ray workers. Phase 2 `load_task_execution_info()` in `make()` is the correct place to handle this.

**5. `named_subsets` glob keys update**

Public fields that moved from `extra_info` to typed subclass fields update their glob keys: `"extra_info.splits"` → `"splits"`, etc.

**6. Cubes with no heavy data**

Cubes like `miniwob`, `arithmetic`, `workarena` have no per-task execution files at all. They skip steps 3–4 entirely. Their `task_metadata.json` (already a package resource) is the complete picture.

---

## Phase 3: `install()` responsibility split

### Motivation

Once `task_metadata.json` is a shipped package resource, `install()` no longer needs to generate it. Its only remaining job is populating the runtime execution cache for cubes that have heavy data.

### Design

**`install()` classmethod** (kept in the public API for cubes that need it):
- Responsibility: download / generate per-task execution cache files into `task_execution_cache_dir()`
- Cubes with no heavy data: don't define it (base no-op)
- Called by users before running tasks: `MyCube.install()`

**`scripts/generate_task_metadata.py`** (developer-only, not part of the published package):
- Responsibility: regenerate `task_metadata.json` from scratch (e.g. re-download HF dataset, re-read repo files)
- Lives in the cube's source repo for reproducibility
- Not expected from all cube developers — only needed when the task list itself changes
- Not imported or shipped in the wheel

This cleanly separates "what users run to set up a cube" (`install()`) from "what cube developers run to update the shipped metadata" (`scripts/generate_task_metadata.py`).
