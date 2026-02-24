# CUBE → AgentLab2 Integration Plan

## Context

Make `cube-standard` a dependency of `AgentLab2` and remove AL2 classes that are now covered by CUBE. This yields a clean separation: CUBE owns the benchmark/task/tool protocol; AL2 owns the agent, LLM, episode-running, and trajectory layers.

---

## Alignment Map

### ✅ Fully Aligned — DONE

| Concept | CUBE location | AL2 location | Status |
| --- | --- | --- | --- |
| `TypedBaseModel` | `cube.core` | ~~`agentlab2.base`~~ deleted | ✅ |
| `ActionSchema` | `cube.core` | ~~defined in `agentlab2.core`~~ | ✅ |
| `Action` | `cube.core` | ~~defined in `agentlab2.core`~~ | ✅ |
| `StepError` | `cube.core` | ~~defined in `agentlab2.core`~~ | ✅ |
| `STOP_ACTION` | `cube.task` | ~~defined in `agentlab2.environment`~~ | ✅ |
| `Content` (+ subclasses) | `cube.core` | ~~defined in `agentlab2.core`~~ | ✅ |

All six are now imported from `cube` in AL2. `base.py` has been deleted.

### ⚠️ Partially Aligned — TODO

| Concept | CUBE | AL2 | Delta |
| --- | --- | --- | --- |
| `Observation` | `cube.core` | defined in `agentlab2.core` | Content done — migrate `Observation` next, then `EnvironmentOutput` follows |
| `EnvironmentOutput` | has `truncated` field | missing `truncated` | Blocked on `Observation` migration; import from cube once `Observation` is done |
| `AbstractTool` | no `close()` method | has `close()` | Add `close()` to cube's `AbstractTool`, then import from cube |
| `ToolConfig` | `make(container=None)` | `make()` | Update `BrowsergymConfig.make()` signature, then import from cube |
| `tool_action` decorator | `cube.tool` | not present in AL2 | Used when migrating `Tool` and `BrowsergymTool` |

### ❌ Diverged — TODO (major rework)

#### 1. ~~Content system~~ — DONE

AL2's `Content` class removed and replaced by cube's polymorphic hierarchy (`TextContent`, `StructuredContent`, `ImageContent`, …). All call sites use `Content.from_data()` and `to_llm_message()`.

#### 2. Task / Environment split → unified Task
- **CUBE**: Single `Task(ABC)` owns `tool_config: ToolConfig`, creates its own tool in `model_post_init`, exposes `reset() → (Observation, dict)` and `step(action) → EnvironmentOutput` and `evaluate(obs) → (float, dict)`.
- **AL2**: Task and Environment are two classes.
  - `Task.setup(tool)` — task logic only, tool injected
  - `Environment(task, tool)` wraps the two and drives the loop via `Environment.step()`
  - `EnvConfig(task, tool_config)` pairs them
- **Change**: Migrate AL2's `Task` subclasses to cube's `Task` contract:
  - `setup(tool)` → `reset()` (task now owns its tool via `tool_config`)
  - `validate_task(obs)` → `evaluate(obs) -> tuple[float, dict]`
  - `finished()` (no args) → `finished(obs: Observation) -> bool`
  - Remove `Environment`, `AbstractEnvironment`, `EnvConfig` from AL2
  - Update `Episode.run()` to call `task.reset()` / `task.step()` directly

#### 3. Benchmark
- **CUBE**: ClassVar-based metadata (`BenchmarkMetadata`, `task_metadata: dict[str, TaskMetadata]`, `task_config_class`). Factory pattern: `get_task_configs() → Generator[TaskConfig]` returns configs, caller creates tasks.
- **AL2**: Instance field `metadata: dict`. Direct object creation: `load_tasks() → list[Task]`. `env_configs()` wraps them.
- **Change**:
  - Add `BenchmarkMetadata` + `TaskMetadata` typed structs (import from cube)
  - Add `TaskConfig(ABC)` subclasses per benchmark
  - Replace `load_tasks()` with `get_task_configs()`
  - Move `tool_config` from Benchmark field → `default_tool_config` (cube convention)
  - Remove `env_configs()`

#### 4. Tool action discovery
- **CUBE**: `@tool_action` decorator marks methods; `Tool.action_set` auto-discovers via introspection.
- **AL2**: `Tool.action_space` is a Protocol; `action_set` extracts from protocol members.
- **Change**: Switch `BrowsergymTool` (and any other Tool subclasses) to `@tool_action`-decorated methods. Then import `Tool` from cube.

---

## Impact on Implemented Benchmarks

### MiniWob (`benchmarks/miniwob/`)

**`MiniWobTask`** changes:
- `setup(tool: BrowserTaskTool)` → `reset() -> (Observation, dict)` (tool accessed via `self.tool`)
- `validate_task()` → `evaluate(obs: Observation) -> tuple[float, dict]`
- `finished()` (no args) → `finished(obs: Observation) -> bool`
- Add `tool_config: ToolConfig` field (injected, no default)

**`MiniWobBenchmark`** changes:

- Add `benchmark_metadata: ClassVar[BenchmarkMetadata]`
- Add `task_metadata: ClassVar[dict[str, TaskMetadata]]`
- Add `MiniWobTaskConfig(TaskConfig)` with `make()` returning `MiniWobTask`
- Replace `load_tasks()` with `get_task_configs()`
- Remove `env_configs()` (inherited method disappears)

### WorkArena (`benchmarks/workarena/`)

Same changes as MiniWob apply.

---

## AL2-Only Concepts to Keep (not in CUBE, stay in AL2)

- `Trajectory`, `TrajectoryStep` — execution records
- `AgentOutput`, `LLMCall`, `Usage` — agent-side data
- `Episode`, `EpisodeConfig`, `Experiment`, `ExpResult` — experiment orchestration
- `LLMConfig`, `LLM`, `Prompt` — LLM layer
- `Agent`, `AgentConfig`, `ReactAgent`, `ReactAgentConfig` — agent abstractions
- `FileStorage`, `Storage` — trajectory persistence
- `BrowsergymConfig`, `BrowsergymTool` — browser-specific tool impl (reworked to cube Tool API)
- Telemetry / metrics

---

## Files Status

| File | Change | Status |
| --- | --- | --- |
| `pyproject.toml` | Add `cube-standard` as dependency | ✅ Done |
| `.vscode/settings.json` | Add `../cube-standard/src` to `extraPaths` | ✅ Done |
| `Makefile` | Clone cube on install, pull on update | ✅ Done |
| `src/agentlab2/base.py` | Delete — `TypedBaseModel` comes from `cube.core` | ✅ Done |
| `src/agentlab2/core.py` | Remove `ActionSchema`, `Action`, `StepError`; import from `cube.core` | ✅ Done |
| `src/agentlab2/llm.py` | Import `TypedBaseModel` from `cube.core` | ✅ Done |
| `src/agentlab2/episode.py` | Import `TypedBaseModel` from `cube.core` | ✅ Done |
| `src/agentlab2/environment.py` | Import `STOP_ACTION` from `cube.task` | ✅ Done |
| `src/agentlab2/core.py` | Remove `Content`; import from `cube.core` | ✅ Done |
| `src/agentlab2/tool.py` | Use `Content.from_data()` | ✅ Done |
| `src/agentlab2/tools/browsergym.py` | Use `Content.from_data()` | ✅ Done |
| `src/agentlab2/tools/playwright.py` | Use `Content.from_data()` | ✅ Done |
| `src/agentlab2/benchmarks/miniwob/task.py` | Use `Content.from_data()` (obs_postprocess) | ✅ Done |
| `src/agentlab2/core.py` | Remove `Observation`, `EnvironmentOutput`; import from cube | TODO (after Observation migration) |
| `src/agentlab2/tool.py` | Import `AbstractTool`, `ToolConfig`, `tool_action`, `Tool` from cube; update `make()` sig; add `close()` to cube's `AbstractTool` | TODO |
| `src/agentlab2/environment.py` | Delete entirely (merged into cube's Task) | TODO (after Task migration) |
| `src/agentlab2/benchmark.py` | Refactor to cube's ClassVar pattern | TODO |
| `src/agentlab2/episode.py` | Remove `EnvConfig` usage; call `task.reset()` / `task.step()` directly | TODO (after Task migration) |
| `src/agentlab2/tools/browsergym.py` | Update `make(container=None)`, replace Protocol with `@tool_action` | TODO |
| `src/agentlab2/benchmarks/miniwob/task.py` | Migrate to cube Task API | TODO |
| `src/agentlab2/benchmarks/miniwob/benchmark.py` | Migrate to cube Benchmark API; add `MiniWobTaskConfig` | TODO |
| `src/agentlab2/benchmarks/workarena/task.py` | Migrate to cube Task API | TODO |
| `src/agentlab2/benchmarks/workarena/benchmark.py` | Migrate to cube Benchmark API; add `WorkArenaTaskConfig` | TODO |

---

## Verification

1. `make test` in `cube-standard/` must still pass (no changes there)
2. `make test` in `AgentLab2/` after each migration step
3. Manual smoke test: instantiate `MiniWobBenchmark`, call `get_task_configs()`, call `TaskConfig.make()`, call `task.reset()` and `task.step()`
4. Run a short `Experiment` with `ReactAgent` on MiniWob (1–2 tasks) end-to-end
