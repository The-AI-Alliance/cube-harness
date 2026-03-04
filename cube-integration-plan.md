# CUBE → AgentLab2 Integration

Make `cube-standard` a dependency of AgentLab2 and establish a clean separation of concerns: CUBE owns the benchmark/task/tool protocol; AL2 owns the agent, LLM, episode-running, and trajectory layers.

The strategy is to introduce the new cube-based path **alongside** the existing one, deprecating old AL2 classes without breaking anything. MiniWob and WorkArena still work as-is. New benchmarks **must** use the cube API directly.

---

## What's Done

### Core types now imported from cube
`TypedBaseModel`, `ActionSchema`, `Action`, `StepError`, `STOP_ACTION`, `Content` (+subclasses), `Observation`, `EnvironmentOutput`, `AbstractTool`, `ToolConfig`, `tool_action` — previously defined in AL2, now imported directly from `cube.core` / `cube.tool` / `cube.task`. All AL2 internals and benchmarks (including MiniWob and WorkArena) import these types directly from cube. `base.py` deleted.

### Tool action discovery
Protocol-based `Tool` replaced by `ToolWithTelemetry(cube.tool.Tool)` — an AL2-specific subclass that wraps `execute_action` with OpenTelemetry tracing. `BrowserActionSpace` / `BidBrowserActionSpace` converted from Protocols to plain ABCs (inheriting `ABC` only), with `@tool_action @abstractmethod` on each action. Concrete tool classes combine both via multiple inheritance: e.g. `SyncPlaywrightTool(ToolWithTelemetry, BrowserActionSpace)`, `BrowsergymTool(ToolWithTelemetry, BidBrowserActionSpace)`. All tool classes (`BrowsergymTool`, `SyncPlaywrightTool`, `AsyncPlaywrightTool`, `Toolbox`, `Computer`) updated accordingly.

### Deprecation layer
AL2's `Task`, `Environment`, `EnvConfig`, `AbstractEnvironment`, and `Benchmark` are kept exclusively to support MiniWob and WorkArena. They are now marked deprecated via `__init_subclass__` warnings, constructor warnings, and docstrings. `Benchmark.install()`/`uninstall()` removed. These classes will be deleted once MiniWob and WorkArena are refactored and moved to the `cubes/` folder.

### Dual-path Episode + Experiment
`EpisodeConfig` gains a `task_config: TaskConfig | None` field (old `task_id`/`tool_config` made optional for backward-compat with stored configs). `Episode.run()` dispatches to `_run_cube_task()` (new cube path) or `_run_legacy()` (existing path, unchanged). `Experiment._create_all_episodes()` dispatches on `CubeBenchmark` vs `AL2Benchmark` with a deprecation warning on the legacy path. `load_episode_from_config(benchmark=None)` — cube configs are self-contained (no benchmark arg needed); legacy configs still require it.

### Arithmetic toy cube (`cubes/arithmetic-cube/`)
Self-contained installable package demonstrating the full cube pattern end-to-end: `ArithmeticTool`, `ArithmeticTask(cube.Task)`, `ArithmeticTaskConfig`, `ArithmeticBenchmark(cube.Benchmark)`. Validated with `cube.testing.run_debug_suite`.

---

## What's Left

1. **Future work (tracked separately)**:
   - Refactor MiniWob (#204) and WorkArena (#205) to the cube Task/Benchmark API and move them to the `cubes/` folder.
   - Delete the deprecated AL2 `Task`, `Benchmark`, `Environment`, and `EnvConfig` classes along with the legacy episode path. Then remove the tests covering these old classes, and move them to the cube-standard repository if needed (#206)

---

## How to Review

The PR is structured as a sequence of independent steps. Reviewing file by file in this order is easiest:

1. **`pyproject.toml` / `Makefile`** — cube-standard added as a dependency.
2. **`src/agentlab2/core.py`, `tool.py`, `environment.py`, `benchmark.py`** — deprecated class definitions and re-exports. Check that deprecation warnings are in the right place and nothing is removed prematurely.
3. **`src/agentlab2/action_spaces/`** — Protocol → ABC migration. Verify `@tool_action @abstractmethod` placement and that subclasses don't repeat the decorator unnecessarily.
4. **`src/agentlab2/tools/`** — Each tool class updated to inherit from `ToolWithTelemetry`. Check that `make(container=None)` signatures are consistent.
5. **`src/agentlab2/episode.py`** — Dual-path logic. Focus on `EpisodeConfig` field additions, `run()` dispatch, and `load_episode_from_config()` branching.
6. **`src/agentlab2/experiment.py`** — `isinstance` dispatch. Verify the deprecation warning fires on the legacy path and that the new path creates episodes correctly.
7. **`cubes/arithmetic-cube/`** — Toy benchmark. Read as documentation of the intended cube API usage pattern.
8. **`tests/`** — Existing tests pass unchanged. New mocks (`MockCubeTask`, `MockCubeTaskConfig`, `MockCubeBenchmark`) are in `conftest.py`. New cube-path tests are in `test_cube_episode.py` and `test_cube_experiment.py`.
