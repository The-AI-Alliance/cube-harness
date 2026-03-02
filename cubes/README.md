# Cubes

This folder contains concrete CUBE implementations — self-contained Python packages that wrap specific benchmarks (e.g., WebArena, SWE-bench, OSWorld) using the `cube` standard library.

Each cube typically provides:

- **`Tool` subclass** — defines the action space the agent can use (e.g., browser clicks, shell commands), implemented with `@tool_action`-decorated methods
- **`ToolConfig`** — serializable config that instantiates the tool (optionally connecting to a container)
- **`Task` subclass** — implements `reset()` (initial observation) and `evaluate()` (scoring), plus optional `finished()` and `filter_actions()`
- **`TaskConfig`** — serializable config with a `make()` method that instantiates the task, wiring together metadata, tool config, and runtime context
- **`Benchmark` subclass** — manages the full task collection: declares `benchmark_metadata`, `task_metadata`, `task_config_class`, and implements `_setup()` / `close()` for shared infrastructure
- **`Container` config** *(optional)* — if the benchmark needs a sandboxed environment (Docker, etc.)

By implementing this interface, any cube can be run by any agent harness that supports the CUBE standard.
