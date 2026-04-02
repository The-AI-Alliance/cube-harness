# WAA Cube vs OSWorld Cube

## Purpose

This note captures findings from reading both `waa-cube` and `osworld-cube` in the same repository.

The goal is to answer two questions:

1. How does `waa-cube` work today?
2. Which implementation patterns from `osworld-cube` should guide future `waa-cube` work?

## Executive Summary

`osworld-cube` is the right implementation template for `waa-cube`.

The good news is that `waa-cube` already follows the `osworld-cube` shape in the upper layers:

- benchmark loads task metadata and produces task configs
- task orchestrates reset, setup, observation, and evaluation
- computer layer reuses `cube_computer_tool`
- evaluator and setup controller are ports of upstream logic behind `GuestAgent`

The biggest gap is the VM backend and reset semantics.

`osworld-cube` keeps infrastructure thin and composes with shared backend primitives. `waa-cube` currently carries much more bespoke infrastructure in `src/waa_cube/vm_backend/backend.py`, and several comments/docstrings still describe snapshot behavior that the live code no longer performs.

## How `osworld-cube` Is Structured

### 1. Benchmark owns install/setup and task loading

`osworld_cube/benchmark.py` is mostly responsible for:

- installing benchmark assets
- cloning a pinned upstream repo
- loading task JSON files into `TaskMetadata`
- generating `OSWorldTaskConfig` objects

Important references:

- `cube-harness/cubes/osworld-cube/src/osworld_cube/benchmark.py`
- `OSWORLD_COMMIT = "e695a10"`
- `OSWorldBenchmark.install()`
- `OSWorldBenchmark._load_task_metadata_from_repo()`
- `OSWorldBenchmark._fix_settings_paths()`

This is clean because the benchmark layer stays declarative and does not absorb VM runtime details.

### 2. Task is a thin runtime orchestrator

`osworld_cube/task.py` does a small number of things:

- lazily launches a VM
- restores the task snapshot
- runs setup steps through `SetupController`
- gets an observation from the shared computer tool
- evaluates with `Evaluator`
- stops the VM on `close()`

Important references:

- `cube-harness/cubes/osworld-cube/src/osworld_cube/task.py`
- `_ensure_vm()`
- `_setup_task()`
- `_evaluate_task()`
- `reset()`
- `close()`

This layer is easy to reason about because it orchestrates other pieces instead of hiding behavior inside itself.

### 3. Computer layer is shared, not custom

`osworld_cube/computer.py` is intentionally tiny. It just re-exports the shared `cube_computer_tool` implementation and sets a sensible cache root.

Important references:

- `cube-harness/cubes/osworld-cube/src/osworld_cube/computer.py`
- `cube-standard/cube-tools/cube-computer-tool/src/cube_computer_tool/computer.py`

This is a strong pattern worth preserving in `waa-cube`.

### 4. Evaluator and setup controller are clean ports

`osworld-cube` ports upstream logic into:

- `vm_backend/evaluator.py`
- `vm_backend/setup_controller.py`
- `vm_backend/getters/`
- `vm_backend/metrics/`

The key architectural move is the `GuestAgentProxy`, which adapts the CUBE guest agent to the interface expected by the ported getter/metric code.

Important references:

- `cube-harness/cubes/osworld-cube/src/osworld_cube/vm_backend/evaluator.py`
- `cube-harness/cubes/osworld-cube/src/osworld_cube/vm_backend/setup_controller.py`

This keeps benchmark logic separate from VM transport details.

### 5. VM backend is thin and compositional

`osworld-cube` uses shared `cube_vm_backend` primitives and mainly adds OSWorld-specific image acquisition.

Important references:

- `cube-harness/cubes/osworld-cube/src/osworld_cube/vm_backend/__init__.py`
- `ensure_base_image()`
- `OSWorldQEMUVMBackend`
- `OSWorldDockerVMBackend`

This is one of the cleanest parts of the design. The benchmark package adds only what is specific to OSWorld and delegates generic VM lifecycle mechanics to reusable backend code.

## How `waa-cube` Works Today

### 1. Benchmark loads WAA task metadata

`WAABenchmark` loads task metadata either from:

- a flat `tasks_file` used by debug mode
- a WindowsAgentArena `evaluation_examples_windows` directory provided via constructor or `WAA_EVAL_EXAMPLES_DIR`

Important references:

- `cube-harness/cubes/windows-agent-arena-cube/src/waa_cube/benchmark.py`
- `WAATaskConfig`
- `WAABenchmark.get_task_configs()`
- `WAABenchmark._load_task_metadata()`
- `WAABenchmark._load_task_metadata_from_file()`

### 2. Task orchestration already looks very similar to OSWorld

`WAATask` is structurally very close to `OSWorldTask`:

- it lazily launches a VM via `vm_backend`
- restores task state before reset
- runs setup steps through `SetupController`
- evaluates through `Evaluator`
- closes the VM per task

Important references:

- `cube-harness/cubes/windows-agent-arena-cube/src/waa_cube/task.py`

This is already aligned with the OSWorld model and should remain so.

### 3. Computer layer is also shared

`waa_cube/computer.py` matches the OSWorld approach: it re-exports the shared computer tool and only overrides the cache directory default.

Important reference:

- `cube-harness/cubes/windows-agent-arena-cube/src/waa_cube/computer.py`

### 4. Evaluator and setup controller are already in the OSWorld style

`waa-cube` ports WAA getters/metrics/setup logic behind a `GuestAgent` boundary in nearly the same way as `osworld-cube`.

Important references:

- `cube-harness/cubes/windows-agent-arena-cube/src/waa_cube/vm_backend/evaluator.py`
- `cube-harness/cubes/windows-agent-arena-cube/src/waa_cube/vm_backend/setup_controller.py`

This is good alignment and should be treated as the stable shape of the design.

### 5. VM backend is much heavier than OSWorld's

`waa-cube` currently owns a lot of bespoke backend logic in-package:

- storage overlay management
- Docker image lifecycle
- port reservation
- container startup
- readiness polling
- DNAT/iptables proxy rules
- cleanup of stale overlays and containers
- Windows image installation

Important reference:

- `cube-harness/cubes/windows-agent-arena-cube/src/waa_cube/vm_backend/backend.py`

This file is the main place where `waa-cube` diverges from the cleanliness of `osworld-cube`.

## Where `waa-cube` Already Matches the OSWorld Pattern

These are strengths, not problems:

### 1. Same high-level object model

Both cubes follow:

- benchmark -> task config -> task
- task -> setup controller + evaluator + shared computer tool

### 2. Same observation and action model

Both rely on:

- `cube_computer_tool`
- `GuestAgent`
- CUBE `Task.step()`
- agent-driven `done()` / `fail()` termination

### 3. Same ported evaluation model

Both cubes use:

- `GuestAgentProxy`
- ported `getters/`
- ported `metrics/`
- `postconfig` before evaluation

### 4. Same small computer shim

Both cubes keep benchmark-specific computer code minimal and defer to the shared tool package.

## Main Gaps Between `waa-cube` and the OSWorld Template

### 1. Backend complexity is concentrated in one custom file

The main divergence is `waa_cube/vm_backend/backend.py`.

Compared to OSWorld:

- more lifecycle logic lives in the benchmark package
- more networking details are benchmark-owned
- more cleanup and state-management logic is custom

This makes WAA harder to reason about and increases the chance that behavior drifts from comments and docs.

### 2. Reset semantics are inconsistent with comments and docs

Several WAA comments and docstrings still describe QMP snapshot restoration, but the active `restore_snapshot()` path currently does not perform `loadvm`.

The live code resets by calling:

- `POST /setup/close_all`

Important reference:

- `cube-harness/cubes/windows-agent-arena-cube/src/waa_cube/vm_backend/backend.py`

This matters because the rest of the package still talks about:

- named QEMU snapshots
- snapshot restore
- snapshot isolation

At the moment, the implementation behaves more like app-level reset than true VM snapshot restore.

### 3. Benchmark-owned reproducibility is weaker than OSWorld

`osworld-cube` owns upstream asset installation and pins the upstream repo commit.

`waa-cube` currently depends on:

- manual placement of a Windows ISO
- external task-data location via `WAA_EVAL_EXAMPLES_DIR`
- Docker image semantics that are described partly in docs and partly in code

This is workable, but less self-contained and less reproducible than the OSWorld model.

### 4. `use_som` is not propagated from benchmark to task

This issue exists in both cubes, but it is still worth calling out because it affects design clarity.

Symptoms:

- `OSWorldBenchmark.use_som` exists
- `WAABenchmark.use_som` exists
- `OSWorldTask.use_som` exists
- `WAATask.use_som` exists
- neither benchmark passes `use_som` through `get_task_configs()` / `TaskConfig.make()`

Result:

- benchmark-level `use_som=True` does not actually configure tasks in the normal path

This should be fixed in both places, and WAA should not copy the bug just because OSWorld has it too.

### 5. WAA benchmark install/source management is less benchmark-owned

OSWorld's benchmark owns:

- cloning
- pinning
- path fixing
- environment normalization

WAA benchmark currently owns less of this lifecycle and delegates more to external environment setup.

If the goal is to make WAA feel like OSWorld, the benchmark should own more of the "make this runnable" story.

## Concrete Recommendations for `waa-cube`

### Keep these parts aligned with OSWorld

Preserve the current structure of:

- `waa_cube/task.py`
- `waa_cube/computer.py`
- `waa_cube/vm_backend/evaluator.py`
- `waa_cube/vm_backend/setup_controller.py`

These already fit the OSWorld pattern well.

### Refactor the backend toward a thinner adapter

Move `waa-cube` closer to the OSWorld backend model:

- keep benchmark-specific logic only where WAA genuinely differs
- push generic VM/container mechanics into reusable backend primitives where possible
- reduce the amount of lifecycle/network/storage orchestration living in the benchmark package itself

The target should be:

- thin WAA-specific backend wrapper
- clear reset contract
- fewer infrastructure details leaking upward

### Make reset semantics explicit and truthful

Pick one of these and document it consistently:

1. True snapshot restore
2. App-level reset via `close_all`

Right now the package describes one thing and often does another.

This is the highest-value conceptual cleanup because it affects correctness, isolation expectations, and test strategy.

### Make WAA benchmark setup more reproducible

Adopt more of the OSWorld benchmark behavior:

- own task-source discovery more explicitly
- pin or standardize the source of `evaluation_examples_windows`
- keep path normalization and environment setup in the benchmark layer

### Fix `use_som` as part of the alignment work

If `use_som` remains a benchmark-level option, it should be passed through to each task config and task instance.

## Recommended Refactor Priorities

### Priority 1

Clarify reset model and update code/comments/docs to match.

### Priority 2

Thin `waa_cube/vm_backend/backend.py` by extracting reusable mechanics or aligning it with shared backend abstractions.

### Priority 3

Make benchmark installation and task-source handling more reproducible and benchmark-owned.

### Priority 4

Wire `use_som` through properly.

### Priority 5

Add comparison-style tests that lock in intended behavior, especially around:

- reset semantics
- task-source loading
- benchmark-level options flowing into tasks

## Files Worth Using as the Template

These are the highest-signal reference files when aligning WAA to OSWorld:

- `cube-harness/cubes/osworld-cube/src/osworld_cube/benchmark.py`
- `cube-harness/cubes/osworld-cube/src/osworld_cube/task.py`
- `cube-harness/cubes/osworld-cube/src/osworld_cube/computer.py`
- `cube-harness/cubes/osworld-cube/src/osworld_cube/vm_backend/__init__.py`
- `cube-harness/cubes/osworld-cube/src/osworld_cube/vm_backend/evaluator.py`
- `cube-harness/cubes/osworld-cube/src/osworld_cube/vm_backend/setup_controller.py`

These are the most important WAA files to compare against them:

- `cube-harness/cubes/windows-agent-arena-cube/src/waa_cube/benchmark.py`
- `cube-harness/cubes/windows-agent-arena-cube/src/waa_cube/task.py`
- `cube-harness/cubes/windows-agent-arena-cube/src/waa_cube/computer.py`
- `cube-harness/cubes/windows-agent-arena-cube/src/waa_cube/vm_backend/backend.py`
- `cube-harness/cubes/windows-agent-arena-cube/src/waa_cube/vm_backend/evaluator.py`
- `cube-harness/cubes/windows-agent-arena-cube/src/waa_cube/vm_backend/setup_controller.py`

## Bottom Line

`waa-cube` should be based on `osworld-cube` structurally.

The upper layers of `waa-cube` are already close.

The main work is to bring the backend and benchmark-owned lifecycle into the same style:

- thinner infrastructure layer
- clearer reset contract
- stronger reproducibility story
- consistent option propagation

That is the path that will make `waa-cube` easier to maintain and much easier to trust.
