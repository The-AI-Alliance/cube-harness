# osworld-cube

[OSWorld](https://os-world.github.io/) benchmark ported to the [CUBE](../../) protocol.

## Prerequisites

`osworld-cube` relies on [desktop_env](https://github.com/xlang-ai/OSWorld) to control desktop VMs. The OSWorld repository is cloned automatically on first `setup()` to:

```
$CUBE_CACHE_DIR/OSWorld        # default: ~/.agentlab2/OSWorld
```

You can override this location with the `OSWORLD_REPO` environment variable.

**Before running any task**, follow the [OSWorld Setup Guide](https://github.com/xlang-ai/OSWorld/blob/main/SETUP_GUIDELINE.md) to install the required system dependencies for your chosen provider (Docker, VMware, etc.). In particular:

- **Docker** (recommended): install Docker. VM images are downloaded automatically by `desktop_env` on first use.
- **VMware / VirtualBox**: install the hypervisor and follow the VM import steps in the guide.

## Overview

`osworld-cube` wraps OSWorld desktop-automation tasks as CUBE-compliant `Task` and `Tool` objects. Agents interact with real VM/container desktops through a unified interface, choosing between two action spaces.

## Installation

```bash
uv pip install -e .
```

## Usage

### Direct task loop

```python
from osworld_cube import OSWorldTask, ComputerConfig
from cube.task import TaskMetadata

task = OSWorldTask(
    metadata=TaskMetadata(
        id="task-uuid",
        abstract_description="Open the calculator app",
        extra_info={
            "domain": "os",
            "snapshot": "init_state",
            "config": [],
            "evaluator": {},
            "related_apps": [],
        },
    ),
    tool_config=ComputerConfig(provider="docker"),
)

obs, info = task.reset()
done = False
while not done:
    action = agent(obs, task.action_set)
    env_out = task.step(action)
    obs, done = env_out.obs, env_out.done
task.close()
```

### Via benchmark (full evaluation run)

```python
from osworld_cube import OSWorldBenchmark, ComputerConfig

bench = OSWorldBenchmark(
    default_tool_config=ComputerConfig(provider="docker"),
)
bench.setup()
for task_config in bench.get_task_configs():
    task = task_config.make()
    obs, info = task.reset()
    # ... agent loop ...
    task.close()
```

## Action Spaces

| Name | Config | Description |
|------|--------|-------------|
| `computer_13` (default) | `ComputerConfig(action_space="computer_13")` | 13 mouse/keyboard primitives (click, drag, scroll, type, hotkey, …) |
| `pyautogui` | `ComputerConfig(action_space="pyautogui")` | Single `run_pyautogui(code)` action — agent writes Python using `tag_N` SoM variables |

## Observations

Each step returns a multimodal `Observation` with:
- **screenshot** — PIL `Image` of the current desktop
- **accessibility_tree** — XML document (optionally post-processed)
- **terminal** — last terminal output (if enabled)

Set `use_som=True` on `OSWorldTask` / `OSWorldBenchmark` to annotate the screenshot with numbered bounding boxes and replace the raw XML with an indexed element table (Set-of-Marks).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUBE_CACHE_DIR` | `~/.agentlab2` | Root directory for VMs and cache |
| `OSWORLD_REPO` | `$CUBE_CACHE_DIR/benchmarks/osworld/OSWorld` | Path to OSWorld repo |

## Debug / Testing

A deterministic `DebugAgent` replays hardcoded action sequences without an LLM:

```python
from osworld_cube import get_debug_task_configs, make_debug_agent

for config in get_debug_task_configs():
    task = config.make()
    agent = make_debug_agent(config.task_id)
    obs, _ = task.reset()
    done = False
    while not done:
        action = agent(obs, task.action_set)
        env_out = task.step(action)
        obs, done = env_out.obs, env_out.done
    task.close()
```

Or run directly:

```bash
python -m osworld_cube.debug
```

## Package Structure

```
src/osworld_cube/
├── __init__.py       # Public exports
├── computer.py       # ComputerBase, Computer13, PyAutoGUIComputer, ComputerConfig
├── task.py           # OSWorldTask
├── benchmark.py      # OSWorldBenchmark, OSWorldTaskConfig
├── axtree.py         # Accessibility tree parsing and Set-of-Marks annotation
└── debug.py          # get_debug_task_configs, make_debug_agent
```
