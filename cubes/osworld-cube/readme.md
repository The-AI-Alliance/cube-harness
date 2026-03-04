# osworld-cube

[OSWorld](https://os-world.github.io/) benchmark ported to the [CUBE](../../) protocol.
#TODO: recheck this
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
while not task.tool._is_done:
    action = agent(obs, task.action_set)
    env_out = task.step(action)
    obs = env_out.obs
task.close()
```

### Via benchmark (full evaluation run)

```python
from osworld_cube import OSWorldBenchmark, ComputerConfig

bench = OSWorldBenchmark(
    default_tool_config=ComputerConfig(provider="docker"),
    domain="chrome",   # or "all"
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
    while not task.tool._is_done:
        action = agent(obs, task.action_set)
        env_out = task.step(action)
        obs = env_out.obs
    task.close()
```

Or run directly:

```bash
python -m osworld_cube.debug_agent
```

## Package Structure

```
src/osworld_cube/
├── __init__.py       # Public exports
├── computer.py       # ComputerBase, Computer13, PyAutoGUIComputer, ComputerConfig
├── task.py           # OSWorldTask
├── benchmark.py      # OSWorldBenchmark, OSWorldTaskConfig
└── axtree.py         # Accessibility tree parsing and Set-of-Marks annotation
```
