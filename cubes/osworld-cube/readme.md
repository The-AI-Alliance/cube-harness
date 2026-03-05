# osworld-cube

[OSWorld](https://os-world.github.io/) benchmark ported to the [CUBE](../../) protocol.

## Prerequisites

`osworld-cube` relies on [desktop_env](https://github.com/xlang-ai/OSWorld) to control desktop VMs. The OSWorld repository is cloned automatically on first `setup()` into the CUBE cache directory (under `CUBE_CACHE_DIR`, default `~/.agentlab2`).

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
| `computer_13` (default) | `ComputerConfig(action_space="computer_13")` | 13 mouse/keyboard primitives: click, double\_click, right\_click, mouse\_down, mouse\_up, move\_to, drag\_to, scroll, typing, press, key\_down, key\_up, hotkey — plus shared wait/done/fail signals |
| `pyautogui` | `ComputerConfig(action_space="pyautogui")` | Single `run_pyautogui(code)` action — agent writes arbitrary Python using pyautogui; `tag_N` coordinate variables (SoM bounding-box centres) are automatically prepended when `use_som=True` |

## Observations

Each step returns a multimodal `Observation` with:
- **screenshot** — PIL `Image` of the current desktop
- **axtree_txt** — linearized accessibility tree as a tab-separated text table (default)
- **terminal** — last terminal output (only included when `ComputerConfig(require_terminal=True)`)

The raw XML accessibility tree from `desktop_env` is always post-processed before being returned to the agent.

Set `use_som=True` on `OSWorldTask` / `OSWorldBenchmark` to switch to Set-of-Marks mode: the screenshot is annotated with numbered bounding boxes, and the axtree is replaced with an indexed element table (`som_elements`). In `pyautogui` mode, `tag_N` coordinate variables (bounding-box centres) are automatically prepended to the agent's code.

## Screenshot

<!-- TODO: add a screenshot of an eval run once we have results -->

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUBE_CACHE_DIR` | `~/.agentlab2` | Root directory for VMs and cache |
| `OSWORLD_REPO` | *(derived from `CUBE_CACHE_DIR`)* | Path used to resolve `settings_file` paths in task configs — override if you have an existing OSWorld clone |

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
