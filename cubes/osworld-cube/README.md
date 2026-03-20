# osworld-cube

[OSWorld](https://os-world.github.io/) benchmark ported to the [CUBE](../../) protocol.

## Prerequisites

`osworld-cube` relies on [desktop_env](https://github.com/xlang-ai/OSWorld) to control desktop VMs. The OSWorld repository is cloned automatically on first `setup()` into the CUBE cache directory (under `CUBE_CACHE_DIR`, default `~/.cube`).

### Platform support

The OSWorld VM images are **x86_64 only**. Hardware acceleration requirements differ by platform:

| Platform | Support | Notes |
|----------|---------|-------|
| Linux x86_64 + KVM | ✅ Full | Recommended |
| macOS Intel (x86_64) | ✅ Full | HVF acceleration available for local QEMU; Docker Desktop exposes `/dev/kvm` to containers |
| macOS Apple Silicon (arm64) | ❌ Not supported | HVF does not accelerate x86_64 guests; pure software emulation is too slow |

**Before running any task**, follow the [OSWorld Setup Guide](https://github.com/xlang-ai/OSWorld/blob/main/SETUP_GUIDELINE.md) to install the required system dependencies for your chosen provider (Docker, VMware, etc.). In particular:

- **Docker** (recommended): install Docker. VM images are downloaded automatically by `desktop_env` on first use.
- **VMware / VirtualBox**: install the hypervisor and follow the VM import steps in the guide.
- **QEMU** (default provider): install QEMU — `qemu-img` must be on your `PATH` for VM overlay creation.

  ```bash
  # macOS (x86_64)
  brew install qemu
  # Ubuntu/Debian
  sudo apt install qemu-utils
  ```

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
from osworld_cube.vm_backend import OSWorldQEMUVMBackend
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
    tool_config=ComputerConfig(),
    vm_backend=OSWorldQEMUVMBackend(),
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
from osworld_cube.vm_backend import OSWorldQEMUVMBackend

bench = OSWorldBenchmark(
    default_tool_config=ComputerConfig(),
    vm_backend=OSWorldQEMUVMBackend(),
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

| Field | `ComputerConfig` flag | Default | Description |
|-------|-----------------------|---------|-------------|
| `screenshot` | *(always included)* | on | PIL `Image` of the current desktop |
| `axtree_txt` | `require_a11y_tree=True` | on | Linearized accessibility tree as a tab-separated text table |
| `terminal` | `require_terminal=True` | off | Last terminal output |

The observation is captured after every action unless `observe_after_action=False` is set (useful when the agent drives observation timing manually).

The raw XML accessibility tree from `desktop_env` is always post-processed before being returned to the agent.

**Example `axtree_txt`** (Ubuntu desktop, idle state):

```
tag             name                             text  class  description  position (top-left x&y)  size (w&h)
label           Home                                                        (1833, 1037)             (40, 17)
menu            System                           ""                         (1814, 0)                (106, 27)
push-button     Google Chrome                    ""                         (0, 33)                  (70, 64)
push-button     Thunderbird Mail                 ""                         (0, 101)                 (70, 64)
push-button     Visual Studio Code               ""                         (0, 169)                 (70, 64)
push-button     VLC media player                 ""                         (0, 237)                 (70, 64)
push-button     LibreOffice Writer               ""                         (0, 305)                 (70, 64)
push-button     LibreOffice Calc                 ""                         (0, 373)                 (70, 64)
push-button     LibreOffice Impress              ""                         (0, 441)                 (70, 64)
push-button     GNU Image Manipulation Program   ""                         (0, 509)                 (70, 64)
push-button     Files                            ""                         (0, 577)                 (70, 64)
push-button     Ubuntu Software                  ""                         (0, 645)                 (70, 64)
push-button     Help                             ""                         (0, 713)                 (70, 64)
push-button     Trash                            ""                         (0, 784)                 (70, 64)
toggle-button   Show Applications                ""                         (0, 1010)                (70, 70)
```

Set `use_som=True` on `OSWorldTask` / `OSWorldBenchmark` to switch to Set-of-Marks mode: the screenshot is annotated with numbered bounding boxes, and the axtree is replaced with an indexed element table (`som_elements`). In `pyautogui` mode, `tag_N` coordinate variables (bounding-box centres) are automatically prepended to the agent's code.

## Screenshot

<!-- TODO: add a screenshot of an eval run once we have results -->

## Reproducibility

| Item | Value |
|------|-------|
| OSWorld repo | [`xlang-ai/OSWorld`](https://github.com/xlang-ai/OSWorld) |
| Pinned commit | `e695a10` |
| Task suite version | `1.0.0` (369 tasks) |
| VM image | Ubuntu 22.04 |
| Task index files | `test_all.json`, `test_small.json`, `test_nogdrive.json`, `test_infeasible.json` |

The OSWorld repo is cloned once and pinned to commit `e695a10`. To use a different commit or a pre-existing clone, set `OSWORLD_REPO` to point at it (see [Environment Variables](#environment-variables)).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUBE_CACHE_DIR` | `~/.cube` | Root directory for VMs and cache |
| `OSWORLD_REPO` | *(derived from `CUBE_CACHE_DIR`)* | Path used to resolve `settings_file` paths in task configs — override if you have an existing OSWorld clone |
| `PROXY_CONFIG_FILE` | *(not set)* | Path to proxy config JSON for OSWorld network routing (e.g. `dataimpulse.json`) |

### Example `.env`

```bash
# Root cache directory for VM images and cloned repos (default: ~/.cube)
# CUBE_CACHE_DIR=~/.cube

# Override if you already have an OSWorld clone somewhere
# OSWORLD_REPO=~/.cube/benchmarks/osworld/OSWorld

# Proxy config for OSWorld network routing (required for some tasks/providers)
# PROXY_CONFIG_FILE=~/.cube/benchmarks/osworld/OSWorld/evaluation_examples/settings/proxy/dataimpulse.json

# LLM API key (whichever provider you use — passed through to LiteLLM)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# AZURE_API_KEY=...
# AZURE_API_BASE=https://your-resource.openai.azure.com/
# AZURE_API_VERSION=2024-02-01
```

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
├── computer.py       # Re-exports from cube_computer_tool; sets osworld-cube cache default
├── task.py           # OSWorldTask
├── benchmark.py      # OSWorldBenchmark, OSWorldTaskConfig
├── axtree.py         # Accessibility tree parsing and Set-of-Marks annotation
├── debug.py          # get_debug_task_configs, make_debug_agent
└── vm_backend/       # QEMU VM backend (auto-downloads OSWorld images from HuggingFace)
    ├── __init__.py   # OSWorldQEMUVMBackend, ensure_base_image
    ├── evaluator.py  # Task evaluation logic
    ├── setup_controller.py  # VM setup/teardown
    ├── pyautogui_utils.py   # PyAutoGUI helpers
    ├── getters/      # Per-app state extractors (chrome, calc, file, gimp, …)
    └── metrics/      # Per-app evaluation metrics (basic_os, chrome, docs, …)
```
