# waa-cube

[WindowsAgentArena](https://github.com/microsoft/WindowsAgentArena) benchmark ported to the [CUBE](../../) protocol.

## Prerequisites

### System requirements

- Docker (required — WAA runs inside `windowsarena/winarena:latest`)
- `/dev/kvm` (strongly recommended — without KVM, Windows will be very slow)
- ~60 GB free disk space for the VM disk image
- 8 GB RAM allocated to the VM (configurable via `ram_size=`)

### Windows 11 ISO

The WAA container does **not** auto-download Windows for licensing reasons. You must provide a
**Windows 11 Enterprise Evaluation ISO** before the first run.

1. Download the ISO from the [Microsoft Evaluation Center](https://www.microsoft.com/en-us/evalcenter/evaluate-windows-11-enterprise)
   (free, requires a short registration form).
2. Tell `waa-cube` where to find it — pick one:
   - Set the environment variable: `export WAA_SETUP_ISO=/path/to/Win11_Eval.iso`
   - Pass it directly: `WAADockerVMBackend(setup_iso_path="/path/to/Win11_Eval.iso")`

## Installation

```bash
uv pip install -e .
```

### First-time image preparation (`install()`)

Before any task can run, the Windows VM disk image must be built. This is a
one-time operation (~20 min) triggered automatically by `cube test waa-cube`
or by calling `bench.install()` explicitly:

```python
from waa_cube.benchmark import WAABenchmark
from waa_cube.vm_backend.backend import WAADockerVMBackend

bench = WAABenchmark(
    vm_backend=WAADockerVMBackend(setup_iso_path="/path/to/Win11_Eval.iso"),
)
bench.install()  # one-time: boots Windows, saves data.qcow2, exits
bench.setup()
```

Or via environment variables:

```bash
export WAA_SETUP_ISO=/path/to/Win11_Eval.iso
export WAA_VM_STORAGE=~/.cube/waa/storage   # optional, this is the default

cube test waa-cube   # triggers install() automatically on first run
```

`install()` is idempotent — it skips if `data.qcow2` already exists in the
storage directory.

## Usage

### Direct task loop

```python
from waa_cube.benchmark import WAABenchmark
from waa_cube.computer import ComputerConfig
from waa_cube.vm_backend.backend import WAADockerVMBackend

bench = WAABenchmark(
    default_tool_config=ComputerConfig(),
    vm_backend=WAADockerVMBackend(),
)
bench.setup()
for task_config in bench.get_task_configs():
    task = task_config.make()
    obs, info = task.reset()
    done = False
    while not done:
        action = agent(obs, task.action_set)
        env_out = task.step(action)
        obs, done = env_out.obs, env_out.done
    task.close()
```

### Filtering by domain

```python
bench = WAABenchmark(...)
bench.setup()
vscode_bench = bench.subset_from_glob("extra_info.domain", "vscode")
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WAA_VM_STORAGE` | `~/.cube/waa/storage` | Host directory for the Windows disk image (`data.qcow2`) |
| `WAA_SETUP_ISO` | *(not set)* | Path to the Windows 11 Enterprise Evaluation ISO, used during `install()` |
| `WAA_EVAL_EXAMPLES_DIR` | *(not set)* | Path to `evaluation_examples_windows/` from the WAA repo |

## Debug / Testing

A deterministic `DebugAgent` runs a minimal end-to-end episode without an LLM:

```bash
cube test waa-cube
```

Or in Python:

```python
from waa_cube.debug import get_debug_benchmark, make_debug_agent

bench = get_debug_benchmark()
bench.install()   # no-op if already installed
bench.setup()
for config in bench.get_task_configs():
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

## Package Structure

```
src/waa_cube/
├── __init__.py           # Public exports
├── computer.py           # ComputerConfig (re-exports from cube_computer_tool)
├── task.py               # WAATask
├── benchmark.py          # WAABenchmark, WAATaskConfig
├── debug.py              # get_debug_benchmark, make_debug_agent
└── vm_backend/           # Docker + QMP VM backend
    ├── __init__.py
    ├── backend.py         # WAADockerVMBackend, WAADockerManager, WAADockerVM
    ├── evaluator.py       # Task evaluation logic
    ├── setup_controller.py
    ├── getters/           # Per-app state extractors
    └── metrics/           # Per-app evaluation metrics
```
