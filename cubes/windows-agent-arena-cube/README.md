# waa-cube

[WindowsAgentArena](https://github.com/microsoft/WindowsAgentArena) benchmark ported to the [CUBE](../../) protocol.

**152 tasks** across 10 domains (file explorer, VS Code, LibreOffice, VLC, Notepad, Paint, Settings, Clock, Calculator) running on a real Windows 11 VM via CUBE InfraConfig backends.

## Prerequisites

### System requirements

- Docker (required — WAA runs inside `windowsarena/winarena:latest`)
- `/dev/kvm` (strongly recommended — without KVM, Windows will be very slow)
- ~60 GB free disk space for the VM disk image
- 8 GB RAM allocated to the VM (configurable via `ram_size=`)
- 8+ CPU cores recommended (`cpu_cores=8`)

### Windows 11 ISO

The WAA container does **not** auto-download Windows for licensing reasons. You must provide a
**Windows 11 Enterprise Evaluation ISO** before the first run.

1. Download the ISO from the [Microsoft Evaluation Center](https://www.microsoft.com/en-us/evalcenter/evaluate-windows-11-enterprise)
   (free, requires a short registration form).
2. Tell your provisioning pipeline where to find it (for example via environment variables used by your chosen infra backend).

## Installation

```bash
uv pip install -e .
```

### First-time provisioning

`WAABenchmark` is infra-driven. Provisioning happens through the configured
`InfraConfig` during `setup()` (idempotent):

```python
from waa_cube.benchmark import WAABenchmark

bench = WAABenchmark()
bench.install()  # no-op
bench.setup()    # provisions resources via infra when needed
```

## Usage

### Running an eval

```bash
# Sequential (1 VM at a time, recommended for single machines)
uv run recipes/waa/haiku.py

# Debug mode (same tasks, sequential, for development)
uv run recipes/waa/haiku.py debug
```

### Direct task loop

```python
from waa_cube.benchmark import WAABenchmark
from waa_cube.computer import ComputerConfig

bench = WAABenchmark(
    default_tool_config=ComputerConfig(),
)
bench.install()
bench.setup()

# Exclude chrome/msedge (CDP timing issue)
keep = [tid for tid, m in bench.task_metadata.items()
        if m.extra_info.get("domain") not in ("chrome", "msedge")]
bench = bench.subset_from_list(keep)  # 122 tasks

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

### Task metadata

Task metadata is shipped as `task_metadata.json` and auto-loaded at import time.
To regenerate from the WAA repo:

```bash
python scripts/create_task_metadata.py --eval-dir /path/to/evaluation_examples_windows --force
```

### Filtering by domain

```python
bench.setup()
vscode_bench = bench.subset_from_glob("extra_info.domain", "vscode")
```

Available domains: `file_explorer`, `libreoffice_calc`, `libreoffice_writer`, `vs_code`, `vlc`, `settings`, `clock`, `notepad`, `microsoft_paint`, `windows_calc`

Excluded domains (not yet working): `chrome`, `msedge`

## Observations

Each step the agent receives:
1. A **screenshot** (1280x800 PNG)
2. An **element table** (linearized accessibility tree):

```
index  tag                name              text  x    y    w    h
1      shell_traywnd      Taskbar           ""    0    752  1280 48
2      togglebutton       Start             ""    396  752  45   48
3      cabinetwclass      Documents - ...   ""    250  86   800  600
```

Click center: `cx = x + w//2`, `cy = y + h//2`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WAA_VM_STORAGE` | `~/.cube/waa/storage` | Host directory for the Windows disk image |
| `WAA_SETUP_ISO` | *(not set)* | Path to the Windows 11 Enterprise Evaluation ISO |

## Debug / Testing

```bash
cube test waa-cube
```

Runs 2 deterministic debug tasks without an LLM:
- **waa-debug-notepad**: Opens Notepad via Win+R, evaluator checks window title (reward=1.0)
- **waa-debug-infeasible**: Calls `fail()` for a nonexistent app (reward=1.0)

## Known Issues

- **Chrome/Edge tasks excluded**: Chrome DevTools Protocol (CDP) setup has a timing issue — Playwright gets a 502 connecting to the CDP port before Chrome is fully ready.
- **QEMU display must start at 1920x1080**: The Windows accessibility API returns an empty tree when QEMU's initial display matches the snapshot resolution (1280x800). The resolution mismatch forces a display reinit on snapshot restore that wakes the UI Automation framework.

## Package Structure

```
src/waa_cube/
├── __init__.py              # Public exports
├── benchmark.py             # WAABenchmark, WAATaskConfig
├── task.py                  # WAATask
├── computer.py              # ComputerConfig wrapper
├── debug.py                 # DebugWAABenchmark, DebugAgent
├── debug_tasks.json         # Debug task definitions
├── debug_task_metadata.json # Debug task metadata (CUBE format)
├── task_metadata.json       # Shipped task metadata (152 tasks)
└── vm_backend/
    ├── backend.py           # Legacy local VM backend utilities
    ├── evaluator.py         # GuestAgentProxy, Evaluator
    ├── setup_controller.py  # Task setup step execution
    ├── getters/             # Per-domain state extractors (used by evaluator)
    └── metrics/             # Per-domain evaluation metrics
scripts/
    create_task_metadata.py  # Developer tool to regenerate task_metadata.json
```
