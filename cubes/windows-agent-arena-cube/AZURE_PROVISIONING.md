# WAA-Cube Azure Provisioning — Design Document

> **Status**: Proposal (not yet implemented)
> **Author**: Auto-generated from codebase analysis
> **Date**: 2026-04-15
> **Reference**: OSWorld Azure provisioning in `cube-infra-azure`

## Overview

This document outlines how to run WAA (WindowsAgentArena) tasks on Azure VMs, following the same `InfraConfig` abstraction that OSWorld uses. The goal is to allow `WAABenchmark` to swap between local Docker+QEMU and Azure cloud VMs transparently.

## Current Architecture (Local Docker)

```
Host Machine
└── Docker container (windowsarena/winarena:latest)
    └── QEMU VM (Windows 11)
        ├── Flask guest agent (:5000)
        ├── QMP socket (:7200)
        └── Named snapshots (init_state, file_explorer, vscode, ...)
```

- **Image source**: Docker Hub (`windowsarena/winarena:latest`)
- **Task isolation**: QMP `loadvm` restores a named snapshot (~5-10s)
- **Networking**: Docker port-forwards → QEMU hostfwd → Windows guest
- **Backend**: `WAADockerVMBackend` (tightly coupled to Docker+QEMU)

## OSWorld's Azure Architecture (Reference)

```
Azure
└── Compute Gallery Image (osworld-ubuntu-vm/1.0.0)
    └── Fresh VM per task (specialized, no cloud-init)
        ├── Flask guest agent (:5000)
        └── SSH tunnel (localhost:free_port → VM:5000)
```

- **Image source**: HuggingFace qcow2.zip → bootstrap VM converts to VHD → Gallery
- **Task isolation**: New VM per task from gallery image (no snapshots)
- **Networking**: Azure NIC + SSH tunnel to guest agent port
- **Snapshots**: Declared in task metadata but **ignored at runtime** — setup scripts bring the VM to the right state

### Key Insight

OSWorld's `restore_snapshot(name)` accepts but **ignores** the snapshot name on all cloud backends. Each task starts from the base image. This is the same approach WAA should take.

## Proposed WAA Azure Architecture

```
Azure
└── Compute Gallery Image (waa-windows-vm/1.0.0)
    └── Fresh VM per task (specialized)
        ├── Flask guest agent (:5000)
        ├── OpenSSH server (:22)
        └── SSH tunnel (localhost:free_port → VM:5000)
```

### Resource Definition

```python
WAA_WINDOWS_RESOURCE = VMResourceConfig(
    name="waa-windows-vm",
    source_url="<huggingface or blob URL to Windows VHD>",
    scope="task",
    default_ttl_seconds=60 * 60 * 24,
    requires_kvm=False,  # Azure Hyper-V, not QEMU
)
```

## Implementation Plan

### Phase 1: Image Extraction and Upload

**Goal**: Get the WAA Windows disk image into a format Azure can boot.

1. **Extract disk from Docker image**:
   ```bash
   # Run the WAA container, copy the disk image out
   docker create --name waa-extract windowsarena/winarena:latest
   docker cp waa-extract:/storage/data.img ./waa-windows.raw
   docker rm waa-extract
   ```

2. **Convert raw → fixed VHD**:
   ```bash
   qemu-img convert -f raw -O vpc -o subformat=fixed,force_size \
       waa-windows.raw waa-windows.vhd
   ```

3. **Upload to HuggingFace or Azure Blob Storage** as the canonical `source_url`.

4. **Bootstrap VM conversion** (reuse OSWorld's pipeline):
   - The existing Azure bootstrap script downloads, converts, and injects SSH
   - For Windows, the SSH injection step needs modification (see Phase 2)

### Phase 2: Windows SSH Access

**Problem**: OSWorld's bootstrap script loop-mounts the VHD and injects SSH keys into Linux paths (`/root/.ssh/`, `/home/user/.ssh/`). Windows uses different paths and may not have OpenSSH pre-installed.

**Options** (pick one):

| Option | Effort | Notes |
|--------|--------|-------|
| **A. Pre-bake OpenSSH into the WAA image** | Low | Add OpenSSH Server + authorized_keys during WAA image build. No bootstrap changes needed. |
| **B. Use Azure VM extensions** | Medium | Azure can install OpenSSH via `VMAccessAgent` extension post-launch. Adds ~1 min to launch time. |
| **C. Skip SSH, use Azure serial console** | Medium | Use `run_command` API instead of SSH tunnel. Slower but no SSH setup needed. |
| **D. Modify bootstrap script for Windows** | High | NTFS mount + Windows registry edits to enable OpenSSH. Fragile. |

**Recommendation**: Option A — modify the WAA Docker image build to include OpenSSH Server with a pre-configured authorized_keys. This is a one-time change and keeps the Azure provisioning pipeline identical to OSWorld's.

### Phase 3: Snapshot-Free Task Setup

**Problem**: WAA tasks currently rely on named QEMU snapshots (`file_explorer`, `vscode`, `libreoffice_calc`, etc.) where apps are pre-opened in specific states. Azure VMs boot from the base image without snapshots.

**Current WAA task flow**:
```
QMP loadvm "file_explorer" → run setup scripts → agent starts
```

**Proposed Azure flow**:
```
Launch fresh VM → run expanded setup scripts → agent starts
```

**What needs to change**:

Each snapshot represents a pre-configured Windows state. The setup scripts need to be expanded to recreate that state from the base image:

| Snapshot | Current State | Setup Script Addition |
|----------|--------------|----------------------|
| `init_state` | Clean Windows desktop | None needed (base image) |
| `file_explorer` | File Explorer open | Launch explorer.exe, navigate to target folder |
| `vscode` | VS Code open with workspace | Launch VS Code, open workspace |
| `libreoffice_calc` | LibreOffice Calc open with file | Launch soffice --calc, open file |
| `notepad` | Notepad open | Launch notepad.exe |
| `vlc` | VLC open | Launch vlc.exe |
| ... | ... | ... |

The setup scripts already support actions like `launch`, `execute`, `create_file`, etc. via `SetupController`. Adding app-launch steps for each snapshot state is straightforward.

**Implementation**: Add a `snapshot_setup_scripts` mapping that runs when `infra` is Azure/AWS:

```python
# In WAATask._setup_task():
if not self._uses_qemu_snapshots():
    # Azure/AWS path: run snapshot-equivalent setup scripts
    snapshot_setup = SNAPSHOT_SETUP_SCRIPTS.get(snapshot_name, [])
    setup_ctrl.setup(snapshot_setup + task_setup_steps)
else:
    # Local QEMU path: restore snapshot, then run task setup
    self._vm.restore_snapshot(snapshot_name)
    setup_ctrl.setup(task_setup_steps)
```

### Phase 4: Integration with InfraConfig

**Add `infra` field to WAABenchmark and WAATask** (like OSWorld):

```python
class WAABenchmark(Benchmark):
    infra: InfraConfig | None = None      # New: Azure/Local/AWS
    vm_backend: VMBackend | None = None   # Existing: Docker (backward compat)

class WAATask(Task):
    infra: InfraConfig | None = None
    vm_backend: VMBackend | None = None
```

**Task reset() dispatches based on which is set**:
- `infra` set → `infra.launch(WAA_WINDOWS_RESOURCE)` → SSH tunnel → HTTP endpoint
- `vm_backend` set → `WAADockerVMBackend.launch()` → Docker → QEMU → snapshot

**Recipe usage**:
```python
# Local (existing)
bench = WAABenchmark(vm_backend=WAADockerVMBackend(cpu_cores=8))

# Azure (new)
from cube_infra_azure import AzureInfraConfig
bench = WAABenchmark(infra=AzureInfraConfig(resource_group="waa-eval"))
```

## Guest Agent Compatibility

The Flask guest agent (`get_screenshot`, `get_accessibility_tree`, `execute_python_command`, etc.) communicates over HTTP. It works identically regardless of infrastructure:

- **Local**: `http://localhost:{docker_port}` → Docker → QEMU hostfwd → Windows :5000
- **Azure**: `http://localhost:{ssh_tunnel_port}` → SSH tunnel → Windows :5000

No changes to the guest agent, getters, or metrics are needed.

## Estimated Effort

| Phase | Description | Effort | Blocking? |
|-------|-------------|--------|-----------|
| 1 | Image extraction + VHD upload | 1-2 days | Yes |
| 2 | Windows SSH access | 1 day (Option A) | Yes |
| 3 | Snapshot-free setup scripts | 3-5 days | Yes (most work) |
| 4 | InfraConfig integration | 1-2 days | No (can merge independently) |

**Total**: ~1-2 weeks

## Risks

1. **Windows image size**: WAA's Windows image is ~60 GB. Azure blob upload and gallery import will be slow (~30-90 min provision time). Acceptable since it's one-time.

2. **Windows boot time**: Windows VMs take longer to boot than Linux (~2-3 min vs ~30s). Each task will have higher overhead.

3. **Accessibility tree on Azure Hyper-V**: The QEMU display resolution trick (start at 1920x1080 to force a display reinit) may not apply on Hyper-V. Needs testing.

4. **Setup script fidelity**: Reproducing snapshot states via setup scripts may not be 100% identical to the original snapshots (e.g., window positions, app state). Some tasks may break and need per-task debugging.

5. **Licensing**: Windows 11 on Azure requires appropriate licensing. Azure provides Windows Server images, but desktop Windows 11 may need BYOL (Bring Your Own License) or Azure Virtual Desktop licensing.
