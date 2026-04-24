# waa-cube

[WindowsAgentArena](https://github.com/microsoft/WindowsAgentArena) benchmark ported to the [CUBE](../../) protocol.

**152 tasks** across 10 domains (file explorer, VS Code, LibreOffice, VLC, Notepad, Paint, Settings, Clock, Calculator) running on a real Windows 11 VM via CUBE InfraConfig backends.

## Prerequisites

### System requirements

- QEMU + KVM (`apt install qemu-system-x86 qemu-utils ovmf swtpm`)
- `/dev/kvm` accessible (Windows 11 is effectively unusable without hardware virt)
- ~60 GB free disk space per Windows image copy (backup, overlay, flatten, etc.)
- 8 GB RAM allocated to the VM
- 8+ CPU cores recommended for the VM

For the image build pipeline specifically: add `packer` (install from HashiCorp
releases or via apt) and a write-capable HuggingFace account if you plan to
re-publish a prepared image.

## Using the pre-built image (fast path)

A pre-built, ready-to-boot Windows 11 image is hosted on HuggingFace at
`kushasareen/waa-windows-image/waa-windows-prepared.qcow2`. It ships with
OpenSSH Server, Azure VM Agent, SSH authorized_keys, and a working AutoAdminLogon
config already baked in. The `source_url` on [WAA_WINDOWS_RESOURCE](src/waa_cube/azure.py)
points at it directly.

You do **not** need to build the image yourself unless you want to:

- Use a different admin password or SSH key
- Rebuild on top of a fresher upstream WAA image
- Customize the guest environment (extra apps, drivers, etc.)

If you just want to run evals, skip to [Installation](#installation) and then
[Running on Azure](#running-on-azure).

## Building the Windows image from scratch

This takes a blank Windows 11 Enterprise Evaluation ISO through to a deployed,
SSH-reachable qcow2. Every stage is scripted; the whole chain takes ~2-4 hours
end-to-end on a typical Azure VM or beefy workstation.

### Stage 1 — Windows 11 ISO

The Microsoft Evaluation ISO can't be auto-downloaded for licensing reasons.

1. Get [Windows 11 Enterprise Evaluation](https://www.microsoft.com/en-us/evalcenter/evaluate-windows-11-enterprise) (free, registration required).
2. Save it locally as e.g. `~/Win11_Eval.iso`.

### Stage 2 — initial qcow2 from upstream WAA

The upstream WAA repo builds a Windows disk image by booting the ISO under a
Docker container that wraps QEMU. It installs Windows automatically via an
unattend.xml, then runs `setup.ps1` to install the app suite (Python 3.10,
Chrome, Edge, Thunderbird, GIMP, VLC, 7zip, ffmpeg, Git, Caddy proxy) and
register the WAA Flask agent as a scheduled task.

```bash
git clone https://github.com/microsoft/WindowsAgentArena ~/WindowsAgentArena
cd ~/WindowsAgentArena/src/win-arena-container
export WAA_SETUP_ISO=~/Win11_Eval.iso
./scripts/build-container-image.sh
./scripts/run-local.sh --prepare-image true    # ~60-90 min the first time
# image ends up at ~/.cube/waa/storage/data.img
```

Copy the result to the location waa-cube expects for the next stage:

```bash
mkdir -p ~/.cube/images ~/.cube/images/backups
cp ~/.cube/waa/storage/data.img ~/.cube/images/waa-windows-vm.qcow2
cp ~/.cube/images/waa-windows-vm.qcow2 \
   ~/.cube/images/backups/waa-windows-vm.qcow2.bak-$(date +%Y-%m-%d)
```

The backup is **mandatory** for the next stage — `bootstrap_winrm.py` refuses to
run without one because it modifies the base image in place.

### Stage 3 — bootstrap WinRM into the base image

The upstream image ships with no WinRM server and an empty Docker password.
`bootstrap_winrm.py` fixes both without VNC or any interactive step:

```bash
export WAA_BUILD_ADMIN_PASSWORD="$(openssl rand -base64 24 | tr -d '/+=' | head -c24)Aa3!"
# Azure password rules: 12-72 chars, 3 of {lower, upper, digit, special}
echo "$WAA_BUILD_ADMIN_PASSWORD" > ~/.cube/waa-build-admin-password.txt
chmod 600 ~/.cube/waa-build-admin-password.txt

make bootstrap-winrm
```

What it does: boots the base qcow2 under QEMU with an overlay, waits for the
WAA Flask agent at port 5000 to respond, POSTs a PowerShell payload to
`/setup/execute` that sets the Docker password + enables WinRM + opens TCP:5985
+ flips the network profile to Private, requests a clean shutdown, and then
`qemu-img commit`s the overlay into the base. ~5-10 min wall-time.

### Stage 4 — Packer build

```bash
# Generate a build-time SSH keypair (the pubkey gets baked into the image)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

make build-image
```

Under the hood `packer/run.sh` spawns `swtpm` for the vTPM socket, copies
OVMF_VARS to a writable per-build location, then invokes `packer build` with
five provisioners:

1. `install-openssh-server.ps1` — side-loads Win32-OpenSSH v9.5 from the GitHub
   release zip (bypasses the image's broken Windows Update).
2. `install-azure-vm-agent.ps1` — downloads and installs WindowsAzureVmAgent.msi
   so Azure's os_profile handshake works at launch time.
3. `drop-authorized-keys.ps1` — places your ed25519 pubkey at
   `C:\ProgramData\ssh\administrators_authorized_keys` with the strict ACLs
   OpenSSH requires (Administrators:F + SYSTEM:F, no inheritance).
4. `configure-autologon.ps1` — overwrites the LSA `DefaultPassword` secret via
   `LsaStorePrivateData` P/Invoke so AutoAdminLogon actually fires on boot with
   the new Docker password. This is the hard-to-spot step — without it, the
   image boots but no one logs in and the WAA Flask agent never starts.
5. *(sysprep.ps1 and embed-waa-server.ps1 are in the tree for reference but
   deliberately not invoked — see [AZURE_FIRST_RUN.md](AZURE_FIRST_RUN.md)
   for the rationale.)*

Output: `packer/output-waa-prepared/waa-windows-prepared.qcow2` — a
~2 GB overlay qcow2 (still references the base as backing file).

~20 min wall-time on an 8-core host.

### Stage 5 — smoke test locally

Before trusting the image on Azure, confirm both endpoints come up under plain
QEMU:

```bash
uv run python packer/smoke_test.py
# Expected:
#   [smoke] GUEST AGENT: UP (~640KB PNG)
#   [smoke] SSH: UP + authenticated as Docker
```

If either fails, fix before uploading — see [AZURE_FIRST_RUN.md](AZURE_FIRST_RUN.md)
debugging section.

### Stage 6 — flatten + upload

The overlay references the base image as its backing file — that won't work
when someone else downloads just the overlay. Flatten first:

```bash
qemu-img convert -O qcow2 \
  packer/output-waa-prepared/waa-windows-prepared.qcow2 \
  ~/.cube/hf-staging/waa-windows-prepared.qcow2
```

Upload to HuggingFace (or any other HTTP-accessible location):

```bash
uv run --with huggingface_hub --with hf_transfer python scripts/upload_image.py \
  --image-path ~/.cube/hf-staging/waa-windows-prepared.qcow2 \
  --repo-id <your-hf-user>/waa-windows-image \
  --filename waa-windows-prepared.qcow2
```

Finally update [WAA_WINDOWS_RESOURCE](src/waa_cube/azure.py) `source_url` to
point at your upload, and commit.

## Installation

```bash
uv pip install -e .
```

## Running evals

### Locally (LocalInfraConfig)

```bash
# Sequential debug run
uv run recipes/waa/haiku.py debug

# Full eval
uv run recipes/waa/haiku.py
```

Requires the prepared image present locally at
`~/.cube/images/waa-windows-vm.qcow2` OR a `source_url` LocalInfraConfig can
download from.

### Running on Azure

```bash
export WAA_WINDOWS_ADMIN_PASSWORD="$(cat ~/.cube/waa-build-admin-password.txt)"
az login
uv run recipes/waa/eval_azure_waa_kusha.py debug
```

See [AZURE_FIRST_RUN.md](AZURE_FIRST_RUN.md) for the full first-run walkthrough,
including the design constraints and failure modes baked into the image build.

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

Shipped as `task_metadata.json`, auto-loaded at import time. Regenerate from
the WAA repo:

```bash
python scripts/create_task_metadata.py --eval-dir /path/to/evaluation_examples_windows --force
```

### Filtering by domain

```python
bench.setup()
vscode_bench = bench.subset_from_glob("extra_info.domain", "vscode")
```

Domains: `file_explorer`, `libreoffice_calc`, `libreoffice_writer`, `vs_code`,
`vlc`, `settings`, `clock`, `notepad`, `microsoft_paint`, `windows_calc`.

Currently excluded: `chrome`, `msedge` (CDP timing issue).

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

Click center: `cx = x + w//2`, `cy = y + h//2`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WAA_BUILD_ADMIN_PASSWORD` | *(required for image build)* | Password used by `bootstrap_winrm.py` and `make build-image`. Must meet Azure complexity rules (12-72 chars, 3 of 4 character classes). |
| `WAA_WINDOWS_ADMIN_PASSWORD` | *(required for Azure launch)* | Same value as above, re-exported for the Azure recipe. |
| `AZURE_RESOURCE_GROUP` | `ui_assist` | Resource group containing the compute gallery + bootstrap storage. |
| `AZURE_STORAGE_ACCOUNT` | `cubeexpvhd` | Storage account for VHD blobs during provisioning. |
| `WAA_SETUP_ISO` | *(only for Stage 2 above)* | Windows 11 Enterprise ISO path, consumed by upstream WAA's image-build container. |

## Debug / Testing

```bash
cube test waa-cube
```

Runs 2 deterministic debug tasks without an LLM:

- **waa-debug-notepad** — opens Notepad via Win+R, evaluator checks the window title
- **waa-debug-infeasible** — calls `fail()` for a nonexistent app

## Known Issues

- **Chrome/Edge tasks excluded**: Chrome DevTools Protocol (CDP) setup has a
  timing issue — Playwright gets a 502 connecting to the CDP port before
  Chrome is fully ready.
- **Accessibility tree walk can hang**: pywinauto UIA can take 5+ minutes on
  certain desktop states. Not a provisioning issue; in-guest bug.

## Related docs

- [AZURE_FIRST_RUN.md](AZURE_FIRST_RUN.md) — hard constraints, first-run
  handoff, design journal, uncertainty callouts, full Packer build history.
- [packer/README.md](packer/README.md) — deep dive on the Packer config and
  its wrapper script.

## Package Structure

```
src/waa_cube/
├── __init__.py              # Public exports
├── azure.py                 # WAA_WINDOWS_RESOURCE definition
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
    ├── getters/             # Per-domain state extractors
    └── metrics/             # Per-domain evaluation metrics

packer/
├── waa-windows.pkr.hcl      # Packer qemu builder config
├── run.sh                   # swtpm + OVMF_VARS wrapper around packer build
├── bootstrap_winrm.py       # One-time WinRM enablement via guest agent
├── smoke_test.py            # Local QEMU end-to-end probe
└── scripts/                 # PowerShell provisioners invoked by Packer

scripts/
├── create_task_metadata.py  # Regenerate task_metadata.json from WAA eval dir
└── upload_image.py          # Push built qcow2 to HuggingFace
```
