# waa-cube

[WindowsAgentArena](https://github.com/microsoft/WindowsAgentArena) benchmark
ported to the [CUBE](../../) protocol.

**152 tasks** across 11 domains (file explorer, VS Code, LibreOffice
Calc/Writer, VLC, Notepad, Paint, Settings, Clock, Calculator, Chrome / MS
Edge) running on a real Windows 11 VM via CUBE InfraConfig backends.

## Prerequisites

### Local (QEMU)

- QEMU + KVM (`apt install qemu-system-x86 qemu-utils ovmf swtpm`)
- `/dev/kvm` accessible — Windows 11 is effectively unusable without hardware
  virtualisation
- ~60 GB free disk (image + overlay + backups)
- 8 GB RAM allocated to the VM, 8+ vCPUs recommended

### Azure

- An Azure subscription with quota for `Standard_D8s_v3` VMs (8 vCPU, 32 GB
  RAM each — n=10 parallel needs ~80 vCPU)
- Resource group containing: a compute gallery, a storage account (for source
  VHD blobs), a vnet/subnet wide enough for the eval cohort (a /24 = 251
  usable IPs is enough for normal use; if you launch 200+ VMs in quick
  succession, async cleanup leaves orphan NICs that can fill the subnet —
  drop the TTL or bulk-delete via `az network nic list ...`)
- `az login` / Azure CLI authenticated locally

For the image-build pipeline specifically: add `packer` (HashiCorp release or
`apt`) and a write-capable HuggingFace account if you plan to publish a new
prepared image.

## Using the pre-built image (fast path)

A pre-built, ready-to-boot Windows 11 image is hosted on HuggingFace at
`kushasareen/waa-windows-image/waa-windows-prepared.qcow2`. It ships with
OpenSSH Server, Azure VM Agent, SSH `authorized_keys`, and AutoAdminLogon
already baked in. The `source_url` on
[`WAA_WINDOWS_RESOURCE`](src/waa_cube/azure.py) points at it directly.

You do **not** need to build the image yourself unless you want to:

- Use a different admin password or SSH key
- Rebuild on top of a fresher upstream WAA image
- Customize the guest environment (extra apps, drivers, etc.)

If you just want to run evals, skip to [Installation](#installation) and then
[Running on Azure](#running-on-azure).

## Running evals

### Locally (LocalInfraConfig)

```bash
# Sequential debug run
uv run recipes/waa/haiku.py debug

# Full local eval (pulls the image to ~/.cube/images/ on first run)
uv run recipes/waa/haiku.py
```

`LocalInfraConfig` boots one QEMU VM at a time per Ray worker. The image lives
at `~/.cube/images/waa-windows-vm.qcow2`. You'll need a beefy laptop for any
n_cpus > 1 — each VM eats 8 GB RAM and ~50% of one core during agent steps.

### Running on Azure (AzureInfraConfig)

```bash
export WAA_WINDOWS_ADMIN_PASSWORD="$(cat ~/.cube/waa-build-admin-password.txt)"
az login
uv run recipes/waa/azure_haiku.py
```

See [How Azure provisioning works](#how-azure-provisioning-works) below for
what this is doing under the hood.

### Recipes

| File | Purpose |
|---|---|
| [`recipes/waa/eval_waa.py`](../../recipes/waa/eval_waa.py) | Local QEMU eval — Genny + GPT-5 + axtree, no screenshots / SoM. Fastest path to a smoke test. |
| [`recipes/waa/haiku.py`](../../recipes/waa/haiku.py) | Local QEMU eval — Genny + Claude Haiku 4.5 (multimodal: screenshot + axtree). |
| [`recipes/waa/azure_haiku.py`](../../recipes/waa/azure_haiku.py) | Azure full-corpus eval — Genny + Claude Haiku 4.5 on the LibreOffice-enabled image (`-kusha-lo`). |

To swap models, change `LLMConfig(model_name=...)`. To add SoM (numbered red
boxes on the screenshot, agent clicks via `tag_N` instead of pixel coords),
pass `use_som=True` to `WAABenchmark(...)` and update the system prompt to
teach the `tag_N` syntax.

### Direct task loop

```python
from cube import LocalInfraConfig
from waa_cube.benchmark import WAABenchmark
from waa_cube.computer import ComputerConfig

bench_config = WAABenchmark(tool_config=ComputerConfig(), infra=LocalInfraConfig())
benchmark = bench_config.make()  # provisions resources + sets up runtime

for task_config in bench_config.get_task_configs():
    task = task_config.make()
    obs, info = task.reset()
    done = False
    while not done:
        action = agent(obs, task.action_set)
        env_out = task.step(action)
        obs, done = env_out.obs, env_out.done
    task.close()

benchmark.close()
```

### Filtering by domain

Domain lives on `WAATaskExecutionInfo` (the heavy per-task data shipped in
`task_execution_info.json`), not on the slim `TaskMetadata`, so
`subset_from_glob` can't reach it directly. Filter via the shipped JSON:

```python
import json
from pathlib import Path
import waa_cube

exec_info = json.loads((Path(waa_cube.__file__).parent / "task_execution_info.json").read_text())
vscode_ids = [tid for tid in bench_config.task_metadata
              if exec_info[tid]["domain"] == "vs_code"]
vscode_bench = bench_config.subset_from_list(vscode_ids)
```

Domains: `file_explorer`, `libreoffice_calc`, `libreoffice_writer`, `vs_code`,
`vlc`, `settings`, `clock`, `notepad`, `microsoft_paint`, `windows_calc`,
`chrome`, `msedge`.

## How Azure provisioning works

`waa-cube` runs on Azure via the `AzureInfraConfig` backend from
`cube-infra-azure`. Recipes never touch Azure SDK / `az` directly — they hand
a declarative resource description (`WAA_WINDOWS_RESOURCE`) to an
`AzureInfraConfig` instance and call `infra.launch(resource)` per task.

### `WAA_WINDOWS_RESOURCE` (declared in [`src/waa_cube/azure.py`](src/waa_cube/azure.py))

```python
WAA_WINDOWS_RESOURCE = VMResourceConfig(
    name="waa-windows-vm",
    source_url="https://huggingface.co/.../waa-windows-prepared.qcow2",
    default_ttl_seconds=60 * 60 * 2,        # auto-cleanup after 2h
    min_cpu_cores=8, min_ram_gb=8,           # pins us to D8s_v3
    uefi=True, tpm=True, os_type="windows",
    specialized=True,                        # image is pre-syspreped
    forwarded_ports=[9222, 8080],            # CDP, VLC HTTP
)
```

`AzureInfraConfig` resolves this into the right `Standard_D8s_v3` VM size and
takes care of every Azure-side side-effect.

### Lifecycle

```
recipe          AzureInfraConfig          Azure
  │
  │ AzureInfraConfig(resource_group=…, vnet=…, …)
  │
  │ bench_config.make(infra=…)
  │  └─ infra.provision(WAA_WINDOWS_RESOURCE)  if not already
  │      │  one-time per resource_group, idempotent:
  │      ├─ download source_url → /data/source.download
  │      │   (uses aria2c -x 16 -s 16 inside a Standard_D4_v3 bootstrap VM
  │      │    so HF→Azure is ~100-250 MB/s instead of 5 MB/s via wget)
  │      ├─ qemu-img convert → vhd (Azure gallery format)
  │      ├─ azcopy upload → blob
  │      ├─ register as Compute Gallery image (waa-windows-vm-{suffix}/1.0.0)
  │      └─ delete bootstrap VM
  │
  │ run_with_ray(exp, n_cpus=10)
  │
  │ episode worker (×n_cpus, parallel, in Ray subprocesses):
  │   infra.launch(WAA_WINDOWS_RESOURCE) per task
  │      ├─ create network: NIC + public IP (cube-{run_id}-…)
  │      ├─ create VM from gallery image (specialized → no os_profile)
  │      ├─ inject SSH key + open NSG firewall via Azure RunCommand
  │      │   (we don't trust unattend.xml password — keys-only)
  │      ├─ wait_for_ssh on public_ip
  │      ├─ open SSH tunnels: localhost:{free}→VM:5000 (Flask),
  │      │                    localhost:{free}→VM:9222 (Chrome CDP),
  │      │                    localhost:{free}→VM:8080 (VLC)
  │      └─ return AzureResourceHandle (run_id, endpoints, _vm_name, _tunnels)
  │
  │ task.reset() → SetupController hits handle.endpoint to drive the VM
  │ ...agent loop...
  │ task.close() → handle.close() → INFRA._delete_vm(run_id)
  │
  │ exp_runner.finally:
  │   INFRA.cleanup_orphaned_resources()  # NICs / IPs whose VM died
```

### Quota footprint

- One D8s_v3 = 8 vCPU = 1 unit of `Standard DSv3 Family` quota.
- A run with `n_cpus=10` needs ~80 vCPU live at peak. Typical default quota is
  350 vCPU; budget ~270 for evals after the bootstrap VM and headroom.
- The bootstrap VM (used by `provision()` only) is `Standard_D4_v3` and is
  deleted after the gallery image is registered.

### Cleanup hygiene

- `default_ttl_seconds=2h` on `WAA_WINDOWS_RESOURCE` → every launched VM has
  an `expires_at` ARM tag. `INFRA.cleanup_stale()` deletes anything past its
  TTL.
- After a run ends normally, the recipe's `finally:` block calls
  `INFRA.cleanup_orphaned_resources()` (orphan NICs / public IPs left after a
  VM is gone but Azure's async delete hasn't finished).
- After a `kill -9` or crash mid-run, VMs sit until their 2 h TTL fires. To
  reclaim quota immediately:

  ```bash
  az vm list -g $RG --query "[?starts_with(name,'cube-') && \
       hardwareProfile.vmSize=='Standard_D8s_v3' && \
       !contains(name,'bootstrap')].name" -o tsv \
    | xargs -P 30 -I {} az vm delete -g $RG --name {} --yes --no-wait
  ```

## Building the Windows image from scratch

If you need to rebuild (different password, new app, fresher upstream WAA),
the chain is below. Whole thing takes ~2-4h end-to-end.

### Stage 1 — Windows 11 ISO

1. Get [Windows 11 Enterprise Evaluation](https://www.microsoft.com/en-us/evalcenter/evaluate-windows-11-enterprise) (free, registration required).
2. Save as e.g. `~/Win11_Eval.iso`. Microsoft's licence terms forbid
   automating this download.

### Stage 2 — initial qcow2 from upstream WAA

The upstream WAA repo builds a Windows disk image by booting the ISO under a
Docker container that wraps QEMU. It runs Windows install via unattend.xml,
then `setup.ps1` (Python 3.10, Chrome, Edge, Thunderbird, GIMP, VLC, 7zip,
ffmpeg, Git, Caddy proxy, the WAA Flask agent as a scheduled task).

```bash
git clone https://github.com/microsoft/WindowsAgentArena ~/WindowsAgentArena
cd ~/WindowsAgentArena/src/win-arena-container
export WAA_SETUP_ISO=~/Win11_Eval.iso
./scripts/build-container-image.sh
./scripts/run-local.sh --prepare-image true   # ~60-90 min the first time
# image lands at ~/.cube/waa/storage/data.img

mkdir -p ~/.cube/images ~/.cube/images/backups
cp ~/.cube/waa/storage/data.img ~/.cube/images/waa-windows-vm.qcow2
cp ~/.cube/images/waa-windows-vm.qcow2 \
   ~/.cube/images/backups/waa-windows-vm.qcow2.bak-$(date +%Y-%m-%d)
```

The backup is **mandatory** for the next stage — `bootstrap_winrm.py` refuses
to run without one because it modifies the base image in place.

### Stage 3 — bootstrap WinRM into the base image

Upstream image ships with no WinRM and an empty Docker password.
`bootstrap_winrm.py` fixes both without VNC:

```bash
export WAA_BUILD_ADMIN_PASSWORD="$(openssl rand -base64 24 | tr -d '/+=' | head -c24)Aa3!"
echo "$WAA_BUILD_ADMIN_PASSWORD" > ~/.cube/waa-build-admin-password.txt
chmod 600 ~/.cube/waa-build-admin-password.txt

make bootstrap-winrm
```

Boots the base qcow2 under QEMU with an overlay, waits for the WAA Flask
agent at port 5000, POSTs a PowerShell payload to `/setup/execute` that sets
the Docker password + enables WinRM + opens TCP:5985 + flips the network
profile to Private + clean shutdown, then `qemu-img commit`s the overlay
into the base. ~5-10 min.

### Stage 4 — Packer build

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
make build-image
```

`packer/run.sh` spawns `swtpm` for the vTPM socket, copies `OVMF_VARS` to a
writable per-build location, then invokes `packer build` with five
provisioners:

1. `install-openssh-server.ps1` — side-loads Win32-OpenSSH v9.5 from GitHub
   release zip (the image's Windows Update is broken).
2. `install-azure-vm-agent.ps1` — installs WindowsAzureVmAgent.msi so Azure's
   `os_profile` handshake works at launch.
3. `drop-authorized-keys.ps1` — places your ed25519 pubkey at
   `C:\ProgramData\ssh\administrators_authorized_keys` with the strict ACLs
   OpenSSH requires (Administrators:F + SYSTEM:F, no inheritance).
4. `configure-autologon.ps1` — overwrites the LSA `DefaultPassword` secret
   via `LsaStorePrivateData` P/Invoke so AutoAdminLogon fires on boot.
   Without it, the image boots but no one logs in and the WAA Flask agent
   never starts. Easy to miss.
5. *(`sysprep.ps1` and `embed-waa-server.ps1` are in the tree for reference
   but deliberately not invoked.)*

Output: `packer/output-waa-prepared/waa-windows-prepared.qcow2` — a ~2 GB
overlay qcow2 (still references the base as backing file). ~20 min on an
8-core host.

### Stage 5 — smoke test locally

```bash
uv run python packer/smoke_test.py
# Expected:
#   [smoke] GUEST AGENT: UP (~640KB PNG)
#   [smoke] SSH: UP + authenticated as Docker
```

### Stage 6 — flatten + upload

The overlay references the base image as its backing file — that won't work
when someone else downloads only the overlay. Flatten:

```bash
qemu-img convert -O qcow2 \
  packer/output-waa-prepared/waa-windows-prepared.qcow2 \
  ~/.cube/hf-staging/waa-windows-prepared.qcow2

uv run --with huggingface_hub --with hf_transfer python scripts/upload_image.py \
  --image-path ~/.cube/hf-staging/waa-windows-prepared.qcow2 \
  --repo-id <your-hf-user>/waa-windows-image \
  --filename waa-windows-prepared.qcow2
```

Then update [`WAA_WINDOWS_RESOURCE`](src/waa_cube/azure.py) `source_url` to
your upload, and commit.

## Installation

```bash
uv pip install -e .
```

## Observations

Each step the agent receives:

1. A **screenshot** (1280×800 PNG)
2. An **element table** (linearized accessibility tree):

```
index  tag                name              text  x    y    w    h
1      shell_traywnd      Taskbar           ""    0    752  1280 48
2      togglebutton       Start             ""    396  752  45   48
3      cabinetwclass      Documents - ...   ""    250  86   800  600
```

Click center: `cx = x + w//2`, `cy = y + h//2`.

With `WAABenchmark(use_som=True)`, the screenshot has numbered red bounding
boxes drawn over each interactive element and the table drops the
coordinates: agents click via `pyautogui.click(*tag_N)`, where `tag_N` is
auto-defined by the harness at execution time as the element's centre.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WAA_BUILD_ADMIN_PASSWORD` | *(required for image build)* | Used by `bootstrap_winrm.py` and `make build-image`. Azure complexity: 12-72 chars, 3 of 4 character classes. |
| `WAA_WINDOWS_ADMIN_PASSWORD` | *(required for Azure launch)* | Same value as above, exported for the Azure recipe. |
| `AZURE_RESOURCE_GROUP` | `ui_assist` | Resource group for the compute gallery + bootstrap storage. |
| `AZURE_STORAGE_ACCOUNT` | `cubeexpvhd` | Storage account for VHD blobs during provisioning. |
| `WAA_SETUP_ISO` | *(only for Stage 2)* | Windows 11 Enterprise ISO path — consumed by upstream WAA's image-build container. |

## Debug / Testing

```bash
cube test waa-cube
```

Runs 2 deterministic debug tasks without an LLM:

- **waa-debug-notepad** — opens Notepad via Win+R, evaluator checks the window title
- **waa-debug-infeasible** — calls `fail()` for a nonexistent app

## Known Issues

- **`/setup/upload` tail at high concurrency** (n=20+): a small fraction of
  freshly-booted VMs return 502 from `/setup/upload` for ~60 s. The cause is
  not fully characterised — see `recipes/waa/probe_n20_502.py`-style probes
  in the experiments branch. The 5-attempt × 2/4/8/16/32 s backoff in
  `_download_setup` recovers most cases. Stay at n=10 if you can't tolerate
  any tail; n=20 has ~28 % per-cohort failure rate when it fires.
- **Accessibility-tree walk can hang**: pywinauto UIA can take 5+ minutes on
  certain desktop states. In-guest bug, not provisioning.

## Package Structure

```
src/waa_cube/
├── __init__.py              # Public exports
├── azure.py                 # WAA_WINDOWS_RESOURCE definition
├── benchmark.py             # WAABenchmark (config), WAABenchmarkRuntime, WAATaskConfig
├── task.py                  # WAATask, WAATaskExecutionInfo
├── computer.py              # ComputerConfig wrapper
├── debug.py                 # DebugWAABenchmark, DebugAgent
├── debug_tasks.json         # Debug task definitions
├── debug_task_metadata.json # Debug task metadata (CUBE format)
├── task_metadata.json       # Shipped slim TaskMetadata (~350 B/task)
├── task_execution_info.json # Shipped heavy execution info (config, evaluator, snapshot, …)
└── vm_backend/
    ├── backend.py           # Legacy local VM backend utilities
    ├── evaluator.py         # GuestAgentProxy, Evaluator
    ├── setup_controller.py  # Task setup step execution + upload retry
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
