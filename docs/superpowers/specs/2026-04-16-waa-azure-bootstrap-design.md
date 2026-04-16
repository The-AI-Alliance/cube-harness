# WAA Windows Azure Bootstrap — Design Spec

**Date**: 2026-04-16
**Branch**: fix/waa-cube-communication
**Status**: Approved

## Overview

Add Windows support to `cube-infra-azure` so that the WAA (WindowsAgentArena) benchmark can run on Azure VMs using the same `AzureInfraConfig` pipeline already used by OSWorld. No new flags on `VMResourceConfig` — OS type is inferred from the partition table during bootstrap and from the gallery image definition at launch time.

**Source image**: `https://huggingface.co/datasets/kushasareen/waa-windows-image` — base Windows 11 qcow2, no OpenSSH, no sysprep.

---

## Section 1: Bootstrap Script (`_AZURE_BOOTSTRAP_SCRIPT`)

### OS detection

After `qemu-img convert`, the script inspects the VHD partition table:

```bash
ROOT_PART=$(lsblk -rno NAME,FSTYPE "$LOOP" | awk '$2=="ext4" {print "/dev/"$1}' | tail -1)
NTFS_PART=$(lsblk -rno NAME,FSTYPE "$LOOP" | awk '$2=="ntfs" {print "/dev/"$1}' | tail -1)
```

- `ext4` found → existing Linux chroot path (unchanged)
- `ntfs` found → new Windows QEMU boot path

### Windows bootstrap path (replaces chroot block)

**Tools added** to the `apt-get install` line: `qemu-system-x86 ovmf winrm` (curl handles WinRM SOAP — no extra tool needed beyond what's already installed).

**Steps:**

1. **Install QEMU + OVMF** (already in tools section, just add packages)

2. **Boot Windows headlessly:**
```bash
qemu-system-x86_64 \
    -m 4096 -smp 4 \
    -drive file=/data/output.vhd,format=vpc,if=virtio \
    -bios /usr/share/ovmf/OVMF.fd \
    -net nic,model=e1000 -net user,hostfwd=tcp::5985-:5985 \
    -display none -daemonize -pidfile /tmp/qemu.pid
```

3. **Poll WinRM port 5985** until Windows is ready (timeout 10 min):
```bash
timeout 600 bash -c 'until nc -z localhost 5985 2>/dev/null; do sleep 15; done'
```

4. **Run PowerShell via WinRM** (curl SOAP envelope) to:
   - `Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0`
   - `Set-Service sshd -StartupType Automatic`
   - Set PowerShell as the OpenSSH default shell (registry key)
   - `C:\Windows\System32\Sysprep\sysprep.exe /generalize /oobe /shutdown /quiet`

5. **Wait for QEMU to exit** — sysprep shuts down the VM (~2-3 min):
```bash
timeout 600 bash -c 'while kill -0 $(cat /tmp/qemu.pid) 2>/dev/null; do sleep 10; done'
```

6. Upload VHD via azcopy (existing, unchanged).

### WinRM invocation

The WAA image has WinRM enabled with HTTP basic auth. The PowerShell payload is sent as a WinRM SOAP `Invoke` request via `curl`:

```bash
curl -s -X POST http://localhost:5985/wsman \
    -H "Content-Type: application/soap+xml;charset=UTF-8" \
    -u "Administrator:${WINRM_PASSWORD}" \
    -d "<SOAP envelope with PowerShell encoded command>"
```

The WinRM credentials (`WINRM_PASSWORD`) are passed into the bootstrap script as a placeholder alongside `{hf_url}`, `{vhd_sas_url}`, etc. — injected by `_bootstrap()` from a new `AzureInfraConfig` field `windows_admin_password` (stored as a secret, excluded from repr/serialization).

---

## Section 2: `_create_image_definition` and `launch()`

### `_create_image_definition`

The bootstrap script writes a small metadata blob `<image-name>.os_type` (e.g. `waa-windows-vm.os_type`) containing either `"windows"` or `"linux"` to the same blob container as the VHD, immediately after OS detection. `_bootstrap()` reads this blob back and passes `os_type` to `_create_image_definition`:

```python
arm_os_type = "Windows" if os_type == "windows" else "Linux"
sku = "windows" if os_type == "windows" else "linux"
# ARM call uses arm_os_type and sku
```

### `launch()`

Query the gallery image definition to detect OS type — no new field on `VMResourceConfig`:

```python
img = compute.gallery_images.get(rg, gallery_name, image_def)
is_windows = (img.os_type.value == "Windows")
```

**Linux path** (unchanged): `linux_configuration` with `ssh.public_keys`.

**Windows path**:
- `os_profile` uses `windows_configuration` (no SSH key injection here)
- After VM creation completes, inject the caller's SSH public key via `VMAccessAgent` extension:

```python
compute.virtual_machine_extensions.begin_create_or_update(
    rg, vm_name, "enableSSH",
    {
        "location": self.location,
        "publisher": "Microsoft.Compute",
        "type_properties_type": "VMAccessAgent",
        "type_handler_version": "2.4",
        "settings": {},
        "protected_settings": {
            "username": "Administrator",
            "ssh_key": pubkey,
        },
    }
).result()
```

- `wait_for_ssh` then tries `Administrator` as the user (with `fallback_users` support already in `infra_utils.py`)
- SSH tunnel and endpoint creation are unchanged

---

## Section 3: `waa-cube` Integration

### `VMResourceConfig` constant

In `waa-cube` (e.g. a new `src/waa_cube/azure.py`):

```python
from cube.resource import VMResourceConfig

WAA_WINDOWS_RESOURCE = VMResourceConfig(
    name="waa-windows-vm",
    source_url="https://huggingface.co/datasets/kushasareen/waa-windows-image/resolve/main/<filename>.qcow2",
    default_ttl_seconds=60 * 60 * 2,
)
```

### `WAABenchmark`

Add `infra: InfraConfig | None = None` field alongside existing `vm_backend`. `WAATask.reset()` dispatches:
- `infra` set → `infra.launch(WAA_WINDOWS_RESOURCE)` → SSH tunnel → HTTP endpoint
- `vm_backend` set → existing Docker+QEMU path (unchanged, backward compatible)

### Recipe (`recipes/waa/eval_azure_waa.py`)

Mirrors `eval_azure_osworld.py`:

```python
INFRA = AzureInfraConfig(
    resource_group=os.environ["AZURE_RESOURCE_GROUP"],
    windows_admin_password=os.environ["WAA_WINDOWS_ADMIN_PASSWORD"],
)
benchmark = WAABenchmark(infra=INFRA, default_tool_config=ComputerConfig())
benchmark.setup()
INFRA.provision(WAA_WINDOWS_RESOURCE)  # one-time ~60-90 min
exp = Experiment(...)
run_with_ray(exp, n_cpus=40)
```

---

## Files Changed

| File | Change |
|------|--------|
| `cube-infra-azure/src/cube_infra_azure/azure.py` | Windows branch in `_AZURE_BOOTSTRAP_SCRIPT`; `_create_image_definition` os_type param; `launch()` Windows path with VMAccessAgent |
| `cube-standard/src/cube/resource.py` | No changes (os_type inferred, not stored) |
| `cubes/windows-agent-arena-cube/src/waa_cube/azure.py` | New file: `WAA_WINDOWS_RESOURCE` constant |
| `cubes/windows-agent-arena-cube/src/waa_cube/benchmark.py` | Add `infra` field to `WAABenchmark` |
| `cubes/windows-agent-arena-cube/src/waa_cube/task.py` | Dispatch on `infra` vs `vm_backend` in `reset()` |
| `recipes/waa/eval_azure_waa.py` | New Azure recipe |

## Risks

1. **WinRM credentials**: The WAA base image's Administrator password must be known. Need to confirm it from the WAA Docker image build scripts.
2. **WinRM enabled**: Assumed enabled in WAA base image — needs verification.
3. **sysprep on WAA image**: sysprep generalizes the image; the resulting OOBE boot on Azure needs to complete before the VMAccessAgent extension can run. Azure handles this automatically for Generalized Windows images.
4. **Bootstrap VM disk size**: WAA qcow2 is ~60 GB. Bootstrap VM OS disk needs to be ≥150 GB (qcow2 + VHD + headroom). `bootstrap_os_disk_gb` should default to `256` for Windows resources.
5. **OVMF vs SeaBIOS**: WAA image may require SeaBIOS (legacy BIOS) rather than OVMF (UEFI). Need to test — fall back to `-bios seabios` if OVMF fails to boot.
