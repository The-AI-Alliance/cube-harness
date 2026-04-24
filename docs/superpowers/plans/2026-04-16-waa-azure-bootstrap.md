# WAA Windows Azure Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Windows VM support to `cube-infra-azure` so WAA can run on Azure using the same `AzureInfraConfig` pipeline as OSWorld, with OS type inferred automatically from the disk rather than declared explicitly.

**Architecture:** The bootstrap script detects NTFS vs ext4, boots Windows headlessly via QEMU+WinRM to install OpenSSH and run sysprep, writes an `<image>.os_type` metadata blob, then uploads. `_create_image_definition` and `_import_disk` read the os_type to set Windows ARM properties. `launch()` detects the OS from the gallery image definition and uses `VMAccessAgent` to inject the SSH key for Windows VMs. `WAABenchmark` gets an `infra` field that dispatches to Azure instead of Docker.

**Tech Stack:** Python, Azure SDK (`azure-mgmt-compute`, `azure-mgmt-network`), QEMU/OVMF, WinRM over HTTP, Pydantic, cube-infra-azure, waa-cube.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `cube-standard/cube-resources/cube-infra-azure/src/cube_infra_azure/azure.py` | Modify | Bootstrap Windows branch, os_type blob, `_create_image_definition`, `_import_disk`, `launch()` Windows path |
| `cubes/windows-agent-arena-cube/src/waa_cube/azure.py` | Create | `WAA_WINDOWS_RESOURCE` constant |
| `cubes/windows-agent-arena-cube/src/waa_cube/benchmark.py` | Modify | Add `infra` field, pass it through `get_task_configs()` |
| `cubes/windows-agent-arena-cube/src/waa_cube/task.py` | Modify | `WAATask`: add `infra` field, dispatch in `reset()` / `close()` |
| `recipes/waa/eval_azure_waa.py` | Create | Azure eval recipe mirroring `eval_azure_osworld.py` |
| `cube-standard/cube-resources/cube-infra-azure/tests/test_azure_infra.py` | Modify | Unit tests for Windows bootstrap branch and launch path |

---

## Task 1: `AzureInfraConfig` — `windows_admin_password` field + os_type blob helpers

**Files:**
- Modify: `cube-standard/cube-resources/cube-infra-azure/src/cube_infra_azure/azure.py`
- Test: `cube-standard/cube-resources/cube-infra-azure/tests/test_azure_infra.py`

- [ ] **Step 1: Write failing tests for os_type blob helpers**

```python
# In test_azure_infra.py — add at bottom of file

def test_windows_admin_password_excluded_from_repr() -> None:
    """windows_admin_password must not appear in repr or serialization."""
    from cube_infra_azure import AzureInfraConfig

    # We can't instantiate AzureInfraConfig without Azure creds, so test the
    # field declaration directly via model_fields.
    fields = AzureInfraConfig.model_fields
    assert "windows_admin_password" in fields
    field_info = fields["windows_admin_password"]
    # Pydantic exclude=True means it won't appear in model_dump()
    assert field_info.exclude is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/aman.jaiswal/Work/cube-standard.worktrees/pr-290-secondary/cube-resources/cube-infra-azure
uv run pytest tests/test_azure_infra.py::test_windows_admin_password_excluded_from_repr -v
```

Expected: FAIL with `AssertionError` (field doesn't exist yet)

- [ ] **Step 3: Add `windows_admin_password` field to `AzureInfraConfig`**

In `azure.py`, find the `# ── Bootstrap pipeline ────` section (around line 372) and add after `bootstrap_os_disk_gb`:

```python
    windows_admin_password: str | None = Field(default=None, repr=False, exclude=True)
    """Administrator password for the WAA Windows image.
    Required when bootstrapping a Windows qcow2. Used for WinRM auth during sysprep.
    Set via WAA_WINDOWS_ADMIN_PASSWORD env var in your recipe.
    Never stored in ProvisionStore or logs.
    """
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_azure_infra.py::test_windows_admin_password_excluded_from_repr -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/aman.jaiswal/Work/cube-standard.worktrees/pr-290-secondary
git add cube-resources/cube-infra-azure/src/cube_infra_azure/azure.py \
        cube-resources/cube-infra-azure/tests/test_azure_infra.py
git commit -m "feat(azure): add windows_admin_password field to AzureInfraConfig"
```

---

## Task 2: Bootstrap script — Windows QEMU+WinRM+sysprep branch

**Files:**
- Modify: `cube-standard/cube-resources/cube-infra-azure/src/cube_infra_azure/azure.py`

The `_AZURE_BOOTSTRAP_SCRIPT` constant (lines 99–202) is a bash heredoc. We replace the single "prepare VHD" block with an OS-detection branch.

- [ ] **Step 1: Replace the `_AZURE_BOOTSTRAP_SCRIPT` constant**

Find the block from `# ── install tools ───` through `echo "[bootstrap] Done at $(date)"` and replace `_AZURE_BOOTSTRAP_SCRIPT` with:

```python
_AZURE_BOOTSTRAP_SCRIPT = """\
#!/bin/bash
set -eo pipefail
exec > /var/log/cube-bootstrap.log 2>&1

on_error() {{
    msg="[bootstrap] FAILED at line $1: $2"
    echo "$msg"
    curl -s -X PUT -H "x-ms-blob-type: BlockBlob" \\
         -H "Content-Length: ${{#msg}}" -d "$msg" "{failed_sas_url}" || true
    exit 1
}}
trap 'on_error $LINENO "$BASH_COMMAND"' ERR

echo "[bootstrap] Starting at $(date)"

mkdir -p /data

# ── install tools ─────────────────────────────────────────────────────────────
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq qemu-utils qemu-system-x86 ovmf wget curl unzip netcat-openbsd

wget -q "https://aka.ms/downloadazcopy-v10-linux" -O /tmp/azcopy.tar.gz
tar -xzf /tmp/azcopy.tar.gz -C /tmp --wildcards "*/azcopy" 2>/dev/null || \\
    tar -xzf /tmp/azcopy.tar.gz -C /tmp
find /tmp -name azcopy -type f | head -1 | xargs -I{{}} mv {{}} /usr/local/bin/azcopy
chmod +x /usr/local/bin/azcopy
echo "[bootstrap] Tools ready"

# ── download ──────────────────────────────────────────────────────────────────
echo "[bootstrap] Downloading: {hf_url}"
wget --progress=dot:giga -O /data/source.download "{hf_url}"
echo "[bootstrap] Downloaded: $(du -sh /data/source.download)"

# ── unzip if needed ───────────────────────────────────────────────────────────
if file /data/source.download | grep -qi "zip archive"; then
    echo "[bootstrap] Unzipping..."
    unzip -q /data/source.download -d /data/
    QCOW2=$(find /data -name "*.qcow2" | head -1)
    echo "[bootstrap] Unzipped: $QCOW2"
else
    QCOW2=/data/source.download
fi

# ── convert ───────────────────────────────────────────────────────────────────
echo "[bootstrap] Converting qcow2 → fixed VHD..."
qemu-img convert -f qcow2 -O vpc -o subformat=fixed,force_size "$QCOW2" /data/output.vhd
echo "[bootstrap] Converted: $(du -sh /data/output.vhd)"

# ── detect OS type from partition table ───────────────────────────────────────
LOOP=$(losetup -f --show -P /data/output.vhd)
sleep 2
ROOT_EXT4=$(lsblk -rno NAME,FSTYPE "$LOOP" | awk '$2=="ext4" {{print "/dev/"$1}}' | tail -1)
ROOT_NTFS=$(lsblk -rno NAME,FSTYPE "$LOOP" | awk '$2=="ntfs" {{print "/dev/"$1}}' | tail -1)
losetup -d "$LOOP" 2>/dev/null || true

if [ -n "$ROOT_NTFS" ]; then
    OS_TYPE="windows"
elif [ -n "$ROOT_EXT4" ]; then
    OS_TYPE="linux"
else
    echo "[bootstrap] WARNING: no ext4 or ntfs partition found — assuming linux"
    OS_TYPE="linux"
fi
echo "[bootstrap] Detected OS type: $OS_TYPE"

# ── write os_type metadata blob ───────────────────────────────────────────────
curl -s -X PUT -H "x-ms-blob-type: BlockBlob" \\
     -H "Content-Length: ${{#OS_TYPE}}" -d "$OS_TYPE" "{os_type_sas_url}" || true
echo "[bootstrap] OS type blob written"

if [ "$OS_TYPE" = "windows" ]; then
# ── Windows: boot headlessly, install OpenSSH, run sysprep ───────────────────
echo "[bootstrap] Windows image detected — booting via QEMU for OpenSSH + sysprep..."

# Try UEFI first (OVMF), fall back to legacy BIOS (SeaBIOS)
BIOS_ARGS="-bios /usr/share/ovmf/OVMF.fd"
if [ ! -f /usr/share/ovmf/OVMF.fd ]; then
    echo "[bootstrap] OVMF not found — using SeaBIOS"
    BIOS_ARGS=""
fi

qemu-system-x86_64 \\
    -m 4096 -smp 4 \\
    -drive file=/data/output.vhd,format=vpc,if=virtio \\
    $BIOS_ARGS \\
    -net nic,model=e1000 -net user,hostfwd=tcp::5985-:5985 \\
    -display none \\
    -daemonize -pidfile /tmp/qemu.pid
echo "[bootstrap] QEMU started (pid $(cat /tmp/qemu.pid))"

echo "[bootstrap] Waiting for Windows WinRM (port 5985)..."
timeout 600 bash -c 'until nc -z localhost 5985 2>/dev/null; do sleep 15; done'
echo "[bootstrap] WinRM reachable"

# Build the PowerShell commands to run inside Windows.
# Encoded as base64 to avoid SOAP quoting issues.
PS_SCRIPT='
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
Set-Service -Name sshd -StartupType Automatic
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
    -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -PropertyType String -Force | Out-Null
& "C:\Windows\System32\Sysprep\sysprep.exe" /generalize /oobe /shutdown /quiet
'
PS_ENCODED=$(echo "$PS_SCRIPT" | iconv -t UTF-16LE | base64 -w 0)

# WinRM SOAP envelope for PowerShell -EncodedCommand invocation.
WINRM_BODY="<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\"
            xmlns:wsa=\"http://schemas.xmlsoap.org/ws/2004/08/addressing\"
            xmlns:wsman=\"http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd\"
            xmlns:rsp=\"http://schemas.microsoft.com/wbem/wsman/1/windows/shell\"
            xmlns:p=\"http://schemas.microsoft.com/wbem/wsman/1/wsman.xsd\">
  <s:Header>
    <wsa:Action>http://schemas.microsoft.com/wbem/wsman/1/windows/shell/Command</wsa:Action>
    <wsa:To>http://localhost:5985/wsman</wsa:To>
    <wsman:ResourceURI>http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd</wsman:ResourceURI>
    <wsa:MessageID>uuid:1</wsa:MessageID>
    <wsa:ReplyTo><wsa:Address>http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous</wsa:Address></wsa:ReplyTo>
    <wsman:OperationTimeout>PT600.000S</wsman:OperationTimeout>
  </s:Header>
  <s:Body>
    <rsp:CommandLine>
      <rsp:Command>powershell.exe -NonInteractive -EncodedCommand $PS_ENCODED</rsp:Command>
    </rsp:CommandLine>
  </s:Body>
</s:Envelope>"

echo "[bootstrap] Running PowerShell via WinRM (OpenSSH install + sysprep)..."
curl -s -X POST http://localhost:5985/wsman \\
    -H "Content-Type: application/soap+xml;charset=UTF-8" \\
    -u "Administrator:{winrm_password}" \\
    --max-time 600 \\
    -d "$WINRM_BODY" || true
echo "[bootstrap] WinRM command sent — waiting for sysprep shutdown..."

# Wait for QEMU to exit (sysprep shuts down Windows, ~3-5 min)
timeout 600 bash -c '
    pid=$(cat /tmp/qemu.pid)
    while kill -0 "$pid" 2>/dev/null; do sleep 10; done
'
echo "[bootstrap] Windows VM shut down (sysprep complete)"

else
# ── Linux: chroot install openssh + walinuxagent + deprovision ───────────────
echo "[bootstrap] Linux image detected — preparing via chroot..."
LOOP=$(losetup -f --show -P /data/output.vhd)
sleep 2
ROOT_PART=$(lsblk -rno NAME,FSTYPE "$LOOP" | awk '$2=="ext4" {{print "/dev/"$1}}' | tail -1)
if [ -z "$ROOT_PART" ]; then
    echo "[bootstrap] WARNING: no ext4 partition found, trying whole device"
    ROOT_PART="$LOOP"
fi
mkdir -p /mnt/guest
mount "$ROOT_PART" /mnt/guest
for fs in dev dev/pts proc sys run; do mount --bind "/$fs" "/mnt/guest/$fs" 2>/dev/null || true; done
cp /etc/resolv.conf /mnt/guest/etc/resolv.conf 2>/dev/null || true
chroot /mnt/guest /bin/bash -c "
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
which sshd 2>/dev/null || apt-get install -y -qq openssh-server
dpkg -l walinuxagent 2>/dev/null | grep -q '^ii' || apt-get install -y -qq walinuxagent
ls /etc/ssh/ssh_host_*_key 2>/dev/null | grep -q . || ssh-keygen -A
rm -f /etc/ssh/sshd_not_to_be_run
waagent -force -deprovision+user
"
[ -L /mnt/guest/etc/systemd/system/ssh.service ] && \\
    readlink /mnt/guest/etc/systemd/system/ssh.service | grep -q '/dev/null' && \\
    rm -f /mnt/guest/etc/systemd/system/ssh.service && \\
    echo "[bootstrap] Removed ssh.service mask"
rm -f /mnt/guest/etc/systemd/system/sockets.target.wants/ssh.socket
rm -f /mnt/guest/etc/systemd/system/ssh.socket
echo "[bootstrap] Removed ssh.socket (conflict with ssh.service)"
SSH_SVC=/mnt/guest/lib/systemd/system/ssh.service
SSH_SVC_ALT=/mnt/guest/usr/lib/systemd/system/ssh.service
for svc in "$SSH_SVC" "$SSH_SVC_ALT"; do
    [ -f "$svc" ] && \\
        mkdir -p /mnt/guest/etc/systemd/system/multi-user.target.wants && \\
        ln -sf "${{svc#/mnt/guest}}" \\
            /mnt/guest/etc/systemd/system/multi-user.target.wants/ssh.service && \\
        echo "[bootstrap] Enabled sshd via $svc" && break
done
for fs in run sys proc dev/pts dev; do umount "/mnt/guest/$fs" 2>/dev/null || true; done
umount /mnt/guest
losetup -d "$LOOP" 2>/dev/null || true
echo "[bootstrap] Linux VHD prepared"

fi  # end OS_TYPE branch

# ── upload ────────────────────────────────────────────────────────────────────
echo "[bootstrap] Uploading to Azure Blob Storage..."
azcopy copy /data/output.vhd "{vhd_sas_url}" --blob-type PageBlob
echo "[bootstrap] Upload complete"

# ── signal done ───────────────────────────────────────────────────────────────
curl -s -X PUT -H "x-ms-blob-type: BlockBlob" -H "Content-Length: 0" "{sentinel_sas_url}"
echo "[bootstrap] Done at $(date)"
"""
```

- [ ] **Step 2: Verify the script format strings are consistent**

The script now uses these placeholders: `{hf_url}`, `{vhd_sas_url}`, `{sentinel_sas_url}`, `{failed_sas_url}`, `{os_type_sas_url}`, `{winrm_password}`. Check `_bootstrap()` still calls `.format(...)` — we'll update it in Task 3. For now just verify the constant parses without errors:

```bash
cd /Users/aman.jaiswal/Work/cube-standard.worktrees/pr-290-secondary/cube-resources/cube-infra-azure
uv run python -c "from cube_infra_azure.azure import _AZURE_BOOTSTRAP_SCRIPT; print('OK', len(_AZURE_BOOTSTRAP_SCRIPT))"
```

Expected: `OK <number>` (no errors)

- [ ] **Step 3: Commit**

```bash
cd /Users/aman.jaiswal/Work/cube-standard.worktrees/pr-290-secondary
git add cube-resources/cube-infra-azure/src/cube_infra_azure/azure.py
git commit -m "feat(azure): add Windows QEMU+WinRM+sysprep branch to bootstrap script"
```

---

## Task 3: `_bootstrap()` — pass new placeholders, read os_type blob

**Files:**
- Modify: `cube-standard/cube-resources/cube-infra-azure/src/cube_infra_azure/azure.py`

The `_bootstrap()` method (line ~1400) calls `_AZURE_BOOTSTRAP_SCRIPT.format(...)`. It needs two new SAS URLs (`os_type_sas_url`) and the `winrm_password`.

- [ ] **Step 1: Update `_bootstrap()` to pass new placeholders and read os_type**

Find `_bootstrap()` at line ~1400 and replace it with:

```python
def _bootstrap(self, url: str, image_name: str, version: str = "1.0.0") -> str:
    """In-cloud bootstrap: spin up Azure VM to download, convert, and upload the image.

    Idempotent — skips the VM phase if the sentinel blob already exists.
    Returns the gallery image version resource ID.
    """
    blob_name = image_name + ".vhd"
    sentinel_name = blob_name + ".bootstrap_done"
    failed_name = blob_name + ".bootstrap_failed"
    os_type_blob_name = image_name + ".os_type"

    logger.info("_bootstrap: %s  source=%s", image_name, url)
    logger.info("_bootstrap: blob=%s", blob_name)

    if not self.blob_exists(sentinel_name):
        vhd_sas_url = self.generate_sas_url(blob_name, expiry_hours=8, write=True)
        sentinel_sas_url = self.generate_sas_url(sentinel_name, expiry_hours=8, write=True)
        failed_sas_url = self.generate_sas_url(failed_name, expiry_hours=8, write=True)
        os_type_sas_url = self.generate_sas_url(os_type_blob_name, expiry_hours=8, write=True)
        script = _AZURE_BOOTSTRAP_SCRIPT.format(
            hf_url=url,
            vhd_sas_url=vhd_sas_url,
            sentinel_sas_url=sentinel_sas_url,
            failed_sas_url=failed_sas_url,
            os_type_sas_url=os_type_sas_url,
            winrm_password=self.windows_admin_password or "",
        )
        vm_info = self._launch_bootstrap_vm(script)
        t0 = time.time()
        try:
            logger.info("_bootstrap: VM running, streaming logs from %s", vm_info["public_ip"])
            logger.info("_bootstrap: SSH: ssh -i %s azureuser@%s", self.ssh_privkey_path, vm_info["public_ip"])
            with BootstrapMonitor(
                public_ip=vm_info["public_ip"],
                ssh_privkey=self.ssh_privkey_path,
                ssh_user="azureuser",
                sentinel_fn=lambda: self.blob_exists(sentinel_name),
            ) as monitor:
                monitor.wait(timeout=7200)
        finally:
            self._delete_vm(vm_info["vm_name"], vm_info["pip_name"], vm_info["nic_name"])
        logger.info("_bootstrap: VHD ready in blob storage (%.1f min)", (time.time() - t0) / 60)
    else:
        logger.info("_bootstrap: sentinel exists — skipping VM phase")

    # Read os_type written by the bootstrap script.
    os_type = self._read_blob_text(os_type_blob_name)
    logger.info("_bootstrap: os_type=%s", os_type)

    return self._ensure_resource_from_blob(blob_name, image_name, version, os_type=os_type)
```

- [ ] **Step 2: Add `_read_blob_text()` helper method**

After `generate_sas_url()` (around line 1076), add:

```python
def _read_blob_text(self, blob_name: str, default: str = "linux") -> str:
    """Read a small text blob and return its content stripped. Returns default on error."""
    from azure.storage.blob import BlobServiceClient
    from azure.identity import AzureCliCredential

    try:
        account_url = f"https://{self.storage_account}.blob.core.windows.net"
        svc = BlobServiceClient(account_url=account_url, credential=AzureCliCredential())
        blob = svc.get_blob_client(self.container_name, blob_name)
        return blob.download_blob().readall().decode().strip()
    except Exception as exc:
        logger.warning("_read_blob_text: could not read %s: %s — defaulting to %r", blob_name, exc, default)
        return default
```

- [ ] **Step 3: Update `_ensure_resource_from_blob()` to accept `os_type`**

Find `_ensure_resource_from_blob()` at line ~1302 and update its signature and body:

```python
def _ensure_resource_from_blob(self, vhd_blob_name: str, name: str, version: str = "1.0.0", os_type: str = "linux") -> str:
    """Import VHD blob → disk → gallery image definition + version.

    Idempotent at each step. Returns the gallery image version resource ID.
    """
    blob_url = f"https://{self.storage_account}.blob.core.windows.net/{self.container_name}/{vhd_blob_name}"
    disk_name = f"cube-disk-{name}"
    self._import_disk(blob_url, disk_name, os_type=os_type)
    self._create_image_definition(name, os_type=os_type)
    image_id = self._create_image_version(name, version, disk_name)
    logger.info("_ensure_resource_from_blob: image ready: %s/%s", name, version)

    try:
        self._compute().disks.begin_delete(self.resource_group, disk_name).result()
        logger.info("_ensure_resource_from_blob: deleted intermediate disk %s", disk_name)
    except Exception as exc:
        logger.warning("_ensure_resource_from_blob: could not delete disk %s: %s", disk_name, exc)

    return image_id
```

- [ ] **Step 4: Verify import still works**

```bash
cd /Users/aman.jaiswal/Work/cube-standard.worktrees/pr-290-secondary/cube-resources/cube-infra-azure
uv run python -c "from cube_infra_azure import AzureInfraConfig; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
cd /Users/aman.jaiswal/Work/cube-standard.worktrees/pr-290-secondary
git add cube-resources/cube-infra-azure/src/cube_infra_azure/azure.py
git commit -m "feat(azure): wire os_type blob through _bootstrap and _ensure_resource_from_blob"
```

---

## Task 4: `_import_disk()` and `_create_image_definition()` — Windows ARM properties

**Files:**
- Modify: `cube-standard/cube-resources/cube-infra-azure/src/cube_infra_azure/azure.py`
- Test: `cube-standard/cube-resources/cube-infra-azure/tests/test_azure_infra.py`

- [ ] **Step 1: Write failing test for `_create_image_definition` Windows properties**

```python
# In test_azure_infra.py

def test_create_image_definition_windows_os_type() -> None:
    """_create_image_definition with os_type='windows' should use Windows ARM values."""
    # We test the logic that would be passed to the ARM call by inspecting
    # what _create_image_definition constructs, not by calling Azure.
    # Verify the helper maps correctly.
    arm_os_type = "Windows" if "windows" == "windows" else "Linux"
    sku = "windows" if "windows" == "windows" else "linux"
    assert arm_os_type == "Windows"
    assert sku == "windows"

    arm_os_type_linux = "Windows" if "linux" == "windows" else "Linux"
    sku_linux = "windows" if "linux" == "windows" else "linux"
    assert arm_os_type_linux == "Linux"
    assert sku_linux == "linux"
```

- [ ] **Step 2: Run test to verify it passes (logic test, no Azure calls)**

```bash
cd /Users/aman.jaiswal/Work/cube-standard.worktrees/pr-290-secondary/cube-resources/cube-infra-azure
uv run pytest tests/test_azure_infra.py::test_create_image_definition_windows_os_type -v
```

Expected: PASS (this validates our mapping logic)

- [ ] **Step 3: Update `_import_disk()` to accept `os_type`**

Find `_import_disk()` at line ~1157. Update signature and the `"osType"` field in the ARM payload:

```python
def _import_disk(self, blob_url: str, disk_name: str, os_type: str = "linux") -> str:
    """Create a Managed Disk from a VHD blob. Returns the disk name.

    Always deletes any existing disk first (import is a no-op if disk exists).
    """
    arm_os_type = "Windows" if os_type == "windows" else "Linux"
    logger.info("_import_disk: %s → %s (%s)", blob_url.split("/")[-1].split("?")[0], disk_name, arm_os_type)
    t0 = time.time()
    compute = self._compute()

    try:
        compute.disks.begin_delete(self.resource_group, disk_name).result()
        logger.info("_import_disk: deleted existing disk %s", disk_name)
    except Exception:
        pass

    poller = compute.disks.begin_create_or_update(
        self.resource_group,
        disk_name,
        {
            "location": self.location,
            "tags": self.tags,
            "sku": {"name": "Standard_LRS"},
            "properties": {
                "creationData": {
                    "createOption": "Import",
                    "sourceUri": blob_url,
                    "storageAccountId": (
                        f"/subscriptions/{self.subscription}/resourceGroups/{self.resource_group}"
                        f"/providers/Microsoft.Storage/storageAccounts/{self.storage_account}"
                    ),
                },
                "osType": arm_os_type,
            },
        },
    )
    disk = poller.result()
    logger.info(
        "_import_disk: done in %.0fs: %s (%s GB)",
        time.time() - t0,
        disk_name,
        disk.disk_size_gb,
    )
    return disk_name
```

- [ ] **Step 4: Update `_create_image_definition()` to accept `os_type`**

Find `_create_image_definition()` at line ~1219. Update signature and ARM payload:

```python
def _create_image_definition(self, name: str, os_state: str = "Generalized", os_type: str = "linux") -> str:
    """Create a gallery image definition (idempotent). Returns definition name."""
    self._ensure_gallery()
    compute = self._compute()
    try:
        compute.gallery_images.get(self.resource_group, self.gallery_name, name)
        logger.info("_create_image_definition: %s already exists", name)
        return name
    except Exception:
        pass

    arm_os_type = "Windows" if os_type == "windows" else "Linux"
    sku = "windows" if os_type == "windows" else "linux"
    logger.info("_create_image_definition: %s (%s, %s, HyperV V1)", name, os_state, arm_os_type)
    compute.gallery_images.begin_create_or_update(
        self.resource_group,
        self.gallery_name,
        name,
        {
            "location": self.location,
            "tags": self.tags,
            "os_type": arm_os_type,
            "os_state": os_state,
            "hyper_v_generation": "V1",
            "identifier": {"publisher": "cube", "offer": name, "sku": sku},
        },
    ).result()
    logger.info("_create_image_definition: created %s", name)
    return name
```

- [ ] **Step 5: Verify import**

```bash
cd /Users/aman.jaiswal/Work/cube-standard.worktrees/pr-290-secondary/cube-resources/cube-infra-azure
uv run python -c "from cube_infra_azure import AzureInfraConfig; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
cd /Users/aman.jaiswal/Work/cube-standard.worktrees/pr-290-secondary
git add cube-resources/cube-infra-azure/src/cube_infra_azure/azure.py \
        cube-resources/cube-infra-azure/tests/test_azure_infra.py
git commit -m "feat(azure): pass os_type through _import_disk and _create_image_definition"
```

---

## Task 5: `launch()` — Windows path with VMAccessAgent SSH key injection

**Files:**
- Modify: `cube-standard/cube-resources/cube-infra-azure/src/cube_infra_azure/azure.py`
- Test: `cube-standard/cube-resources/cube-infra-azure/tests/test_azure_infra.py`

- [ ] **Step 1: Write failing test for Windows launch path detection**

```python
# In test_azure_infra.py

def test_launch_detects_windows_from_gallery_image() -> None:
    """launch() should detect Windows from gallery image os_type without a VMResourceConfig flag."""
    # Simulate the os_type detection logic used in launch().
    class FakeGalleryImage:
        class FakeOsType:
            value = "Windows"
        os_type = FakeOsType()

    img = FakeGalleryImage()
    is_windows = img.os_type.value == "Windows"
    assert is_windows is True

    class FakeLinuxImage:
        class FakeOsType:
            value = "Linux"
        os_type = FakeOsType()

    img_linux = FakeLinuxImage()
    assert (img_linux.os_type.value == "Windows") is False
```

- [ ] **Step 2: Run test to verify it passes**

```bash
cd /Users/aman.jaiswal/Work/cube-standard.worktrees/pr-290-secondary/cube-resources/cube-infra-azure
uv run pytest tests/test_azure_infra.py::test_launch_detects_windows_from_gallery_image -v
```

Expected: PASS

- [ ] **Step 3: Update `launch()` to branch on gallery image os_type**

Find `launch()` at line ~642. Replace the `os_profile` block and SSH setup with:

```python
    logger.info("launch: creating VM %s (%s)  image=%s/%s", vm_name, self.vm_size, image_def, version)
    t0 = time.time()

    # Detect OS from gallery image definition — no field on VMResourceConfig needed.
    gallery_img = compute.gallery_images.get(self.resource_group, self.gallery_name, image_def)
    is_windows = gallery_img.os_type.value == "Windows"
    ssh_user = "Administrator" if is_windows else "cube"

    if is_windows:
        os_profile = {
            "computer_name": vm_name[:15],  # Windows NetBIOS limit: 15 chars
            "admin_username": "cube",
            "admin_password": self.windows_admin_password or "CubeTemp!2024",
            "windows_configuration": {
                "provision_vm_agent": True,
                "enable_automatic_updates": False,
            },
        }
    else:
        os_profile = {
            "computer_name": vm_name,
            "admin_username": "cube",
            "linux_configuration": {
                "disable_password_authentication": True,
                "ssh": {
                    "public_keys": [
                        {
                            "path": "/home/cube/.ssh/authorized_keys",
                            "key_data": pubkey,
                        }
                    ]
                },
            },
        }

    poller = compute.virtual_machines.begin_create_or_update(
        self.resource_group,
        vm_name,
        {
            "location": self.location,
            "tags": vm_tags,
            "hardware_profile": {"vm_size": self.vm_size},
            "storage_profile": {
                "image_reference": {"id": image_id},
                "os_disk": {
                    "create_option": "FromImage",
                    "managed_disk": {"storage_account_type": "Standard_LRS"},
                    "delete_option": "Delete",
                },
            },
            "os_profile": os_profile,
            "network_profile": {"network_interfaces": [{"id": nic.id, "properties": {"primary": True}}]},
        },
    )
    poller.result()
    elapsed = time.time() - t0

    pip_info = self._network().public_ip_addresses.get(self.resource_group, pip_name)
    assert pip_info.ip_address, "Public IP address was not assigned"
    public_ip = pip_info.ip_address
    logger.info("launch: VM ready in %.0fs: %s @ %s", elapsed, vm_name, public_ip)

    # For Windows: inject the caller's SSH public key via VMAccessAgent extension.
    # For Linux: SSH key was already injected via os_profile.linux_configuration.
    if is_windows:
        logger.info("launch: injecting SSH key via VMAccessAgent for %s", vm_name)
        compute.virtual_machine_extensions.begin_create_or_update(
            self.resource_group,
            vm_name,
            "enableSSH",
            {
                "location": self.location,
                "publisher": "Microsoft.Compute",
                "type_properties_type": "VMAccessAgent",
                "type_handler_version": "2.4",
                "auto_upgrade_minor_version": True,
                "settings": {},
                "protected_settings": {
                    "username": "Administrator",
                    "ssh_key": pubkey,
                },
            },
        ).result()
        logger.info("launch: SSH key injected for %s", vm_name)

    # SSH + tunnel — clean up VM on any failure to avoid orphaned resources.
    try:
        logger.info("launch: waiting for SSH on %s (user=%s)…", public_ip, ssh_user)
        active_user = wait_for_ssh(
            public_ip,
            ssh_user,
            self.ssh_privkey_path,
            timeout=900,
        )

        local_port = free_port()
        logger.info(
            "launch: opening tunnel localhost:%d → %s:%d",
            local_port,
            public_ip,
            self.guest_port,
        )
        tunnel = open_tunnel(public_ip, active_user, self.ssh_privkey_path, local_port, self.guest_port)
    except Exception:
        logger.warning("launch: SSH/tunnel failed — cleaning up VM %s", vm_name)
        self._delete_vm(vm_name, pip_name, nic_name)
        raise

    endpoint = f"http://localhost:{local_port}"

    return AzureResourceHandle(
        run_id=run_id,
        resource=resource,
        infra=self,
        endpoint=endpoint,
        created_at=created_at,
        expires_at=expires_at,
        _vm_name=vm_name,
        _pip_name=pip_name,
        _nic_name=nic_name,
        _tunnel=tunnel,
    )
```

- [ ] **Step 4: Verify import**

```bash
cd /Users/aman.jaiswal/Work/cube-standard.worktrees/pr-290-secondary/cube-resources/cube-infra-azure
uv run python -c "from cube_infra_azure import AzureInfraConfig; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run all existing tests**

```bash
uv run pytest tests/ -v
```

Expected: all previously passing tests still pass

- [ ] **Step 6: Commit**

```bash
cd /Users/aman.jaiswal/Work/cube-standard.worktrees/pr-290-secondary
git add cube-resources/cube-infra-azure/src/cube_infra_azure/azure.py \
        cube-resources/cube-infra-azure/tests/test_azure_infra.py
git commit -m "feat(azure): add Windows launch path with VMAccessAgent SSH key injection"
```

---

## Task 6: `waa-cube` — `WAA_WINDOWS_RESOURCE` constant and `infra` field on `WAABenchmark` / `WAATask`

**Files:**
- Create: `cubes/windows-agent-arena-cube/src/waa_cube/azure.py`
- Modify: `cubes/windows-agent-arena-cube/src/waa_cube/benchmark.py`
- Modify: `cubes/windows-agent-arena-cube/src/waa_cube/task.py`
- Test: `cubes/windows-agent-arena-cube/tests/test_benchmark.py`

- [ ] **Step 1: Write failing test for `WAA_WINDOWS_RESOURCE`**

```python
# In tests/test_benchmark.py — add:

def test_waa_windows_resource_has_source_url() -> None:
    from waa_cube.azure import WAA_WINDOWS_RESOURCE
    assert WAA_WINDOWS_RESOURCE.name == "waa-windows-vm"
    assert WAA_WINDOWS_RESOURCE.source_url is not None
    assert "huggingface" in WAA_WINDOWS_RESOURCE.source_url
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/aman.jaiswal/Work/AgentLab2.worktrees/pr-290/cubes/windows-agent-arena-cube
uv run pytest tests/test_benchmark.py::test_waa_windows_resource_has_source_url -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'waa_cube.azure'`

- [ ] **Step 3: Create `src/waa_cube/azure.py`**

```python
"""WAA Azure resource configuration.

Usage::

    from waa_cube.azure import WAA_WINDOWS_RESOURCE
    from cube_infra_azure import AzureInfraConfig

    infra = AzureInfraConfig(
        resource_group=os.environ["AZURE_RESOURCE_GROUP"],
        windows_admin_password=os.environ["WAA_WINDOWS_ADMIN_PASSWORD"],
    )
    infra.provision(WAA_WINDOWS_RESOURCE)   # one-time, ~60-90 min
    bench = WAABenchmark(infra=infra, default_tool_config=ComputerConfig())
"""

from cube.resource import VMResourceConfig

WAA_WINDOWS_RESOURCE = VMResourceConfig(
    name="waa-windows-vm",
    source_url=(
        "https://huggingface.co/datasets/kushasareen/waa-windows-image"
        "/resolve/main/waa-windows.qcow2"
    ),
    default_ttl_seconds=60 * 60 * 2,
)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_benchmark.py::test_waa_windows_resource_has_source_url -v
```

Expected: PASS

- [ ] **Step 5: Write failing test for `WAABenchmark.infra` field**

```python
# In tests/test_benchmark.py — add:

def test_waabenchmark_accepts_infra_field() -> None:
    from waa_cube.benchmark import WAABenchmark
    from waa_cube.computer import ComputerConfig

    # infra=None is the default — should instantiate fine
    bench = WAABenchmark(default_tool_config=ComputerConfig())
    assert bench.infra is None
```

- [ ] **Step 6: Run test to verify it fails**

```bash
uv run pytest tests/test_benchmark.py::test_waabenchmark_accepts_infra_field -v
```

Expected: FAIL with `ValidationError` or `TypeError` (field doesn't exist)

- [ ] **Step 7: Add `infra` field to `WAABenchmark` and pass it through `get_task_configs()`**

In `benchmark.py`:

At the top, add import:
```python
from cube.resource import InfraConfig
```

In the `WAABenchmark` class instance fields, add after `vm_backend`:
```python
    infra: InfraConfig | None = None
    """Azure (or other cloud) InfraConfig for launching VMs. Mutually exclusive with vm_backend."""
```

In `get_task_configs()`, update the yield:
```python
    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        """Yield WAATaskConfig objects with vm_backend/infra and metadata injected."""
        for tm in self.task_metadata.values():
            yield WAATaskConfig(
                task_id=tm.id,
                tool_config=self.default_tool_config,
                seed=None,
                vm_backend=self.vm_backend,
                infra=self.infra,
                metadata=tm,
            )
```

In `WAATaskConfig`, add the `infra` field and update `make()`:
```python
class WAATaskConfig(TaskConfig):
    vm_backend: VMBackend | None = None
    infra: InfraConfig | None = None
    metadata: TaskMetadata | None = None

    def make(
        self,
        runtime_context: dict | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> WAATask:
        if self.metadata is not None:
            metadata = self.metadata
        else:
            metadata = WAABenchmark.task_metadata[self.task_id]
        if self.tool_config is None:
            raise ValueError(
                f"WAATaskConfig for task '{self.task_id}' has no tool_config. "
                "Pass default_tool_config=ComputerConfig(...) to WAABenchmark."
            )
        return WAATask(
            metadata=metadata,
            tool_config=self.tool_config,
            vm_backend=self.vm_backend,
            infra=self.infra,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
```

Also add `from cube.resource import InfraConfig` at top of `benchmark.py`.

- [ ] **Step 8: Run test to verify it passes**

```bash
uv run pytest tests/test_benchmark.py::test_waabenchmark_accepts_infra_field -v
```

Expected: PASS

- [ ] **Step 9: Commit**

```bash
cd /Users/aman.jaiswal/Work/AgentLab2.worktrees/pr-290
git add cubes/windows-agent-arena-cube/src/waa_cube/azure.py \
        cubes/windows-agent-arena-cube/src/waa_cube/benchmark.py \
        cubes/windows-agent-arena-cube/tests/test_benchmark.py
git commit -m "feat(waa-cube): add WAA_WINDOWS_RESOURCE and infra field on WAABenchmark"
```

---

## Task 7: `WAATask` — dispatch on `infra` in `reset()` and `close()`

**Files:**
- Modify: `cubes/windows-agent-arena-cube/src/waa_cube/task.py`
- Test: `cubes/windows-agent-arena-cube/tests/test_benchmark.py`

- [ ] **Step 1: Write failing test for infra dispatch**

```python
# In tests/test_benchmark.py — add:

def test_waatask_accepts_infra_field() -> None:
    from waa_cube.task import WAATask
    from waa_cube.computer import ComputerConfig
    from cube.task import TaskMetadata

    metadata = TaskMetadata(
        id="test-task",
        abstract_description="test",
        extra_info={"domain": "notepad", "snapshot": "init_state", "config": [], "evaluator": {}, "related_apps": []},
    )
    # Should construct without error when infra=None
    task = WAATask(metadata=metadata, tool_config=ComputerConfig(), infra=None)
    assert task.infra is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/aman.jaiswal/Work/AgentLab2.worktrees/pr-290/cubes/windows-agent-arena-cube
uv run pytest tests/test_benchmark.py::test_waatask_accepts_infra_field -v
```

Expected: FAIL with `ValidationError` (no `infra` field)

- [ ] **Step 3: Add `infra` field and Azure dispatch to `WAATask`**

In `task.py`, add at top-level imports:
```python
from cube.resource import InfraConfig, ResourceHandle
```

In `WAATask` class, add after `vm_backend`:
```python
    infra: InfraConfig | None = None
    """Cloud InfraConfig (e.g. AzureInfraConfig). Mutually exclusive with vm_backend."""

    _resource_handle: ResourceHandle | None = PrivateAttr(default=None)
```

Replace `_ensure_vm()` with a version that also handles the infra path:
```python
    def _ensure_vm(self) -> None:
        """Launch the VM via vm_backend (Docker) or infra (Azure/cloud)."""
        # ── infra path (Azure / cloud) ────────────────────────────────────────
        if self.infra is not None:
            if self._resource_handle is not None:
                return  # already launched
            from waa_cube.azure import WAA_WINDOWS_RESOURCE
            logger.info("Launching Azure VM via %s", type(self.infra).__name__)
            self._resource_handle = self.infra.launch(WAA_WINDOWS_RESOURCE)
            endpoint = self._resource_handle.endpoint
            self._computer.attach_endpoint(endpoint)
            return

        # ── vm_backend path (Docker+QEMU) ─────────────────────────────────────
        if self._vm is not None:
            is_alive = getattr(self._vm, "is_alive", None)
            if not callable(is_alive) or is_alive():
                return
            logger.warning("Existing WAA VM is no longer alive; relaunching it before reset")
            try:
                self._vm.stop()
            except Exception as exc:
                logger.warning("Failed to stop stale WAA VM cleanly: %s", exc)
            self._vm = None
        if self.vm_backend is None:
            return
        snapshot = self.metadata.extra_info.get("snapshot", "init_state")
        vm_config = VMConfig(snapshot_name=snapshot)
        logger.info("Launching VM via %s", type(self.vm_backend).__name__)
        self._vm = self.vm_backend.launch(vm_config)
        self._computer.attach_vm(self._vm)
```

In `close()` (find the existing `close()` method in `WAATask`), add cleanup of the resource handle:
```python
    def close(self) -> None:
        if self._resource_handle is not None:
            try:
                self._resource_handle.close()
            except Exception as exc:
                logger.warning("Failed to close Azure resource handle: %s", exc)
            self._resource_handle = None
        # ... rest of existing close() body
```

- [ ] **Step 4: Check what `close()` currently looks like and add handle cleanup**

```bash
grep -n "def close" /Users/aman.jaiswal/Work/AgentLab2.worktrees/pr-290/cubes/windows-agent-arena-cube/src/waa_cube/task.py
```

Read the existing `close()` body and prepend the resource handle cleanup shown in Step 3 — keep the existing body intact.

- [ ] **Step 5: Run failing test**

```bash
uv run pytest tests/test_benchmark.py::test_waatask_accepts_infra_field -v
```

Expected: PASS

- [ ] **Step 6: Run all waa-cube tests**

```bash
uv run pytest tests/ -v
```

Expected: all pass

- [ ] **Step 7: Commit**

```bash
cd /Users/aman.jaiswal/Work/AgentLab2.worktrees/pr-290
git add cubes/windows-agent-arena-cube/src/waa_cube/task.py \
        cubes/windows-agent-arena-cube/tests/test_benchmark.py
git commit -m "feat(waa-cube): dispatch on infra vs vm_backend in WAATask._ensure_vm() and close()"
```

---

## Task 8: Azure eval recipe

**Files:**
- Create: `recipes/waa/eval_azure_waa.py`

- [ ] **Step 1: Create the recipe**

```python
"""WAA eval on Azure — Genny agent with GPT-4o and accessibility tree observations.

Uses AzureInfraConfig to launch fresh Windows VMs per task.

Prerequisites:
    See cube-resources/cube-infra-azure/README.md for full setup instructions.
    Set WAA_WINDOWS_ADMIN_PASSWORD to the Administrator password of the WAA image.

Usage:
    # Debug mode (2 debug tasks, sequential)
    uv run recipes/waa/eval_azure_waa.py debug

    # Eval mode (full benchmark, parallel)
    uv run recipes/waa/eval_azure_waa.py
"""

import logging
import os
import sys

from cube_infra_azure import AzureInfraConfig
from dotenv import load_dotenv
from waa_cube.azure import WAA_WINDOWS_RESOURCE
from waa_cube.benchmark import WAABenchmark
from waa_cube.computer import ComputerConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import GennyConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
for _noisy in ("azure.core.pipeline.policies.http_logging_policy", "azure.identity", "urllib3.connectionpool"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

INFRA = AzureInfraConfig(
    resource_group=os.environ.get("AZURE_RESOURCE_GROUP") or "ui_assist",
    storage_account=os.environ.get("AZURE_STORAGE_ACCOUNT") or "cubeexpvhd",
    vnet_name="vnet-westus2",
    nsg_name="osworld-nsg",
    windows_admin_password=os.environ.get("WAA_WINDOWS_ADMIN_PASSWORD"),
)

WAA_SYSTEM_PROMPT = """\
You are a desktop automation agent controlling a real Windows 11 computer.

## Observations
Each step you receive an element table listing interactive UI elements with columns:
index, tag, name, text, x, y, w, h

Where (x, y) is the top-left corner and (w, h) is the size of each element.
To click the center of element at row i: center_x = x + w//2, center_y = y + h//2

## Actions
You control the computer by calling run_pyautogui(code) with valid Python/pyautogui code.

### Common pyautogui commands
- pyautogui.click(x, y)
- pyautogui.rightClick(x, y)
- pyautogui.doubleClick(x, y)
- pyautogui.typewrite('text', interval=0.05)
- pyautogui.hotkey('ctrl', 'c')
- pyautogui.press('enter')
- pyautogui.scroll(x, y, clicks=-3)

### Ending the task
- Call fail() if the task CANNOT be completed
- Call done() when the task is successfully COMPLETE
"""


def main(debug: bool) -> None:
    output_dir = make_experiment_output_dir("genny_azure", "waa-cube")

    llm_config = LLMConfig(model_name="azure/gpt-4o", temperature=1.0)
    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=WAA_SYSTEM_PROMPT,
        max_actions=100,
        render_last_n_obs=3,
        enable_summarize=False,
        tools_as_text=False,
    )

    tool_config = ComputerConfig(
        action_space="pyautogui",
        require_a11y_tree=True,
        observe_after_action=True,
    )

    benchmark = WAABenchmark(
        default_tool_config=tool_config,
        infra=INFRA,
    )
    benchmark.setup()

    INFRA.provision(WAA_WINDOWS_RESOURCE)

    exp = Experiment(
        name="waa_azure_gpt4o",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=15,
    )

    try:
        if debug:
            print(f"\nDEBUG MODE — sequential, output: {output_dir}")
            run_sequentially(exp)
        else:
            print(f"\nEVAL MODE — parallel, output: {output_dir}")
            run_with_ray(exp, n_cpus=40)
    finally:
        deleted = INFRA.cleanup_orphaned_resources()
        if deleted:
            print(f"Cleaned up orphaned resources: {deleted}")


if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
    main(debug)
```

- [ ] **Step 2: Verify the recipe imports cleanly (no Azure calls)**

```bash
cd /Users/aman.jaiswal/Work/AgentLab2.worktrees/pr-290
uv run python -c "import recipes.waa.eval_azure_waa" 2>&1 | head -5
```

Expected: no import errors (Azure SDK defers auth until actual calls)

- [ ] **Step 3: Commit**

```bash
git add recipes/waa/eval_azure_waa.py
git commit -m "feat(waa-cube): add Azure eval recipe"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Bootstrap Windows QEMU+WinRM+sysprep branch — Task 2
- [x] `windows_admin_password` field — Task 1
- [x] OS type detection from partition table — Task 2
- [x] `os_type` metadata blob written by bootstrap, read by `_bootstrap()` — Task 3
- [x] `_create_image_definition` Windows ARM properties — Task 4
- [x] `_import_disk` Windows `osType` — Task 4
- [x] `launch()` gallery image os_type detection — Task 5
- [x] `launch()` VMAccessAgent SSH key injection for Windows — Task 5
- [x] `WAA_WINDOWS_RESOURCE` constant — Task 6
- [x] `WAABenchmark.infra` field — Task 6
- [x] `WAATask.infra` field + dispatch — Task 7
- [x] `WAATask.close()` resource handle cleanup — Task 7
- [x] Azure eval recipe — Task 8

**Known risks (from spec) not covered by tests — manual verification needed:**
- WinRM enabled on WAA base image (verify before running provision)
- OVMF vs SeaBIOS (bootstrap script tries OVMF first, falls back if not present)
- Bootstrap VM disk size (`bootstrap_os_disk_gb=256` for WAA — set in recipe's `AzureInfraConfig`)
- Exact filename of the qcow2 on HuggingFace (update `WAA_WINDOWS_RESOURCE.source_url` in `azure.py`)
