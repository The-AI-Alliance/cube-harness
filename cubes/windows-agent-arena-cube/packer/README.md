# WAA Windows Image Builder (Packer)

This directory produces the prepared WAA Windows qcow2 that both `LocalInfraConfig`
(local QEMU) and `AzureInfraConfig` (cloud) consume.

The prepared image ships with:

| Capability | Why |
|---|---|
| OpenSSH Server enabled (sshd auto-start, firewall rule, port 22) | Runtime SSH access from the harness to the guest. |
| Azure VM Agent (`WindowsAzureGuestAgent`) | Azure applies `os_profile.admin_password` via this agent on first boot. Without it, Windows VMs never become reachable. |
| `administrators_authorized_keys` pre-baked | `launch()` does not inject per-VM SSH keys for Windows — the matching private key must be on the caller at `ssh_privkey_path`. |
| Sysprepped / Generalized | Azure can clone the image into fresh VMs with new hostnames, SIDs, and credentials. |

## Prerequisites

```bash
# Packer (one-time on build host)
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt-get update && sudo apt-get install -y packer

# QEMU + OVMF
sudo apt-get install -y qemu-system-x86 qemu-utils ovmf

# In this dir:
packer init .
```

## One-time base-image prep: enable WinRM

Packer connects to the base qcow2 via WinRM, but the image shipped at
`kushasareen/waa-windows-image/data.img` does **not** have WinRM enabled and
ships with a blank password on the `Docker` admin user (which blocks WinRM
network auth). `bootstrap_winrm.py` fixes both in one idempotent step:

```bash
export WAA_BUILD_ADMIN_PASSWORD='ChooseAComplexPassword123!'
make bootstrap-winrm
# or directly:
uv run python packer/bootstrap_winrm.py --admin-password "$WAA_BUILD_ADMIN_PASSWORD"
```

Use the same value later as `PKR_VAR_admin_password` when running `make
build-image` — that's how Packer authenticates to the now-WinRM-enabled VM.

What it does: boots the base image under QEMU (overlay, not in-place), waits
for the WAA Flask guest agent on guest port 5000, POSTs to `/setup/execute` to
run `Set-LocalUser -Name Docker -Password …` + `Enable-PSRemoting` + firewall
rules + basic/unencrypted auth, requests a clean shutdown, and finally
`qemu-img commit`s the overlay back into the base. If any step fails the
overlay is discarded and the base is untouched.

Refuses to run unless a backup exists under `~/.cube/images/backups/`. Pass
`--skip-backup-check` to override, or `--keep-overlay` to dry-run without
committing.

## Building

The Packer build has two pieces of external state Packer itself doesn't manage:
a writable copy of OVMF_VARS per build, and a running `swtpm` daemon (Windows 11
requires TPM 2.0). The `run.sh` wrapper sets both up, runs Packer, and cleans up
on exit.

```bash
export PKR_VAR_admin_password='<Docker user password from WAA upstream>'
make build-image          # from cubes/windows-agent-arena-cube/
# or directly:
packer/run.sh
```

Override the defaults via env vars if needed:

```bash
SOURCE_QCOW2=/some/other.qcow2 \
SSH_PUBKEY=/some/other.pub \
PKR_VAR_admin_password='…' \
packer/run.sh
```

Output: `packer/output-waa-prepared/waa-windows-prepared.qcow2`.

## Uploading the result

After a successful build:

```bash
# Replace the local cache:
cp output-waa-prepared/waa-windows-prepared.qcow2 $HOME/.cube/images/waa-windows-vm.qcow2

# Re-publish to HuggingFace:
python ../scripts/upload_image.py \
  --image-path output-waa-prepared/waa-windows-prepared.qcow2
```

## Design notes

* **Why not start from a Windows install ISO?** The WAA image ships with pre-baked
  apps, pre-created user state, and the Flask guest agent that the harness
  depends on. Rebuilding from ISO would re-create all of that. We only need to
  augment the existing image.
* **Why bake the SSH key into the image instead of injecting per-launch?**
  Azure's `os_profile` for Windows doesn't accept SSH keys the way it does for
  Linux. Using `CustomScriptExtension` to inject per-launch would add ~60s to
  every VM start. Since the harness is single-user in practice, a build-time
  key is simpler. The `ssh_pubkey_path` variable keeps the choice of key
  configurable per build.
* **Why sysprep `/mode:vm`?** Skips hardware driver generalization since we're
  moving between identical QEMU/Hyper-V VMs. Shaves ~3 min off the build.
