# WAA on Azure — First End-to-End Run

**Audience**: An agent or engineer picking up this branch on a fresh laptop to run the first real Azure-backed WAA task. Assumes `az login` works and this machine has the repos cloned.

If you are continuing work from the Azure VM where this branch was built, you already have everything. This doc is for **the laptop-side continuation**.

---

## Hard constraints — do not "fix" these

Read before making any changes. These are deliberate design decisions, not oversights:

1. **The image is "Specialized", not "Generalized"**. `WAA_WINDOWS_RESOURCE.specialized = True`. This is correct — the agent sandbox needs a deterministic state (same Docker user, same installed apps, same profile) across every launch. Do NOT change to `specialized=False`, do NOT add a sysprep step back.
2. **There is no `sysprep.ps1` in the Packer build** by design. The WAA image has per-user VS Code that blocks `sysprep /generalize` anyway. The file `packer/scripts/sysprep.ps1` exists for reference but is NOT wired into `waa-windows.pkr.hcl`. Leave it that way.
3. **OpenSSH is side-loaded from GitHub, not installed via `Add-WindowsCapability`**. Windows Update is broken in this base image, which makes the capability route hang indefinitely. Do NOT try to "modernize" `install-openssh-server.ps1`.
4. **`configure-autologon.ps1` uses C# P/Invoke to `LsaStorePrivateData`** — not an overreach. The LSA secret takes precedence over the registry DefaultPassword key on boot. Without the P/Invoke, Docker auto-login silently fails and the guest agent never starts. Don't "simplify".
5. **All PS scripts are strict ASCII**. Em-dash / curly-quote characters get mangled by WinRM's XML encoding and produce parser errors. If you add new scripts, keep them ASCII-only.
6. **The axtree (`/accessibility_tree`) endpoint is known to hang** after ~5 min in the guest. This is a pywinauto/UIA bug inside the WAA Flask agent, **NOT a provisioning issue**. Do not try to fix it as part of this Azure PR. Tasks may hang; VM provisioning is still successful.

**Success criterion for this branch**: an Azure VM launches from `WAA_WINDOWS_RESOURCE`, SSH works against it with `~/.ssh/id_ed25519`, `curl http://127.0.0.1:<tunneled-port>/screenshot` returns a non-empty PNG. Task execution completing is explicitly out of scope.

---

## Step 1: Pull the branches

```bash
cd /path/to/cube-standard
git fetch origin
git checkout waa-azure-windows-specialized    # existing on remote

cd /path/to/cube-harness
git fetch origin
git checkout waa-azure-packer-pipeline         # existing on remote, 2 commits

# uv sync — picks up cube-standard via path source
cd /path/to/cube-harness
uv sync --extra waa
```

Verify the updated fields are in place:

```bash
uv run python -c "
from waa_cube.azure import WAA_WINDOWS_RESOURCE
print('os_type:     ', WAA_WINDOWS_RESOURCE.os_type)       # expect: windows
print('specialized: ', WAA_WINDOWS_RESOURCE.specialized)   # expect: True
print('uefi/tpm:    ', WAA_WINDOWS_RESOURCE.uefi, WAA_WINDOWS_RESOURCE.tpm)  # expect: True True
print('source_url:  ', WAA_WINDOWS_RESOURCE.source_url)
# expect: https://huggingface.co/datasets/kushasareen/waa-windows-image/resolve/main/waa-windows-prepared.qcow2
"
```

---

## Step 2: Transfer two secrets from the build VM

Two files from the build VM (`/home/azureuser/...`) must be copied to your laptop:

1. **`~/.cube/waa-build-admin-password.txt`** — the Docker-user password baked into the image. SSH and (Packer rebuilds) both need this. File is 25 chars, 0600 perms.
2. **`~/.ssh/id_ed25519`** — the private half of the SSH keypair whose pubkey is at `C:\ProgramData\ssh\administrators_authorized_keys` inside every VM launched from the prepared image.

How to transfer (from laptop, once VPN is up):

```bash
# Adjust hostname / bastion as needed for your setup:
scp azureuser@<build-vm>:/home/azureuser/.cube/waa-build-admin-password.txt ~/.cube/
scp azureuser@<build-vm>:/home/azureuser/.ssh/id_ed25519 ~/.ssh/waa_id_ed25519
chmod 600 ~/.cube/waa-build-admin-password.txt ~/.ssh/waa_id_ed25519
```

If the build VM is unreachable from your laptop, read the files over the terminal (they're short enough) and recreate them locally.

---

## Step 3: Azure auth

```bash
az login                                                 # laptop, VPN on
az account set --subscription aeb958d3-a614-450e-94bc-88f284dc0664
# (This is the subscription the osworld/ui_assist RG lives in. If your fleet uses a
# different RG, change the recipe hardcode — see Step 4.)
```

Verify:

```bash
az group show --name ui_assist --query "{name:name, location:location}" -o table
# Expect: ui_assist, westus2
```

---

## Step 4: Configure the recipe

Open `cube-harness/recipes/waa/eval_azure_waa.py`. Current hardcodes:

```python
INFRA = AzureInfraConfig(
    resource_group=os.environ.get("AZURE_RESOURCE_GROUP") or "ui_assist",
    storage_account=os.environ.get("AZURE_STORAGE_ACCOUNT") or "cubeexpvhd",
    vnet_name="vnet-westus2",
    nsg_name="osworld-nsg",
    vm_size="Standard_D8s_v3",
    windows_admin_password=_ADMIN_PASSWORD,
)
```

**Adjust only if necessary**:
- `vnet_name` / `nsg_name` must exist in the resource group. If `osworld-nsg` is the wrong name, `az network nsg list -g ui_assist` shows the correct one.
- `vm_size` needs 8 cores AND TrustedLaunch support. `D8s_v3` / `D8s_v5` / `D8ads_v5` all qualify in westus2.

**DO NOT** add an `image_name_suffix` field — the osworld recipe references that, but it's not a real `AzureInfraConfig` field. That's a known bug in the osworld recipe that survives only because it's never actually passed through Pydantic validation.

Also: **for Specialized images the SSH pubkey is baked into the image, not injected via os_profile**. `ssh_pubkey_path` on `AzureInfraConfig` is irrelevant for this recipe — don't worry if you see it unset.

---

## Step 5: Run the debug recipe

```bash
export WAA_WINDOWS_ADMIN_PASSWORD="$(cat ~/.cube/waa-build-admin-password.txt)"
# AzureInfraConfig requires windows_admin_password even when specialized=True
# and it's ignored at launch — it's still a historical field requirement.
# Once the code lands and is tested, this will be safe to remove.

cd /path/to/cube-harness
uv run python recipes/waa/eval_azure_waa.py debug
```

**What should happen (and how long each phase takes):**

1. **provision()** — first time only, ~30-60 min:
   - Launches an Ubuntu bootstrap VM in `ui_assist`
   - Bootstrap VM downloads 16 GB from HF (~2-5 min on Azure bandwidth)
   - Runs `qemu-img convert -O vpc -o subformat=fixed,force_size` on the qcow2 (~5 min)
   - azcopy uploads the VHD to the blob storage account (~10 min)
   - Imports as managed disk, creates gallery image definition + version (~5 min)
   - Bootstrap VM is destroyed; sentinel blob is written so re-runs skip this entire phase
2. **launch()** — ~5-10 min:
   - Creates NIC, public IP, NSG rules
   - Issues `virtual_machines.begin_create_or_update` with TrustedLaunch + vTPM + secureBoot, **no `os_profile`** (specialized)
   - Waits for SSH to respond as `Docker` user using `~/.ssh/waa_id_ed25519`
   - Opens SSH tunnel local:<free-port> → VM:5000
3. **task.reset()** — ~30s: uploads setup files, runs setup_controller steps
4. **task.step()** — **WILL LIKELY HANG FOR 5 MIN** on the first `get_observation()` call because of the axtree issue. This is expected. The VM is fine; only the accessibility tree walk inside the guest is stuck.

**Success indicators** (not task completion!):
- `launch()` returns without error
- Log line: `SSH available as Docker@<ip>`
- `ssh -i ~/.ssh/waa_id_ed25519 Docker@<ip> 'Get-Service sshd'` works
- `curl http://127.0.0.1:<tunneled-port>/screenshot` returns a PNG

At that point, the provisioning pipeline is proven end-to-end. Merge the PRs.

---

## Likely failure modes and their fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `AuthorizationFailed` at `resource_groups.get` | Wrong subscription, or account lacks Contributor on `ui_assist` | `az account show` to confirm sub; ask admin for role |
| `Image definition already exists but is Generalized` | Old image from a prior provision attempt is cached | `az sig image-version delete -r <gal> -i waa-windows-vm -e 1.0.0 -g ui_assist`, then re-run |
| `SKU Standard_D8s_v3 is not available in region westus2 for Trusted Launch` | Region quota / SKU deprecation | Try `Standard_D8s_v5` or check `az vm list-skus -l westus2 --zone` |
| `SSH connection refused` after launch | NSG doesn't allow port 22 inbound | Add rule: `az network nsg rule create -g ui_assist --nsg-name <nsg> -n AllowSSH --priority 100 --destination-port-ranges 22 --protocol Tcp --access Allow` |
| `wait_for_ssh` times out | VM booted but OpenSSH inside guest didn't start — likely AutoLogon broken | Confirm with `az vm run-command invoke ... --command-id RunPowerShellScript --scripts "query user; Get-Service sshd \| fl"`. If no one is logged in, the LSA fix from `configure-autologon.ps1` didn't take during the Packer build — rebuild the image. |
| Hangs at `get_observation()` | axtree issue (see Hard Constraints) | Not your problem. Time out and declare provisioning success. |

---

## If something breaks that's not in the table

Before "fixing" anything, confirm the symptom reproduces and check the cube-infra-azure log output. The Packer build on the source VM is reproducible via `make build-image` — if the image itself has regressed, re-run the build rather than patching runtime.

Background resources:
- `packer/README.md` — Packer pipeline overview
- `HANDOFF.md` (in the build VM's `/home/azureuser/.cube/handoff/`) — overnight build journey
- https://huggingface.co/datasets/kushasareen/waa-windows-image/resolve/main/waa-windows-prepared.qcow2 — the uploaded image

Branches:
- `The-AI-Alliance/cube-standard @ waa-azure-windows-specialized` (1 commit, based on `fix/waa-guest-agent-methods`)
- `The-AI-Alliance/cube-harness @ waa-azure-packer-pipeline` (2 commits, based on `fix/waa-cube-communication`)

---

## Things explicitly out of scope for this branch

- Fixing the axtree hang
- Re-publishing the image under a different name or org (the HF URL is fine)
- Converting Specialized → Generalized
- Making the pipeline work without a password file (per-launch key injection is a reasonable future task; not needed for first run)
- Parallelism or production scaling concerns

---

## What I'm uncertain about (read before "fixing")

I built and tested this pipeline up to and including the local-QEMU smoke test.
Everything past that (Azure provision + launch) is **unvalidated code paths**.
Here are the specific things I don't know for sure:

### Very likely correct, but untested on Azure

1. **`_WINDOWS_BOOTSTRAP_SCRIPT` on the Ubuntu bootstrap VM**. The script
   `qemu-img convert -O vpc -o subformat=fixed,force_size` on a 16 GB qcow2 →
   fixed VHD is standard, but I haven't actually run it on the HF-hosted image
   in Azure. If the conversion fails, check the bootstrap VM's
   `/var/log/cube-bootstrap.log` via SSH (IP printed in `_launch_bootstrap_vm`
   log).
2. **TrustedLaunch feature flag shape in the image definition**. I'm using
   `features=[{"name": "SecurityType", "value": "TrustedLaunchSupported"}]`
   in `_create_image_definition`. Azure API has evolved — this may need to be
   `TrustedLaunchAndConfidentialVmSupported`, or the feature may be implicit
   from `hyper_v_generation=V2`. If `gallery_images.begin_create_or_update`
   rejects the spec, try omitting the `features` array and see if the image
   still accepts TrustedLaunch VMs at launch time.
3. **`vnet_name="vnet-westus2"` and `nsg_name="osworld-nsg"`**. I inherited
   these from the osworld recipe without verifying they exist in `ui_assist`.
   Run `az network vnet list -g ui_assist` and `az network nsg list -g ui_assist`
   to confirm. If they don't exist, set the fields explicitly or let the
   `_autodiscover` validator pick the only available one in the RG.
4. **`image_name_suffix` in the osworld recipe**. I verified it's **not** a
   field on `AzureInfraConfig` (grep returned 0 matches in the source). Yet
   `eval_azure_osworld.py` passes it as a kwarg. Either (a) Pydantic accepts
   and drops it silently, or (b) the osworld recipe would actually fail if
   anyone ran it. I noted "don't copy this field into your recipe" but I'm
   only 70% sure the osworld recipe even works as-is — someone may have
   updated things without me noticing.

### Meaningful unknowns — might need iteration

5. **Specialized + `os_profile` = None interaction with Azure ARM**. My
   `launch()` code OMITS the `os_profile` key entirely for specialized Windows
   VMs (Azure docs say this is the right thing). But Azure's CreateVM API
   occasionally returns inscrutable 400s when a key it expected is missing.
   If `virtual_machines.begin_create_or_update` errors with a message mentioning
   `os_profile`, try adding `"os_profile": {}` (empty dict) instead of omitting.
6. **`windows_admin_password` requirement for Specialized**. I left a guard in
   `launch()` that still requires `windows_admin_password` for Windows, even
   when `specialized=True`. That's probably unnecessary — Azure doesn't read
   it for specialized images. The guard exists "just in case" and doesn't
   break anything, but if it annoys future you, it's safe to relax.
7. **`wait_for_ssh` user order**. I set `primary_user = windows_admin_username`
   ("cubeadmin" by default) and `fallback_users = ["Administrator"]`. For the
   WAA specialized image, the actual user is `Docker`. The SSH wait will fail
   on `cubeadmin` first, then `Administrator`, then time out — UNLESS I also
   need to add `Docker` to the fallback list. If you see
   `SSH not available after 900s` and the VM is up, SSH as `Docker` manually
   to confirm, then add `"Docker"` to the fallback list on [azure.py:870](cube-standard/cube-resources/cube-infra-azure/src/cube_infra_azure/azure.py#L870)
   or override via a `waa_admin_username` field. This is the most likely real
   snag.
8. **The HF-hosted qcow2 being readable from the bootstrap VM**. I used
   `kushasareen/waa-windows-image` (private dataset, write token is yours).
   The bootstrap VM does `wget <source_url>`. HF public-read works for most
   datasets but if yours is private, the VM would need an HF token. Check
   `curl -I <source_url>` from a non-authenticated context — if it returns
   401, switch the dataset to public or add token handling to the bootstrap
   script.
9. **The Packer image's boot behavior on Hyper-V vs KVM**. We smoke-tested
   under KVM. Azure runs Hyper-V Gen 2. The qcow2 should behave identically
   (same UEFI/TPM shape), but device names and drivers sometimes differ. If
   the Azure VM boots but no one logs in (AutoLogon never fires), it may be
   a driver-load ordering issue on Hyper-V — harder to fix, would require
   testing on Azure directly.
10. **Gallery image capacity**. Each gallery image version is ~30 GB
    Standard_LRS storage. The existing gallery in `ui_assist` may be near
    quota. Symptom: `_create_image_version` fails with quota error. Fix: delete
    old versions via `az sig image-version list -r <gal> -g ui_assist`.

### Things I'm fairly confident about

- The image itself (the 16 GB on HF) is good — smoke-tested end-to-end under
  QEMU. SSH + guest agent both come up, auto-login works.
- The `cube-standard` changes (`os_type`, `specialized`, plumbing through
  `AzureInfraConfig`) are minimal and backwards-compatible. OSWorld path is
  unchanged.
- The Packer pipeline is reproducible. Running `make build-image` again from
  the source VM would produce a byte-identical image (modulo timestamps).

### How to debug if "it just doesn't work"

Priority order:
1. Does `az login` get you a valid subscription? `az account show`.
2. Does `AzureInfraConfig(resource_group="ui_assist")` construct without
   error? Try it in `uv run python -c ...` before running the recipe.
3. Does `infra.provision(WAA_WINDOWS_RESOURCE)` complete? It prints the
   bootstrap VM's public IP — SSH into it with your `ssh_privkey_path` as user
   `azureuser` and `tail -f /var/log/cube-bootstrap.log` to watch the
   download/convert/upload in real time.
4. Does the gallery image exist after provision? `az sig image-version show -r <gal> -i waa-windows-vm -e 1.0.0`.
5. Does `infra.launch(WAA_WINDOWS_RESOURCE)` succeed? If it does but SSH
   times out, the AutoLogon-or-user issue from #7 above is the most likely
   cause. SSH as `Docker` manually to confirm the VM is actually running.

If stuck: the build VM at `/home/azureuser/workspace` still has all the state,
including the `~/.cube/images/backups/waa-windows-prepared-lsa-v9.qcow2` (the
exact image on HF) and a working LocalInfraConfig setup. The full history of
how each issue was diagnosed is in `/home/azureuser/.cube/handoff/HANDOFF.md`
on that VM.
