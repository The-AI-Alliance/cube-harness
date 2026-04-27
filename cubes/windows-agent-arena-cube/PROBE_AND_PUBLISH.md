# Probe-and-Publish runbook (autonomous mode)

User left me running this end-to-end while they're away (2026-04-25 ~13:45 UTC). The Stage 2 data.img rebuild (LibreOffice fix) just finished — see `LIBREOFFICE_REBUILD.md` for the journey here.

This file is the source of truth for what's next. I check it on each loop iteration and execute the next pending step.

---

## Current state (as of writing)

- ✅ Stage 2 rebuild complete (09:43:55 UTC) — `data.img` 18 GB actual / 30 GB sparse at `WindowsAgentArena/src/win-arena-container/vm/storage/data.img`
- ✅ **Backup made** at `~/.cube/waa/storage.NEW-libreoffice-2026-04-25/data.img` (18 GB actual, sparse). This is the "never-do-this-again" copy.
- 🔄 dockurr container `winarena` boot in progress (`/tmp/waa-probe-boot.log`) for the probe
- ⏳ Monitor `bharufp7c` armed — fires on `Windows Arena Server is ready` or `Shutdown|exit=|Error|...`

Old backup (pre-LibreOffice-fix) still at `~/.cube/waa/storage.bak-2026-04-25/`.

---

## Workflow steps

### Step 1 — Boot probe + inventory verification

Goal: confirm LibreOffice is actually installed, plus no regression in the other 7 apps.

1. Wait for monitor `bharufp7c` to fire `Windows Arena Server is ready` (boot ~5–10 min after container start).
2. Once Flask is up, run inventory probe via `docker exec winarena curl http://<vm-ip>:5000/execute -X POST -d '...'`.
   - The Flask server runs in the guest. From the container, the VM is reachable on `host.lan` or the container's bridge — easiest: use the same probe that Stage 2's `entry_setup.sh` used. Look in `WindowsAgentArena/src/win-arena-container/vm/setup/entry_setup.sh` for the actual URL.
3. Probe payload (PowerShell):
   ```powershell
   $paths = @{
     'LibreOffice'  = 'C:\Program Files\LibreOffice\program\soffice.exe';
     'VLC'          = 'C:\Program Files\VideoLAN\VLC\vlc.exe';
     'GIMP'         = 'C:\Program Files\GIMP 2\bin\gimp-2.10.exe';
     'Chrome'       = 'C:\Program Files\Google\Chrome\Application\chrome.exe';
     'Edge'         = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe';
     '7zip'         = 'C:\Program Files\7-Zip\7z.exe';
     'Git'          = 'C:\Program Files\Git\bin\git.exe';
     'Thunderbird'  = 'C:\Program Files\Mozilla Thunderbird\thunderbird.exe';
   }
   foreach ($k in $paths.Keys) { '{0,-14} {1} {2}' -f $k, (Test-Path $paths[$k]), $paths[$k] }
   ```
4. Verify LibreOffice version: `(Get-WmiObject Win32_Product -Filter "Name like 'LibreOffice%'").Version` → expect `24.8.2.1`.

**Decision gate**: ALL 8 apps must `Test-Path` true. If any fail (especially LibreOffice) → halt, write a finding to this file under "Failures encountered", and stop the loop with a clear status message for the user.

5. After verification passes: `docker stop winarena` (graceful, will trigger guest shutdown via dockurr). Wait for the container to exit cleanly.

### Step 2 — Promote data.img → cube image cache

```bash
cp --sparse=always /home/azureuser/workspace/WindowsAgentArena/src/win-arena-container/vm/storage/data.img \
   ~/.cube/images/waa-windows-vm.qcow2
```

(Yes the destination has a `.qcow2` extension but the source is raw — that's how the existing pipeline names it. Don't convert format.)

Verify: `ls -lh ~/.cube/images/waa-windows-vm.qcow2`

### Step 3 — Stage 3: bootstrap-winrm

```bash
cd /home/azureuser/workspace/cube-harness/cubes/windows-agent-arena-cube/packer
make bootstrap-winrm
```

This fires up the QEMU VM with the new data.img, drives the keyboard via QEMU monitor to enable WinRM, and snapshots into a "winrm-ready" qcow2. ~10 min.

### Step 4 — Stage 4: build-image (Packer overlay)

```bash
cd /home/azureuser/workspace/cube-harness/cubes/windows-agent-arena-cube/packer
export PKR_VAR_admin_password='<look in conversation history>'
make build-image
```

This is the Packer overlay: OpenSSH-Server, Azure VM Agent, SSH pubkey baked into authorized_keys, autologon, sysprep. ~30–45 min.

**Note on Packer-layer LibreOffice cruft**: REMOVED from `packer/waa-windows.pkr.hcl` mid-run because the WSMan upload of the 346 MB MSI was running at ~5 KB/s (projected ~19 hours). The first Packer attempt was killed after 2.5h with the file still uploading. The pkr.hcl now no longer references LibreOffice — the file `scripts/install-libreoffice.ps1` is still on disk but nothing invokes it.

### Step 5 — Smoke test

```bash
cd /home/azureuser/workspace/cube-harness/cubes/windows-agent-arena-cube/packer
python smoke_test.py
```

End-to-end: boot the prepared image, SSH in, run a simple command, shut down.

**Decision gate**: smoke test must exit 0. If it fails → halt, log to "Failures encountered", stop the loop.

### Step 6 — Push to HuggingFace

Two artifacts to upload to `kushasareen/waa-windows-image`:
1. `data.img` (the rebuilt one from Stage 2) — 18 GB
2. Prepared image (from Stage 4) — also multi-GB

```bash
huggingface-cli upload kushasareen/waa-windows-image \
   ~/.cube/waa/storage.NEW-libreoffice-2026-04-25/data.img data.img

huggingface-cli upload kushasareen/waa-windows-image \
   <packer-output-path> prepared.qcow2
```

(Confirm the exact dataset name and file paths by checking `cubes/windows-agent-arena-cube/README.md` or searching for `huggingface` in the repo before uploading.)

Once upload completes successfully → done. Update this file's "Status" section with completion time and stop the loop.

---

## Loop discipline

- **Fallback heartbeat**: ~25 min via ScheduleWakeup. Monitor wakes are the primary signal.
- **Tasks tracked via TaskCreate** — check TaskList at the start of each iteration.
- **Never act destructively without checking** — if I find unexpected state (e.g. a winarena container I didn't start, a stale build artifact), investigate before deleting.
- **Stop conditions**:
   - Step 6 succeeded → declare victory
   - Any decision gate fails → halt, log, leave a clear summary for the user
   - User interrupts → respect that

---

## Failures encountered

### bootstrap_winrm — multiple iterations to find the working approach (resolved manually)

The script as committed is **not yet idempotent on a fresh data.img**. On the run that worked, several issues had to be debugged + fixed live in the running VM:

1. **120s Flask timeout** — `Enable-PSRemoting` alone takes >120s on Windows 11. Solved by writing a self-contained PS script and launching it detached (now via WMI `Win32_Process.Create`), then polling for a sentinel file.
2. **Powershell cold-start under load >30s** — initial poll used `powershell -Command Test-Path ...` which timed out. Switched to `cmd.exe`-based poll (`if exist ... type ... else echo pending`).
3. **`Set-NetConnectionProfile` hangs forever** on QEMU usermode networking — NLM never classifies the network. Removed the network-private step entirely.
4. **`Set-Item WSMan:\...\AllowUnencrypted $true` hangs the powershell process** ("Not Responding" in tasklist) even when WinRM is up — likely WSMan provider deadlock. Fix applied manually: set GPO registry keys directly:
   ```
   HKLM\SOFTWARE\Policies\Microsoft\Windows\WinRM\Service\AllowUnencryptedTraffic = 1
   HKLM\SOFTWARE\Policies\Microsoft\Windows\WinRM\Service\AllowAutoConfig = 1
   HKLM\SOFTWARE\Policies\Microsoft\Windows\WinRM\Service\IPv4Filter = "*"
   HKLM\SOFTWARE\Policies\Microsoft\Windows\WinRM\Service\IPv6Filter = "*"
   ```
   then `sc stop WinRM && sc start WinRM`. Verified `winrm get winrm/config/service` reports `AllowUnencrypted = true [Source="GPO"]`.
5. **Firewall rule** added via `netsh advfirewall firewall add rule ... profile=any` (PowerShell New-NetFirewallRule was hanging too).
6. **Sentinel** written manually with `echo winrm-ready > C:\winrm-bootstrap.status` to satisfy the bootstrap_winrm.py poller, then overlay committed via `qemu-img commit`.

**TODO before next fresh run**: rewrite the launch script in bootstrap_winrm.py to:
- Set the GPO registry keys via `reg add` BEFORE running Enable-PSRemoting
- Use `netsh advfirewall` (not New-NetFirewallRule) for the firewall rule
- Skip `Set-Item WSMan:` entirely (GPO + Auth Basic via `winrm set winrm/config/service/auth @{Basic="true"}` is enough)

Not blocking the current publish — the committed base image at `~/.cube/images/waa-windows-vm.qcow2` (17.7G) is WinRM-ready.

---

## Status

- 2026-04-25 13:45 UTC — autonomous run started after Stage 2 rebuild
- 2026-04-26 00:08 UTC — **complete**
  - prepared image → `https://huggingface.co/datasets/kushasareen/waa-windows-image/resolve/main/waa-windows-prepared.qcow2` (19.4 GB qcow2)
  - data.img → `https://huggingface.co/datasets/kushasareen/waa-windows-image/resolve/main/data.img` (19.2 GB qcow2-wrapped)
  - Backup of pristine data.img kept at `~/.cube/waa/storage.NEW-libreoffice-2026-04-25/data.img` (raw 18 G actual / 30 G sparse)
  - End-to-end wall time: ~10.5 hours (most spent fighting WSMan upload speed; manual_finish_image.py worked around it)
