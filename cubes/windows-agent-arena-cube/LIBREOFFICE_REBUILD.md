# LibreOffice fix — data.img rebuild journal

Why this exists: `data.img` was missing LibreOffice. We tried two approaches; this is the journal of the second one (the working one) and where we are right now.

---

## Problem

Inventory of the previous `data.img` (queried via the Flask `/execute` endpoint on a running VM):

```
7zip           ✅ C:\Program Files\7-Zip\7z.exe
Chrome         ✅ C:\Program Files\Google\Chrome\Application\chrome.exe
Edge           ✅ C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
GIMP           ✅ C:\Program Files\GIMP 2\bin\gimp-2.10.exe
Git            ✅ C:\Program Files\Git\bin\git.exe
Thunderbird    ✅ C:\Program Files\Mozilla Thunderbird\thunderbird.exe
VLC            ✅ C:\Program Files\VideoLAN\VLC\vlc.exe
LibreOffice    ❌ C:\Program Files\LibreOffice\program\soffice.exe   ← only failure
```

7 of 8 apps install correctly via upstream WAA's `setup.ps1`. LibreOffice has two distinct problems:

1. **Extension bug** in `WindowsAgentArena/src/win-arena-container/vm/setup/setup.ps1:197`:
   ```
   $libreOfficeInstallerFilePath = "$env:TEMP\libreOffice_installer.exe"
   ```
   The downloaded file is an MSI but saved with `.exe`. `msiexec /i FILE.exe /quiet` silently no-ops on a non-MSI extension. Upstream then prints "LibreOffice has been installed" without checking — silent failure.
2. **Stale mirror URLs** in `tools_config.json` — all 3 entries 404 in 2026 (24.8.2 has been moved off `stable/` and into `downloadarchive/.../old/24.8.2.1/`).

---

## Approach 1 — Packer layer (abandoned)

Idea: leave data.img alone, layer LibreOffice on via a new Packer provisioner that runs after the existing OpenSSH / Azure-VM-Agent provisioners.

Why we abandoned it: QEMU's user-mode networking (SLiRP) throttles all guest↔internet traffic to ~8 KB/s. Even with the Packer `file` provisioner pre-staging the MSI from the host (where bandwidth is fine — measured 19 MB/s), Packer's WSMan upload is also SLiRP-bound and only got us to ~13 KB/s. Projected ~7 hours for the 346 MB MSI, which is unacceptable. Bypass options (CD-ROM ISO attach, virtio-9p, TAP networking) all add significant complexity.

Code from this attempt is still in the repo but currently no-op'd:
- `packer/scripts/install-libreoffice.ps1`
- `packer/waa-windows.pkr.hcl` (file provisioner block)
- `packer/run.sh` (`fetch_if_missing` cache step)
- `packer/scripts/install-openssh-server.ps1`, `packer/scripts/install-azure-vm-agent.ps1` (added local-cache fallback paths)

These will be **reverted** once Approach 2 succeeds — see "TODO after rebuild" below.

---

## Approach 2 — fix upstream `setup.ps1` and rebuild data.img (in progress)

Stage 2 of the WAA pipeline (`./scripts/run-local.sh --prepare-image true`) runs Windows install + setup.ps1 inside the `dockurr/windows` container, which uses TAP networking — kernel-space, line-rate, no SLiRP throttling. Same script that already installs Chrome/Edge/GIMP/etc. successfully. Only LibreOffice fails and only because of the bugs above. So: patch the bugs, rebuild data.img, done.

### Patches applied

`WindowsAgentArena/src/win-arena-container/vm/setup/setup.ps1` (line 197):
```diff
-    $libreOfficeInstallerFilePath = "$env:TEMP\libreOffice_installer.exe"
+    # File MUST end in .msi — `msiexec /i` validates the extension and silently
+    # no-ops on .exe (the original bug that left LibreOffice missing from data.img).
+    $libreOfficeInstallerFilePath = "$env:TEMP\LibreOffice_24.8.2.1_Win_x86-64.msi"
```

`WindowsAgentArena/src/win-arena-container/vm/setup/tools_config.json`:
```diff
-    "LibreOffice": {
-        "mirrors": [
-            "https://mirror.raiolanetworks.com/tdf/libreoffice/stable/24.8.2/...msi",
-            "https://mirrors.iu13.net/tdf/libreoffice/stable/24.8.2/...msi",
-            "https://download.documentfoundation.org/libreoffice/stable/24.8.2/...msi"
-        ]
-    },
+    "LibreOffice": {
+        "mirrors": [
+            "https://downloadarchive.documentfoundation.org/libreoffice/old/24.8.2.1/win/x86_64/LibreOffice_24.8.2.1_Win_x86-64.msi"
+        ]
+    },
```

Also patched (one-time) `WindowsAgentArena/config.json` → set `AZURE_ENDPOINT` to a placeholder URL because `run-local.sh` chokes on an empty value when forwarding `--azure-endpoint` to `run.sh`.

### Steps executed

1. ✅ Killed the in-progress Packer build from Approach 1.
2. ✅ Backed up the existing `~/.cube/waa/storage/*` to `~/.cube/waa/storage.bak-2026-04-25/` (10 files moved, no copy — same FS).
3. ✅ Patched upstream `setup.ps1` and `tools_config.json` (above).
4. ✅ Rebuilt the WAA Docker image (`./scripts/build-container-image.sh`) — confirmed via `docker run --rm winarena:latest cat /oem/setup.ps1 | grep LibreOffice` that our patches are in the image.
5. 🔄 Running Stage 2 (`./scripts/run-local.sh --prepare-image true --start-client false`) — **currently in progress**.
6. ⏳ Verify LibreOffice is in new data.img via Flask `/execute` probe (`Test-Path soffice.exe` + `Get-WmiObject Win32_Product` for version).
7. ⏳ Copy new data.img → `~/.cube/images/waa-windows-vm.qcow2`.
8. ⏳ Re-run `make bootstrap-winrm` (Stage 3 — needs to re-do because data.img is new).
9. ⏳ Re-run `make build-image` (Stage 4 — Packer overlay with OpenSSH / Azure agent / SSH key / autologon).

### Current state (~136 min into Stage 2)

- Container `winarena` up ~2.3 hours (slower than the user's ~60 min baseline, see "Why slow" below)
- Windows OS install complete; OOBE done; AutoLogon fired; **install.bat is now running setup.ps1** (visible in screendump as "Administrator: Install" cmd window)
- VM clock advancing normally (last check 9:25 PM 4/24/2026 inside guest)
- VM has downloaded ~404 MB total from the internet — likely Chrome / Thunderbird / smaller tools done. **LibreOffice MSI (346 MB) probably has not started downloading yet** (would push counter past ~750 MB).
- Container CPU mostly 500-800% (7+ cores busy)
- `data.img` 9.8 GB (sparse, will grow toward ~14 GB final)

### Why this Stage 2 is slower than the previous run

Best guesses, all speculative:
- `/dev/root` at 82% full → fragmented sparse writes
- Windows pulled new cumulative updates this run that weren't in the previous run
- Concurrent KVM contention (the long-running `waa-checklist` VM on PID 3908130 is still up using ~1.5 cores)
- `dockurr/windows` base image refresh between runs

### Verification plan after Stage 2 finishes

The Flask `/probe` endpoint returning 200 means `setup.ps1` finished and `on-logon.ps1` started Flask — **necessary but not sufficient** for LibreOffice success. Before declaring victory:

1. Spin the new data.img up briefly via dockurr container with `--prepare-image false` (just boots the existing image, no install) — ~5 min.
2. Run Flask `/execute` inventory probe (PowerShell):
   ```powershell
   $paths = @{
     'LibreOffice' = 'C:\Program Files\LibreOffice\program\soffice.exe';
     'VLC' = 'C:\Program Files\VideoLAN\VLC\vlc.exe';
     # ... etc
   }
   foreach ($k in $paths.Keys) { '{0,-14} {1} {2}' -f $k, (Test-Path $paths[$k]), $paths[$k] }
   ```
3. Verify version: `(Get-WmiObject Win32_Product -Filter "Name like 'LibreOffice%'").Version` → expect `24.8.2.1`.
4. Verify NO regression in the other 7 apps.

Only after all 8 ✅ → copy data.img into `~/.cube/images/waa-windows-vm.qcow2` and proceed.

---

## TODO after rebuild succeeds

- Revert the Packer-layer LibreOffice additions (now obsolete):
  - Delete `packer/scripts/install-libreoffice.ps1`
  - Remove its line from `packer/waa-windows.pkr.hcl`'s `scripts = [...]`
  - Remove the second `provisioner "file"` block (the one with `sources = [...]`) from `packer/waa-windows.pkr.hcl`
  - Optionally: remove the `fetch_if_missing` cache step + the local-file fallback in `install-openssh-server.ps1` / `install-azure-vm-agent.ps1` (harmless to keep, but adds noise)
- Restore `WindowsAgentArena/config.json` → set `AZURE_ENDPOINT` back to empty if that's the desired default.
- Bootstrap-WinRM + Packer-build the new prepared image.
- Run smoke test, push to HuggingFace if needed.

---

## Files of interest

- `WindowsAgentArena/src/win-arena-container/vm/setup/setup.ps1` (patched, line 197)
- `WindowsAgentArena/src/win-arena-container/vm/setup/tools_config.json` (patched)
- `WindowsAgentArena/src/win-arena-container/vm/setup/install.bat` (calls setup.ps1 at first user logon)
- `WindowsAgentArena/src/win-arena-container/vm/setup/on-logon.ps1` (starts Flask server after setup.ps1 done)
- `~/.cube/waa/storage.bak-2026-04-25/` — backup of old data.img + firmware state
- `/tmp/waa-stage2.log` — current Stage 2 build log
- `~/.cube/cache/` — pre-staged installer files from Approach 1 (still present, used by current Packer scripts as fallback)
