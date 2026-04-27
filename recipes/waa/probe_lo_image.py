"""Provision the LibreOffice-enabled WAA image and inventory the 8 baseline apps.

This is the first run against the new image (uploaded to HuggingFace 2026-04-26
at the same URL as the old one — only the content changed). To avoid stomping
on the still-working old gallery image, we use a fresh cache namespace:

    image_name_suffix    = "-kusha-lo"   (was "-kusha")
    source_cache_blob    = "sources/waa-windows-prepared-lo.qcow2"  (was .qcow2)

Both old keys are preserved in Azure storage. Switching back is a one-line revert.

First run: ~20–40 min (HF download + bootstrap VM + gallery image build).
Subsequent runs: cache-hit, ~2 min provisioning.

Usage:
    uv run recipes/waa/probe_lo_image.py
"""

import json
import logging
import os
import time

import requests
from cube_infra_azure import AzureInfraConfig
from dotenv import load_dotenv
from waa_cube.azure import WAA_WINDOWS_RESOURCE

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
for _noisy in ("azure.core.pipeline.policies.http_logging_policy", "azure.identity", "urllib3.connectionpool"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

INFRA = AzureInfraConfig(
    resource_group=os.environ.get("AZURE_RESOURCE_GROUP") or "ui_assist",
    storage_account=os.environ.get("AZURE_STORAGE_ACCOUNT") or "cubeexpvhd",
    vnet_name="vnet-westus2",
    nsg_name="osworld-nsg",
    windows_admin_username="Docker",
    image_name_suffix="-kusha-lo",
    source_cache_blob="sources/waa-windows-prepared-lo.qcow2",
)

# The 8 apps WAA's setup.ps1 is supposed to install. LO is the new addition.
APP_PATHS = {
    "LibreOffice": r"C:\Program Files\LibreOffice\program\soffice.exe",
    "VLC": r"C:\Program Files\VideoLAN\VLC\vlc.exe",
    "GIMP": r"C:\Program Files\GIMP 2\bin\gimp-2.10.exe",
    "Chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "Edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "7zip": r"C:\Program Files\7-Zip\7z.exe",
    "Git": r"C:\Program Files\Git\bin\git.exe",
    "Thunderbird": r"C:\Program Files\Mozilla Thunderbird\thunderbird.exe",
}


def post_execute(base: str, command: list[str], shell: bool = True) -> dict:
    payload = json.dumps({"command": command, "shell": shell})
    try:
        r = requests.post(f"{base}/execute", data=payload, headers={"Content-Type": "application/json"}, timeout=60)
        return r.json() if r.status_code == 200 else {"http_status": r.status_code, "text": r.text[:300]}
    except Exception as exc:
        return {"error": str(exc)}


def main() -> None:
    # First-run bootstrap: HF download → qcow2 cache → VHD → gallery image.
    # Idempotent — no-ops once the gallery image exists for this image_name_suffix.
    INFRA.provision(WAA_WINDOWS_RESOURCE)
    handle = INFRA.launch(WAA_WINDOWS_RESOURCE)
    base = handle.endpoint
    print(f"\n=== VM up at {base} ===\n")
    try:
        print("Waiting 60s for Windows to finish booting...")
        time.sleep(60)

        # PowerShell probe — matches what user did to verify HF qcow2.
        # Explicitly checks 64-bit context, drive list, and Test-Path each app.
        ps_probe = r"""
$arch = [System.Environment]::Is64BitProcess
Write-Output "Is64BitProcess=$arch"
Write-Output "ProcessorArch=$env:PROCESSOR_ARCHITECTURE"
Write-Output ""
Write-Output "=== Drives ==="
Get-PSDrive -PSProvider FileSystem | ForEach-Object { '{0}: Used={1}GB Free={2}GB Root={3}' -f $_.Name, [int]($_.Used/1GB), [int]($_.Free/1GB), $_.Root }
Write-Output ""
Write-Output "=== Apps via Test-Path (matches user verification) ==="
$apps = @{
    'LibreOffice' = 'C:\Program Files\LibreOffice\program\soffice.exe'
    'VLC'         = 'C:\Program Files\VideoLAN\VLC\vlc.exe'
    'GIMP'        = 'C:\Program Files\GIMP 2\bin\gimp-2.10.exe'
    'Chrome'      = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
    'Edge'        = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    '7zip'        = 'C:\Program Files\7-Zip\7z.exe'
    'Git'         = 'C:\Program Files\Git\bin\git.exe'
    'Thunderbird' = 'C:\Program Files\Mozilla Thunderbird\thunderbird.exe'
}
foreach ($k in $apps.Keys) {
    $exists = Test-Path $apps[$k]
    Write-Output ('{0,-12} {1} {2}' -f $k, $exists, $apps[$k])
}
Write-Output ""
Write-Output "=== C:\ root ==="
Get-ChildItem C:\ -Force | ForEach-Object { $_.Name } | Sort-Object
Write-Output ""
Write-Output "=== C:\Program Files\ ==="
Get-ChildItem 'C:\Program Files' -Force -ErrorAction SilentlyContinue | ForEach-Object { $_.Name } | Sort-Object
Write-Output ""
Write-Output "=== C:\Program Files (x86)\ ==="
Get-ChildItem 'C:\Program Files (x86)' -Force -ErrorAction SilentlyContinue | ForEach-Object { $_.Name } | Sort-Object
"""
        r = post_execute(base, ["powershell", "-NoProfile", "-Command", ps_probe], shell=False)
        print("--- PowerShell probe output ---")
        print(r.get("output") or r.get("error") or str(r))
        print("--- end PowerShell probe ---\n")

        results: dict[str, bool] = {}
        first_raw = None
        for name, path in APP_PATHS.items():
            r = post_execute(base, ["cmd", "/c", f'if exist "{path}" (echo YES) else (echo NO)'])
            out = (r.get("output") or "").strip().upper()
            present = out.startswith("YES")
            results[name] = present
            if first_raw is None and not present:
                first_raw = r  # capture raw for debugging
            print(f"  {name:<12} {'✅' if present else '❌'}  {path}")

        if first_raw is not None:
            print(f"\n--- raw response of first failure (for diagnosis) ---\n  {first_raw}")

        # LO version check — confirms the new install actually works, not just exists on disk
        if results.get("LibreOffice"):
            print("\n--- LibreOffice version ---")
            r = post_execute(
                base,
                [
                    "powershell",
                    "-Command",
                    r"(Get-WmiObject Win32_Product -Filter \"Name like 'LibreOffice%'\").Version",
                ],
                shell=False,
            )
            print(f"  {(r.get('output') or '').strip() or r}")

        missing = [n for n, ok in results.items() if not ok]
        print()
        if missing:
            print(f"❌ FAIL — missing apps: {missing}")
            # Bonus diagnostic: list top of Program Files to see what IS there
            print("\n--- C:\\Program Files\\ contents ---")
            r = post_execute(base, ["cmd", "/c", 'dir "C:\\Program Files" /B'])
            print(f"  {(r.get('output') or '').strip() or r}")
            print("\n--- C:\\Program Files (x86)\\ contents ---")
            r = post_execute(base, ["cmd", "/c", 'dir "C:\\Program Files (x86)" /B'])
            print(f"  {(r.get('output') or '').strip() or r}")
        else:
            print("✅ PASS — all 8 apps present on the new image")
    finally:
        print("\n=== tearing down ===")
        handle.close()


if __name__ == "__main__":
    main()
