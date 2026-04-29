"""Diagnostic: characterize chrome CDP /json/version/ readiness across fresh VMs.

What we know from the Haiku full eval (152 tasks, 9 with CDP errors):
- 7/9 had transient errors but recovered within 2-3 retries (~10-15s)
- 2/9 (ep9 06fe7178, ep84 82bc8d6a) hit full 30 retries × 5s = 150s and gave up
- ep84: 29 consecutive `status 500` from /json/version/
- ep9:  21x `status 500`, then 8x `socket hang up` (proxy dies)
- All 9 had IDENTICAL chrome launch args — not task-specific
- Status 500 means socat/proxy on VM:9222 IS forwarding, but Chrome on :1337 isn't
  responding properly. So the ssh tunnel + socat are alive — Chrome itself isn't.

Hypotheses (ranked):
  H1: Chrome crashed/never came up (Defender flagged user-data-dir, or chrome.exe died)
  H2: Chrome is alive but CDP service took >150s to initialize
  H3: socat dies mid-run (would explain ep9's 500→hang_up transition)

Probe:
  1. Launch N fresh VMs in parallel (mirror eval cohort pattern)
  2. On each: replay the failed-task setup sequence:
       taskkill chrome.exe -> sleep 2s -> launch chrome with same args ->
       poll /json/version/ every 1s for up to 300s
  3. After 30s of unsuccessful polling, snapshot VM-side state:
       tasklist (chrome alive?), netstat 1337/9222 (listeners?),
       Defender threats, /Temp/cdp-profile contents
  4. Save per-VM record with attempts timeline + snapshots
  5. Tear down

Output: /tmp/cube-harness-logs/chrome-cdp-probe.json

Usage:
    N_VMS=20 uv run recipes/waa/probe_chrome_cdp.py
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse

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

N_VMS = int(os.environ.get("N_VMS", "20"))
POLL_INTERVAL_S = 1.0
MAX_POLL_S = 300
SNAPSHOT_AFTER_S = 30  # if no 200 within 30s, capture VM state
OUT = Path("/tmp/cube-harness-logs/chrome-cdp-probe.json")

# Exact launch args from the 2 failed eval episodes (06fe7178, 82bc8d6a)
CHROME_LAUNCH_CMD = [
    "cmd", "/c", "start", "", "/b",
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "--remote-debugging-port=1337",
    "--user-data-dir=C:\\Temp\\cdp-profile",
    "--no-first-run",
    "--no-default-browser-check",
    "--force-renderer-accessibility",
]


def post_execute(base: str, command: list[str], shell: bool = False, timeout: int = 30) -> dict:
    """Run a command on the VM via /setup/execute. Returns parsed JSON or error dict."""
    try:
        r = requests.post(
            f"{base}/setup/execute",
            data=json.dumps({"command": command, "shell": shell}),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json()
        return {"http_status": r.status_code, "text": r.text[:300]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def post_launch(base: str, command: list[str], shell: bool = False, timeout: int = 15) -> dict:
    """Fire-and-forget launch via /setup/launch."""
    try:
        r = requests.post(
            f"{base}/setup/launch",
            data=json.dumps({"command": command, "shell": shell}),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        return {"status": r.status_code, "text": r.text[:200]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def poll_cdp(chromium_url: str) -> tuple[int, float, str]:
    """One GET to /json/version/. Returns (status, elapsed_s, body_preview)."""
    t0 = time.time()
    try:
        r = requests.get(f"{chromium_url}/json/version/", timeout=5)
        return r.status_code, time.time() - t0, r.text[:120]
    except Exception as exc:
        return -1, time.time() - t0, f"EXC {type(exc).__name__}: {exc}"


def snapshot_vm_state(base: str) -> dict:
    """Capture chrome/socat/defender state inside the VM."""
    snap: dict = {}
    snap["tasklist_chrome"] = post_execute(
        base,
        ["powershell", "-NoProfile", "-Command", "Get-Process chrome -ErrorAction SilentlyContinue | Select-Object Id,StartTime,CPU,WorkingSet64,MainWindowTitle | Format-List"],
        timeout=15,
    ).get("output", "")[:2000]
    snap["netstat_1337"] = post_execute(
        base, ["cmd", "/c", "netstat -ano | findstr :1337"], timeout=10
    ).get("output", "")[:1000]
    snap["netstat_9222"] = post_execute(
        base, ["cmd", "/c", "netstat -ano | findstr :9222"], timeout=10
    ).get("output", "")[:1000]
    snap["defender_threats"] = post_execute(
        base,
        ["powershell", "-NoProfile", "-Command", "Get-MpThreatDetection -ErrorAction SilentlyContinue | Select-Object -First 10 ThreatName,Resources,InitialDetectionTime | Format-List"],
        timeout=15,
    ).get("output", "")[:2000]
    snap["cdp_profile_dir"] = post_execute(
        base, ["cmd", "/c", "dir C:\\Temp\\cdp-profile 2>NUL"], timeout=10
    ).get("output", "")[:1500]
    snap["cdp_profile_lockfile"] = post_execute(
        base,
        ["powershell", "-NoProfile", "-Command", "Get-Item C:\\Temp\\cdp-profile\\SingletonLock,C:\\Temp\\cdp-profile\\Default\\Cookies-journal -ErrorAction SilentlyContinue | Select-Object Name,LastWriteTime,Length | Format-List"],
        timeout=10,
    ).get("output", "")[:800]
    return snap


def probe_one_vm(vm_idx: int) -> dict:
    """Provision -> run chrome setup -> poll until 5 consecutive 200s or 300s elapsed."""
    record: dict = {"vm_idx": vm_idx, "started_at": time.time()}
    print(f"[VM {vm_idx:2d}] launching...")
    try:
        handle = INFRA.launch(WAA_WINDOWS_RESOURCE)
        record["vm_up_at_s"] = time.time() - record["started_at"]
        base = handle.endpoint
        endpoints = getattr(handle, "endpoints", {}) or {}
        chromium_url_full = endpoints.get("vm_port_9222")
        if not chromium_url_full:
            record["error"] = "no vm_port_9222 endpoint"
            handle.close()
            return record
        # chromium_url_full is something like http://localhost:15008 or http://host:9222
        chromium_host_port = urlparse(chromium_url_full)
        chromium_url = f"http://{chromium_host_port.hostname}:{chromium_host_port.port}"
        record["base"] = base
        record["chromium_url"] = chromium_url
        print(f"[VM {vm_idx:2d}] up base={base} chromium={chromium_url} (after {record['vm_up_at_s']:.1f}s)")

        # Setup sequence — same as failed eval tasks
        seq_t0 = time.time()
        record["setup"] = []
        record["setup"].append({"step": "taskkill_chrome", "t": time.time() - seq_t0,
                                "result": post_execute(base, ["taskkill", "/IM", "chrome.exe", "/F"], shell=True, timeout=15)})
        time.sleep(2)
        record["setup"].append({"step": "sleep_2s", "t": time.time() - seq_t0})
        record["setup"].append({"step": "launch_chrome", "t": time.time() - seq_t0,
                                "result": post_launch(base, CHROME_LAUNCH_CMD)})

        # Poll CDP
        attempts = []
        consecutive_200 = 0
        first_200_at: float | None = None
        first_500_at: float | None = None
        snapshot_taken = False
        snapshot: dict | None = None
        poll_t0 = time.time()

        while time.time() - poll_t0 < MAX_POLL_S:
            status, elapsed, body = poll_cdp(chromium_url)
            t_rel = time.time() - poll_t0
            attempts.append({"t": round(t_rel, 2), "status": status,
                             "elapsed_s": round(elapsed, 3), "body": body[:80]})

            if status == 200 and first_200_at is None:
                first_200_at = t_rel
            if status == 500 and first_500_at is None:
                first_500_at = t_rel

            if status == 200:
                consecutive_200 += 1
                if consecutive_200 >= 5:
                    print(f"[VM {vm_idx:2d}] settled at t={t_rel:.1f}s")
                    break
            else:
                consecutive_200 = 0

            # snapshot once at 30s if still stuck
            if not snapshot_taken and t_rel >= SNAPSHOT_AFTER_S and consecutive_200 == 0:
                print(f"[VM {vm_idx:2d}] still stuck at t={t_rel:.1f}s — snapshotting")
                snapshot = snapshot_vm_state(base)
                snapshot_taken = True

            time.sleep(POLL_INTERVAL_S)

        record["first_200_at_s"] = first_200_at
        record["first_500_at_s"] = first_500_at
        record["settled"] = consecutive_200 >= 5
        record["total_attempts"] = len(attempts)
        record["attempts"] = attempts
        record["snapshot_at_30s"] = snapshot

        handle.close()
        record["finished_at_s"] = time.time() - record["started_at"]
        status_summary = "SETTLED" if record["settled"] else "STUCK"
        print(f"[VM {vm_idx:2d}] {status_summary} first200={first_200_at}s attempts={len(attempts)}")
        return record
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["finished_at_s"] = time.time() - record["started_at"]
        print(f"[VM {vm_idx:2d}] ERROR: {record['error']}")
        return record


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nProvisioning {N_VMS} VMs in parallel; each runs chrome setup + polls /json/version/ for up to {MAX_POLL_S}s\n")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_VMS) as ex:
        futures = [ex.submit(probe_one_vm, i) for i in range(N_VMS)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Sort and summarise
    results.sort(key=lambda r: r.get("vm_idx", 0))
    settled = [r for r in results if r.get("settled")]
    stuck = [r for r in results if r.get("settled") is False]
    errored = [r for r in results if "error" in r]

    print()
    print("=" * 60)
    print(f"SUMMARY ({time.time() - t0:.1f}s wall)")
    print("=" * 60)
    print(f"  settled: {len(settled)}/{N_VMS}")
    print(f"  stuck:   {len(stuck)}/{N_VMS}")
    print(f"  errored: {len(errored)}/{N_VMS}")
    print()
    if settled:
        ttfs = [r["first_200_at_s"] for r in settled if r.get("first_200_at_s") is not None]
        if ttfs:
            print(f"  settled time-to-first-200: min={min(ttfs):.1f}s  median={sorted(ttfs)[len(ttfs)//2]:.1f}s  max={max(ttfs):.1f}s")
    for r in stuck:
        snap = r.get("snapshot_at_30s") or {}
        chrome = snap.get("tasklist_chrome", "")[:200].replace("\n", " | ")
        port1337 = snap.get("netstat_1337", "")[:200].replace("\n", " | ")
        port9222 = snap.get("netstat_9222", "")[:200].replace("\n", " | ")
        print(f"\n  STUCK VM {r['vm_idx']}:")
        print(f"    chrome process: {chrome[:150]}")
        print(f"    netstat 1337:   {port1337[:120]}")
        print(f"    netstat 9222:   {port9222[:120]}")

    OUT.write_text(json.dumps({"results": results, "n_vms": N_VMS,
                               "wall_time_s": time.time() - t0}, indent=2))
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
