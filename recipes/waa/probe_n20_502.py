"""Diagnostic probe — characterize the /setup/upload 502 issue at n_cpus=20.

Why we're running this: failure rate jumps from ~0% at n=10 to ~28% at n=20.
That magnitude of jump rules out independent per-VM probability — there must
be a shared resource that gets contended at higher concurrency. Candidates
listed in RESULTS.md and the session notes.

What this probe does, per VM (run in parallel, N=20 default):
  1. Provision a fresh VM
  2. Record timeline events (provision-start, VM-ready, SSH-key-injected,
     tunnels-open) with timestamps
  3. As soon as the handle is up, POST one /setup/upload of a real task file
  4. If 502 hits, **immediately** snapshot:
       - tasklist (Flask, OEM server, Defender, every other process)
       - netstat 5000 (Flask listening?)
       - last 200 lines of C:\\oem\\server\\server.log
       - Get-MpThreatDetection (Defender events)
       - Last 20 system event log entries (warnings/errors)
       - dir C:\\Users\\Docker\\Downloads (file-write target dir state)
  5. Continue polling every 2s, capture transition timestamps
  6. Save per-VM record + a global cohort timeline

Output:
  /tmp/cube-harness-logs/probe-n20-502/cohort.json  — global timeline
  /tmp/cube-harness-logs/probe-n20-502/vm-{idx}.json — per-VM record

Usage:
  N_VMS=20 uv run recipes/waa/probe_n20_502.py
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import logging
import os
import threading
import time
from pathlib import Path

import requests
from cube_infra_azure import AzureInfraConfig
from dotenv import load_dotenv
from requests_toolbelt import MultipartEncoder
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
POLL_INTERVAL_S = 2.0
MAX_POLL_S = 300
SNAPSHOT_AT_FIRST_502 = True
OUT_DIR = Path("/tmp/cube-harness-logs/probe-n20-502")

# Real task file used by the failing eval episodes — same upload payload that
# triggers 502s in production.
REAL_FILE_URL = "https://raw.githubusercontent.com/rogeriobonatti/winarenafiles/main/task_files/calc/04d9aeaf-7bed-4024-bedb-e10e6f00eb7f-WOS/config/SmallBalanceSheet.xlsx"
REAL_FILE_DEST = "C:\\Users\\Docker\\Downloads\\SmallBalanceSheet.xlsx"

# Cohort-level event timeline. Append-only with mutex so threads don't tangle.
COHORT_EVENTS: list[dict] = []
COHORT_LOCK = threading.Lock()


def cohort_event(vm_idx: int, kind: str, **extra: object) -> None:
    with COHORT_LOCK:
        COHORT_EVENTS.append({"t": time.time(), "vm_idx": vm_idx, "kind": kind, **extra})


def post_execute(base: str, command: list[str], shell: bool = False, timeout: int = 30) -> dict:
    try:
        r = requests.post(
            f"{base}/setup/execute",
            data=json.dumps({"command": command, "shell": shell}),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        return r.json() if r.status_code == 200 else {"http_status": r.status_code, "text": r.text[:300]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def upload_attempt(base: str, data: bytes) -> dict:
    """Single /setup/upload attempt — captures status, timing, body."""
    form = MultipartEncoder({
        "file_path": REAL_FILE_DEST,
        "file_data": ("SmallBalanceSheet.xlsx", io.BytesIO(data), "application/octet-stream"),
    })
    t0 = time.time()
    try:
        r = requests.post(
            f"{base}/setup/upload",
            headers={"Content-Type": form.content_type},
            data=form,
            timeout=30,
        )
        return {"status": r.status_code, "elapsed_s": time.time() - t0,
                "body_preview": r.text[:120], "headers": dict(r.headers)}
    except Exception as exc:
        return {"status": -1, "elapsed_s": time.time() - t0,
                "body_preview": f"EXC {type(exc).__name__}: {exc}"}


def snapshot_vm(base: str) -> dict:
    """Capture VM-side state at moment of 502."""
    snap: dict = {}
    snap["tasklist"] = post_execute(
        base,
        ["powershell", "-NoProfile", "-Command",
         "Get-Process | Where-Object {$_.ProcessName -match 'python|flask|oem|MsMpEng|Defender|werkzeug'} | Select-Object Id,ProcessName,StartTime,WorkingSet64,CPU | Format-List"],
        timeout=20,
    ).get("output", "")[:3000]
    snap["netstat_5000"] = post_execute(
        base, ["cmd", "/c", "netstat -ano | findstr :5000"], timeout=10
    ).get("output", "")[:1000]
    snap["server_log_tail"] = post_execute(
        base,
        ["powershell", "-NoProfile", "-Command", "Get-Content C:\\oem\\server\\server.log -Tail 50 -ErrorAction SilentlyContinue"],
        timeout=15,
    ).get("output", "")[:3000]
    snap["defender_threats"] = post_execute(
        base,
        ["powershell", "-NoProfile", "-Command",
         "Get-MpThreatDetection -ErrorAction SilentlyContinue | Select-Object -First 10 ThreatName,Resources,InitialDetectionTime | Format-List"],
        timeout=15,
    ).get("output", "")[:2000]
    snap["sys_eventlog_recent"] = post_execute(
        base,
        ["powershell", "-NoProfile", "-Command",
         "Get-WinEvent -LogName System -MaxEvents 20 -ErrorAction SilentlyContinue | Where-Object {$_.LevelDisplayName -ne 'Information'} | Select-Object TimeCreated,LevelDisplayName,ProviderName,Message | Format-List"],
        timeout=20,
    ).get("output", "")[:3000]
    snap["downloads_dir"] = post_execute(
        base, ["cmd", "/c", "dir C:\\Users\\Docker\\Downloads"], timeout=10
    ).get("output", "")[:1500]
    snap["http_sys_log"] = post_execute(
        base,
        ["powershell", "-NoProfile", "-Command",
         "Get-ChildItem C:\\Windows\\System32\\LogFiles\\HTTPERR -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { Get-Content $_.FullName -Tail 30 }"],
        timeout=15,
    ).get("output", "")[:3000]
    return snap


def probe_one_vm(vm_idx: int, file_data: bytes) -> dict:
    record: dict = {"vm_idx": vm_idx, "started_at": time.time()}
    cohort_event(vm_idx, "probe_start")
    try:
        cohort_event(vm_idx, "launch_call")
        handle = INFRA.launch(WAA_WINDOWS_RESOURCE)
        cohort_event(vm_idx, "launch_returned", endpoint=handle.endpoint)
        record["vm_endpoint"] = handle.endpoint
        record["provision_s"] = time.time() - record["started_at"]
        base = handle.endpoint

        # Poll /setup/upload, snapshot on first 502
        attempts = []
        snapshot = None
        first_502_at: float | None = None
        first_200_at: float | None = None
        consecutive_200 = 0
        poll_t0 = time.time()
        while time.time() - poll_t0 < MAX_POLL_S:
            attempt = upload_attempt(base, file_data)
            t_rel = time.time() - poll_t0
            attempt["t_rel"] = round(t_rel, 2)
            attempts.append(attempt)
            cohort_event(vm_idx, f"upload_status_{attempt['status']}", t_rel=round(t_rel, 2))

            if attempt["status"] == 502 and first_502_at is None:
                first_502_at = t_rel
                if SNAPSHOT_AT_FIRST_502:
                    cohort_event(vm_idx, "snapshot_start")
                    try:
                        snapshot = snapshot_vm(base)
                        cohort_event(vm_idx, "snapshot_done")
                    except Exception as exc:
                        snapshot = {"error": f"{type(exc).__name__}: {exc}"}

            if attempt["status"] == 200:
                if first_200_at is None:
                    first_200_at = t_rel
                consecutive_200 += 1
                if consecutive_200 >= 3:
                    break
            else:
                consecutive_200 = 0

            time.sleep(POLL_INTERVAL_S)

        record["first_502_at_s"] = first_502_at
        record["first_200_at_s"] = first_200_at
        record["settled"] = consecutive_200 >= 3
        record["total_attempts"] = len(attempts)
        record["attempts"] = attempts
        record["snapshot_at_first_502"] = snapshot

        cohort_event(vm_idx, "probe_close")
        handle.close()
        record["finished_at_s"] = time.time() - record["started_at"]
        return record
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["finished_at_s"] = time.time() - record["started_at"]
        cohort_event(vm_idx, "probe_error", error=str(exc)[:200])
        return record


def wait_for_quota(infra: AzureInfraConfig, vms_needed: int, vcpus_per_vm: int = 8,
                   poll_s: float = 15.0, max_wait_s: float = 600.0) -> None:
    """Block until enough Standard DSv3 quota is free for `vms_needed` D8s_v3 VMs.

    Avoids racing with async VM teardown after a previous run/eval. Logs the
    state every poll. Raises after `max_wait_s` if quota never frees.
    """
    needed = vms_needed * vcpus_per_vm
    compute = infra._compute()
    location = "westus2"  # AzureInfraConfig has no public attr, but our path is single-region
    t0 = time.time()
    while True:
        current = limit = 0
        for u in compute.usage.list(location):
            if u.name and u.name.value == "standardDSv3Family":
                current, limit = int(u.current_value or 0), int(u.limit or 0)
                break
        free = limit - current
        if free >= needed:
            logging.info("quota OK: %d/%d used, %d free, need %d", current, limit, free, needed)
            return
        elapsed = time.time() - t0
        if elapsed > max_wait_s:
            raise RuntimeError(f"quota wait exceeded {max_wait_s:.0f}s; have {free}, need {needed}")
        logging.info("quota wait: %d/%d used, %d free, need %d (waited %.0fs)",
                     current, limit, free, needed, elapsed)
        time.sleep(poll_s)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Block until enough vCPU quota is free for our cohort. Without this, a
    # restart that races VM teardown from a prior run hits OperationNotAllowed
    # on every launch.
    wait_for_quota(INFRA, vms_needed=N_VMS)

    print(f"Pre-fetching {REAL_FILE_URL}...")
    file_data = requests.get(REAL_FILE_URL, timeout=30).content
    print(f"  {len(file_data)} bytes")

    print(f"\nProvisioning {N_VMS} VMs in parallel; each polls /setup/upload until 3 consecutive 200s\n")
    cohort_event(-1, "cohort_start", n_vms=N_VMS)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_VMS) as ex:
        futures = [ex.submit(probe_one_vm, i, file_data) for i in range(N_VMS)]
        results = []
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            results.append(r)
            (OUT_DIR / f"vm-{r['vm_idx']:02d}.json").write_text(json.dumps(r, indent=2, default=str))
            status = "SETTLED" if r.get("settled") else ("ERROR" if "error" in r else "STUCK")
            print(f"  VM {r['vm_idx']:2d} {status}  first_502={r.get('first_502_at_s')}  first_200={r.get('first_200_at_s')}  attempts={r.get('total_attempts',0)}")

    cohort_event(-1, "cohort_end")
    elapsed = time.time() - t0
    n_settled = sum(1 for r in results if r.get("settled"))
    n_stuck = sum(1 for r in results if r.get("settled") is False)
    n_error = sum(1 for r in results if "error" in r)

    print()
    print("=" * 60)
    print(f"COHORT SUMMARY ({elapsed:.1f}s wall)")
    print("=" * 60)
    print(f"  settled: {n_settled}/{N_VMS}")
    print(f"  stuck:   {n_stuck}/{N_VMS}")
    print(f"  errored: {n_error}/{N_VMS}")
    if n_stuck > 0:
        print(f"\n  Stuck VMs reproduced the 502 issue. Per-VM snapshots saved in {OUT_DIR}.")

    (OUT_DIR / "cohort.json").write_text(json.dumps({
        "events": COHORT_EVENTS,
        "n_vms": N_VMS,
        "wall_time_s": elapsed,
        "summary": {"settled": n_settled, "stuck": n_stuck, "errored": n_error},
    }, indent=2, default=str))
    print(f"\nCohort timeline saved → {OUT_DIR}/cohort.json")


if __name__ == "__main__":
    main()
