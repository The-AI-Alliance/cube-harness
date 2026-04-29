"""Diagnostic: characterize the /setup/upload 502 race window.

What we know about the race:
- /setup/upload returns transient empty-body 502 on a small fraction of fresh VMs
- Settled VMs handle the same upload fine (10x concurrent works)
- No 502 in WAA Flask source (C:\\oem\\server\\main.py) → coming from somewhere
  in the TCP/http.sys stack, possibly Defender file-write blocking the response

What we don't know:
- Does the race ALWAYS clear given enough time? (transient vs systematic)
- Distribution of race window durations across VMs
- Whether specific Azure-host placements correlate with longer windows
- What state the VM is in at the moment of the first 502

This probe answers all of those:
1. Provisions one fresh VM
2. As soon as SSH is up + tunnel open, polls /setup/upload every 1s with a
   tiny payload, capturing each (status, response_time, body) until we get N
   consecutive 200s
3. On first 502 of each VM: snapshot Defender state, Flask server.log,
   tasklist, network listeners
4. Records to /tmp/cube-harness-logs/race-N.json — one record per VM
5. Repeats for N_VMS provisions, fully sequential (no concurrent pressure)

After running, analyze:
- If every VM eventually settles → race is purely transient, retry-tolerant
- If some VMs never settle → systematic per-VM problem (image / boot)
- Distribution of windows tells us right retry budget

Usage:
    N_VMS=3 uv run recipes/waa/probe_setup_upload_race.py
"""

from __future__ import annotations

import io
import json
import logging
import os
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

N_VMS = int(os.environ.get("N_VMS", "3"))
POLL_INTERVAL_S = 1.0
MAX_POLL_S = 600  # 10 minutes per VM — generous; we want to know if it never settles
CONSECUTIVE_OK_TARGET = 5  # require 5 in a row to call it settled
OUT_DIR = Path("/tmp/cube-harness-logs/race-probe")


def post_execute(base: str, command: list[str], shell: bool = False, timeout: int = 30) -> dict:
    payload = json.dumps({"command": command, "shell": shell})
    try:
        r = requests.post(f"{base}/execute", data=payload, headers={"Content-Type": "application/json"}, timeout=timeout)
        return r.json() if r.status_code == 200 else {"http_status": r.status_code, "text": r.text[:300]}
    except Exception as exc:
        return {"error": str(exc)}


def try_upload(base: str, idx: int) -> tuple[int, float, str]:
    """One /setup/upload attempt with a tiny in-memory payload. Returns (status, elapsed_s, body)."""
    form = MultipartEncoder(
        {
            "file_path": f"C:\\Users\\Docker\\Downloads\\probe_{idx}.txt",
            "file_data": (f"probe_{idx}.txt", io.BytesIO(b"hi"), "text/plain"),
        }
    )
    t0 = time.time()
    try:
        r = requests.post(
            f"{base}/setup/upload",
            headers={"Content-Type": form.content_type},
            data=form,
            timeout=15,
        )
        return r.status_code, time.time() - t0, r.text[:200]
    except Exception as exc:
        return -1, time.time() - t0, f"EXC {type(exc).__name__}: {exc}"


def snapshot_vm_state(base: str) -> dict:
    """Capture diagnostic state from inside the VM at moment of failure."""
    snap = {}
    snap["defender_status"] = post_execute(
        base,
        ["powershell", "-NoProfile", "-Command", "Get-MpComputerStatus | Select-Object -Property * | Format-List"],
        timeout=20,
    ).get("output", "")[:2000]
    snap["server_log_tail"] = post_execute(
        base, ["cmd", "/c", "type C:\\oem\\server\\server.log"], timeout=10
    ).get("output", "")[-2000:]
    snap["python_proc"] = post_execute(
        base,
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-Process python | Select-Object Id,StartTime,WorkingSet64,CPU | Format-List",
        ],
        timeout=10,
    ).get("output", "")[:1000]
    snap["listeners"] = post_execute(
        base, ["cmd", "/c", "netstat -ano | findstr LISTENING | findstr :5000"], timeout=10
    ).get("output", "")[:500]
    snap["downloads_dir"] = post_execute(
        base, ["cmd", "/c", "dir C:\\Users\\Docker\\Downloads"], timeout=10
    ).get("output", "")[:1000]
    return snap


def probe_one_vm(vm_idx: int) -> dict:
    """Provision a fresh VM, poll /setup/upload until settled, record everything."""
    record: dict = {"vm_idx": vm_idx, "started_at": time.time()}
    print(f"\n=== VM {vm_idx + 1}/{N_VMS}: provisioning ===")

    handle = INFRA.launch(WAA_WINDOWS_RESOURCE)
    base = handle.endpoint
    record["vm_ip"] = base
    print(f"VM up at {base}")
    record["vm_up_at"] = time.time()

    consecutive_ok = 0
    first_502_at: float | None = None
    first_200_at: float | None = None
    failure_snapshot: dict | None = None
    attempts: list[dict] = []

    poll_start = time.time()
    attempt_idx = 0
    while time.time() - poll_start < MAX_POLL_S:
        status, elapsed, body = try_upload(base, attempt_idx)
        ts = time.time() - poll_start
        attempts.append({"t": ts, "status": status, "elapsed_s": elapsed, "body_preview": body[:80]})
        print(f"  t={ts:6.2f}s  upload[{attempt_idx:3d}] → {status} ({elapsed:.2f}s) {body[:60]}")

        if status == 502 and first_502_at is None:
            first_502_at = ts
            try:
                failure_snapshot = snapshot_vm_state(base)
                print(f"  [snapshot taken at t={ts:.2f}s]")
            except Exception as exc:
                failure_snapshot = {"error": str(exc)}

        if status == 200:
            if first_200_at is None:
                first_200_at = ts
            consecutive_ok += 1
            if consecutive_ok >= CONSECUTIVE_OK_TARGET:
                print(f"  ✓ settled (5 consecutive 200s, first at t={first_200_at:.2f}s)")
                break
        else:
            consecutive_ok = 0

        attempt_idx += 1
        time.sleep(POLL_INTERVAL_S)
    else:
        print(f"  ✗ never settled within {MAX_POLL_S}s")

    record["first_502_at_s"] = first_502_at
    record["first_200_at_s"] = first_200_at
    record["settled"] = consecutive_ok >= CONSECUTIVE_OK_TARGET
    record["race_window_s"] = (first_200_at - first_502_at) if (first_200_at and first_502_at) else None
    record["total_attempts"] = len(attempts)
    record["attempts"] = attempts
    record["failure_snapshot"] = failure_snapshot

    print(f"=== VM {vm_idx + 1}: tearing down ===")
    handle.close()
    record["finished_at"] = time.time()
    return record


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for i in range(N_VMS):
        rec = probe_one_vm(i)
        out = OUT_DIR / f"race-{i + 1}.json"
        out.write_text(json.dumps(rec, indent=2))
        results.append(rec)
        print(f"saved → {out}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        s = "settled" if r["settled"] else "STUCK"
        win = f"{r['race_window_s']:.1f}s" if r["race_window_s"] is not None else "no 502s observed"
        first502 = f"{r['first_502_at_s']:.1f}s" if r["first_502_at_s"] is not None else "—"
        first200 = f"{r['first_200_at_s']:.1f}s" if r["first_200_at_s"] is not None else "—"
        print(f"  VM {r['vm_idx'] + 1}: {s}  first502@{first502}  first200@{first200}  window={win}")


if __name__ == "__main__":
    main()
