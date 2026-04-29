"""Diagnostic: reproduce the eval-time multi-VM cohort 502.

Hypothesis: failures aren't single-VM (probe_setup_upload_race showed clean),
aren't single-VM concurrent (probe_upload_concurrency showed clean),
aren't task-content (clean), aren't cumulative (clean). They MUST be from
N independent VMs all hitting /setup/upload around the same instant.

This probe mimics the eval pattern minimally:
1. Provision N VMs in parallel (default 10)
2. As each VM becomes ready, immediately upload one real task file
3. Tear down

If this reproduces 502s reliably → we have a testbed for the fix.

Output: /tmp/cube-harness-logs/parallel-vm-probe.json

Usage:
    N_VMS=10 uv run recipes/waa/probe_parallel_vms.py
"""

from __future__ import annotations

import concurrent.futures
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

N_VMS = int(os.environ.get("N_VMS", "10"))
OUT = Path("/tmp/cube-harness-logs/parallel-vm-probe.json")

REAL_FILE_URL = "https://raw.githubusercontent.com/rogeriobonatti/winarenafiles/main/task_files/calc/04d9aeaf-7bed-4024-bedb-e10e6f00eb7f-WOS/config/SmallBalanceSheet.xlsx"
REAL_FILE_DEST = "C:\\Users\\Docker\\Downloads\\SmallBalanceSheet.xlsx"


def upload_one_attempt(base: str, data: bytes) -> dict:
    """Single upload attempt, no retry. Captures status, timing, body."""
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


def run_one_vm(vm_idx: int, file_data: bytes) -> dict:
    """Provision a VM, upload the real task file once on first connect, tear down."""
    record: dict = {"vm_idx": vm_idx, "started_at": time.time()}
    try:
        handle = INFRA.launch(WAA_WINDOWS_RESOURCE)
        record["vm_up_at"] = time.time()
        record["vm_endpoint"] = handle.endpoint
        record["provision_s"] = time.time() - record["started_at"]

        # Immediately try the upload — mirror eval's behavior of hitting /upload
        # right after launch returns. No retry — we want to capture the raw 502.
        attempt = upload_one_attempt(handle.endpoint, file_data)
        record["upload_at_s"] = time.time() - record["started_at"]
        record["upload"] = attempt
        record["status"] = attempt["status"]

        # Don't tear down immediately — give 1s in case Flask is still settling
        # then try a follow-up upload to see if it succeeds without waiting
        time.sleep(1.0)
        record["upload_followup"] = upload_one_attempt(handle.endpoint, file_data)
        handle.close()
        record["finished_at"] = time.time()
        return record
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["finished_at"] = time.time()
        return record


def main() -> None:
    print(f"Pre-fetching {REAL_FILE_URL}...")
    file_data = requests.get(REAL_FILE_URL, timeout=30).content
    print(f"  {len(file_data)} bytes")

    print(f"\nProvisioning {N_VMS} VMs IN PARALLEL — mirrors eval cohort pattern")
    print("Each VM: provision → immediately upload → 1s pause → follow-up upload → teardown\n")

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_VMS) as ex:
        futures = [ex.submit(run_one_vm, i, file_data) for i in range(N_VMS)]
        results = []
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            results.append(r)
            status = r.get("status", "ERR")
            up_t = r.get("upload_at_s", -1)
            fu = r.get("upload_followup", {})
            fu_st = fu.get("status", "?")
            err = r.get("error")
            if err:
                print(f"  VM {r['vm_idx']:2d} — ERROR: {err}")
            else:
                print(f"  VM {r['vm_idx']:2d} — first_upload={status} (t={up_t:.1f}s) followup={fu_st}")

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.1f}s")

    n_502_first = sum(1 for r in results if r.get("status") == 502)
    n_200_first = sum(1 for r in results if r.get("status") == 200)
    n_502_fu = sum(1 for r in results if r.get("upload_followup", {}).get("status") == 502)
    n_200_fu = sum(1 for r in results if r.get("upload_followup", {}).get("status") == 200)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  First upload:   {n_200_first}/{N_VMS} OK  ({n_502_first} 502s)")
    print(f"  Follow-up (+1s): {n_200_fu}/{N_VMS} OK  ({n_502_fu} 502s)")
    if n_502_first > 0:
        print("\n  → REPRODUCED: multi-VM cohort triggers 502s")
        if n_502_fu < n_502_first:
            print(f"     and {n_502_first - n_502_fu} of those clear within 1s of follow-up")
    elif n_502_first == 0:
        print("\n  → no 502s; multi-VM-cohort hypothesis weakened. Need different probe pattern.")

    OUT.write_text(json.dumps({"results": results, "summary": {
        "first_502": n_502_first, "first_200": n_200_first,
        "fu_502": n_502_fu, "fu_200": n_200_fu, "n_vms": N_VMS, "wall_time_s": elapsed,
    }}, indent=2))
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
