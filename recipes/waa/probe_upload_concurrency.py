"""Diagnostic: isolate whether /setup/upload 502s are
  (a) concurrency-driven (multiple VMs simultaneously hammering /upload), or
  (b) task-content-specific (some specific files trigger Flask/Defender 502s).

Method:
1. Provision ONE fresh VM, settled
2. PHASE A — sequential upload of 10 different task files (varied content + paths)
3. PHASE B — same 10 files uploaded CONCURRENTLY (10 threads in parallel)
4. PHASE C — settled VM (post B), upload 30x the failing-task file in tight loop

If A is clean and B has 502s → concurrency on a single Flask instance triggers it
If A has 502s on specific files → content-specific
If C has 502s on already-settled VM → cumulative state issue (file count, fragmentation)

Also tries the NEXT scenario the eval simulates: a fresh VM that gets hit
with a multi-file upload + execute + open in quick succession.

Output: /tmp/cube-harness-logs/concurrency-probe.json

Usage:
    uv run recipes/waa/probe_upload_concurrency.py
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

OUT = Path("/tmp/cube-harness-logs/concurrency-probe.json")

# Real task files to test — sourced from the LO smoke task configs we know failed.
# All in the same destination dir to mirror real eval behavior.
TASK_FILES = [
    ("https://raw.githubusercontent.com/rogeriobonatti/winarenafiles/main/task_files/calc/01b269ae-2111-4a07-81fd-3fcd711993b0-WOS/config/Student_Level_Fill_Blank.xlsx",
     "C:\\Users\\Docker\\Downloads\\Student_Level_Fill_Blank.xlsx"),
    ("https://raw.githubusercontent.com/rogeriobonatti/winarenafiles/main/task_files/calc/035f41ba-6653-43ab-aa63-c86d449d62e5-WOS/config/IncomeStatement2.xlsx",
     "C:\\Users\\Docker\\Downloads\\IncomeStatement2.xlsx"),
    ("https://raw.githubusercontent.com/rogeriobonatti/winarenafiles/main/task_files/calc/04d9aeaf-7bed-4024-bedb-e10e6f00eb7f-WOS/config/SmallBalanceSheet.xlsx",
     "C:\\Users\\Docker\\Downloads\\SmallBalanceSheet.xlsx"),
    ("https://raw.githubusercontent.com/rogeriobonatti/winarenafiles/main/task_files/writer/0810415c-bde4-4443-9047-d5f70165a697-WOS/config/Novels_Intro_Packet.docx",
     "C:\\Users\\Docker\\Downloads\\Novels_Intro_Packet.docx"),
    ("https://raw.githubusercontent.com/rogeriobonatti/winarenafiles/main/task_files/writer/0a0faba3-5580-44df-965d-f562a99b291c-WOS/config/04%20CHIN9505%20EBook%20Purchasing%20info%202021%20Jan.docx",
     "C:\\Users\\Docker\\Downloads\\04 CHIN9505 EBook Purchasing info 2021 Jan.docx"),
    ("https://raw.githubusercontent.com/rogeriobonatti/winarenafiles/main/task_files/calc/0a2e43bf-b26c-4631-a966-af9dfa12c9e5-WOS/config/SalesRep.xlsx",
     "C:\\Users\\Docker\\Downloads\\SalesRep.xlsx"),
    ("https://raw.githubusercontent.com/rogeriobonatti/winarenafiles/main/task_files/calc/0acbd372-ca7a-4507-b949-70673120190f-WOS/config/NetIncome.xlsx",
     "C:\\Users\\Docker\\Downloads\\NetIncome.xlsx"),
    ("https://raw.githubusercontent.com/rogeriobonatti/winarenafiles/main/task_files/writer/0b17a146-2934-46c7-8727-73ff6b6483e8-WOS/config/H2O_Factsheet_WA.docx",
     "C:\\Users\\Docker\\Downloads\\H2O_Factsheet_WA.docx"),
    ("https://raw.githubusercontent.com/rogeriobonatti/winarenafiles/main/task_files/writer/0e47de2a-32e0-456c-a366-8c607ef7a9d2-WOS/LibreOffice_Open_Source_Word_Processing.docx",
     "C:\\Users\\Docker\\Downloads\\LibreOffice_Open_Source_Word_Processing.docx"),
    ("https://raw.githubusercontent.com/rogeriobonatti/winarenafiles/main/task_files/writer/0e763496-b6bb-4508-a427-fad0b6c3e195-WOS/Dublin_Zoo_Intro.docx",
     "C:\\Users\\Docker\\Downloads\\Dublin_Zoo_Intro.docx"),
]


def fetch(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content


def upload(base: str, data: bytes, dest_path: str, name: str, idx: int) -> dict:
    form = MultipartEncoder({
        "file_path": dest_path,
        "file_data": (name, io.BytesIO(data), "application/octet-stream"),
    })
    t0 = time.time()
    try:
        r = requests.post(
            f"{base}/setup/upload",
            headers={"Content-Type": form.content_type},
            data=form,
            timeout=60,
        )
        return {"idx": idx, "status": r.status_code, "elapsed_s": time.time() - t0,
                "body_preview": r.text[:80], "size_bytes": len(data), "dest": dest_path}
    except Exception as exc:
        return {"idx": idx, "status": -1, "elapsed_s": time.time() - t0,
                "body_preview": f"EXC {type(exc).__name__}: {exc}", "size_bytes": len(data), "dest": dest_path}


def main() -> None:
    print("Pre-fetching task files...")
    files = []
    for url, dest in TASK_FILES:
        data = fetch(url)
        files.append((data, dest, os.path.basename(dest)))
        print(f"  {os.path.basename(dest):<55} {len(data):>8} bytes")

    print("\nProvisioning fresh VM...")
    handle = INFRA.launch(WAA_WINDOWS_RESOURCE)
    base = handle.endpoint
    print(f"VM up at {base}\n")

    record: dict = {"phases": {}}

    try:
        # PHASE A: sequential upload of all 10 files
        print("=== PHASE A — sequential upload of 10 task files ===")
        seq_results = []
        for idx, (data, dest, name) in enumerate(files):
            r = upload(base, data, dest, name, idx)
            seq_results.append(r)
            print(f"  [{idx}] {r['status']} {r['elapsed_s']:.2f}s  {name} ({r['size_bytes']}B)")
        record["phases"]["A_sequential"] = seq_results
        a_502 = sum(1 for r in seq_results if r["status"] == 502)
        a_200 = sum(1 for r in seq_results if r["status"] == 200)
        print(f"\n  PHASE A summary: {a_200} OK / {a_502} 502 / {len(seq_results)} total\n")

        # PHASE B: concurrent upload of all 10 files (10 threads)
        print("=== PHASE B — concurrent upload (10 threads) ===")
        # Use different dest paths to avoid file-write conflicts
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = []
            for idx, (data, dest, name) in enumerate(files):
                # Replace dest filename with concurrent-suffix
                concurrent_dest = dest.replace(".xlsx", f"_c{idx}.xlsx").replace(".docx", f"_c{idx}.docx")
                futures.append(ex.submit(upload, base, data, concurrent_dest, name, idx))
            conc_results = [f.result() for f in futures]
        for r in sorted(conc_results, key=lambda x: x["idx"]):
            print(f"  [{r['idx']}] {r['status']} {r['elapsed_s']:.2f}s  {os.path.basename(r['dest'])}")
        record["phases"]["B_concurrent"] = conc_results
        b_502 = sum(1 for r in conc_results if r["status"] == 502)
        b_200 = sum(1 for r in conc_results if r["status"] == 200)
        print(f"\n  PHASE B summary: {b_200} OK / {b_502} 502 / {len(conc_results)} total\n")

        # PHASE C: 30 sequential uploads of the same file to settled VM
        print("=== PHASE C — 30 sequential of SmallBalanceSheet (settled VM) ===")
        small_data, _, _ = files[0]
        c_results = []
        for i in range(30):
            r = upload(base, small_data, f"C:\\Users\\Docker\\Downloads\\loop_{i}.xlsx", f"loop_{i}.xlsx", i)
            c_results.append(r)
            if r["status"] != 200:
                print(f"  [{i}] {r['status']} ({r['body_preview']})")
        record["phases"]["C_settled_loop"] = c_results
        c_502 = sum(1 for r in c_results if r["status"] == 502)
        c_200 = sum(1 for r in c_results if r["status"] == 200)
        print(f"\n  PHASE C summary: {c_200} OK / {c_502} 502 / {len(c_results)} total\n")

        # Overall conclusion
        print("=" * 60)
        print("CONCLUSION")
        print("=" * 60)
        print(f"  A (1 VM seq, 10 files):       {a_200}/{len(seq_results)} OK, {a_502} 502s")
        print(f"  B (1 VM concurrent, 10 thr):  {b_200}/{len(conc_results)} OK, {b_502} 502s")
        print(f"  C (1 VM seq loop, 30 calls):  {c_200}/{len(c_results)} OK, {c_502} 502s")
        if a_502 == 0 and b_502 > 0:
            print("\n  → CONCURRENT pressure on a SINGLE VM Flask triggers 502s")
        elif a_502 > 0:
            print("\n  → CONTENT-SPECIFIC: some files 502 even sequential")
        elif c_502 > 0:
            print("\n  → CUMULATIVE STATE: settled VM accumulates state until 502s")
        else:
            print("\n  → no 502s seen; eval-time concurrency must be the trigger")

    finally:
        OUT.write_text(json.dumps(record, indent=2))
        print(f"\nSaved → {OUT}")
        print("Tearing down...")
        handle.close()


if __name__ == "__main__":
    main()
