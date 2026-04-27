"""Same probe as probe_lo_image.py but pointed at the OLD -kusha gallery image
(known good — yesterday's chrome eval used it). If this returns all apps present,
the new -kusha-lo image is genuinely broken. If it also returns all-NO, the probe
itself is broken.
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
    image_name_suffix="-kusha",  # OLD image — yesterday's chrome eval used this
    source_cache_blob="sources/waa-windows-prepared.qcow2",
)

APP_PATHS = {
    "LibreOffice": r"C:\Program Files\LibreOffice\program\soffice.exe",
    "VLC":         r"C:\Program Files\VideoLAN\VLC\vlc.exe",
    "GIMP":        r"C:\Program Files\GIMP 2\bin\gimp-2.10.exe",
    "Chrome":      r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "Edge":        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "7zip":        r"C:\Program Files\7-Zip\7z.exe",
    "Git":         r"C:\Program Files\Git\bin\git.exe",
    "Thunderbird": r"C:\Program Files\Mozilla Thunderbird\thunderbird.exe",
}


def post_execute(base: str, command: list[str], shell: bool = True) -> dict:
    payload = json.dumps({"command": command, "shell": shell})
    try:
        r = requests.post(f"{base}/execute", data=payload,
                          headers={"Content-Type": "application/json"}, timeout=60)
        return r.json() if r.status_code == 200 else {"http_status": r.status_code, "text": r.text[:300]}
    except Exception as exc:
        return {"error": str(exc)}


def main() -> None:
    INFRA.provision(WAA_WINDOWS_RESOURCE)  # cache-hit, already registered
    handle = INFRA.launch(WAA_WINDOWS_RESOURCE)
    base = handle.endpoint
    print(f"\n=== OLD -kusha VM up at {base} ===\n")
    try:
        time.sleep(45)
        sanity = post_execute(base, ["cmd", "/c", "echo HELLO && dir C:\\ /B"])
        print(f"--- C:\\ listing ---\n{(sanity.get('output') or '').strip()}\n")

        results: dict[str, bool] = {}
        for name, path in APP_PATHS.items():
            r = post_execute(base, ["cmd", "/c", f'if exist "{path}" (echo YES) else (echo NO)'])
            out = (r.get("output") or "").strip().upper()
            present = out.startswith("YES")
            results[name] = present
            print(f"  {name:<12} {'YES' if present else 'NO '}  {path}")

        missing = [n for n, ok in results.items() if not ok]
        print()
        if missing:
            print(f"OLD image FAIL — missing: {missing}")
        else:
            print("OLD image PASS — all 8 apps present (so the NEW -kusha-lo image is broken)")
    finally:
        print("\n=== tearing down ===")
        handle.close()


if __name__ == "__main__":
    main()
