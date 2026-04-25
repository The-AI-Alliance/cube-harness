"""One-shot probe: launch a WAA VM, ask it whether socat is installed, tear down.

Usage:
    uv run recipes/waa/probe_socat.py
"""

import json
import logging
import os
import sys
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
    image_name_suffix="-kusha",
    source_cache_blob="sources/waa-windows-prepared.qcow2",
)


def post_execute(base_url: str, command: list[str], shell: bool) -> dict | None:
    payload = json.dumps({"command": command, "shell": shell})
    try:
        r = requests.post(f"{base_url}/execute", data=payload,
                          headers={"Content-Type": "application/json"}, timeout=30)
        if r.status_code == 200:
            return r.json()
        return {"status_code": r.status_code, "text": r.text[:500]}
    except Exception as exc:
        return {"error": str(exc)}


def main() -> None:
    handle = INFRA.launch(WAA_WINDOWS_RESOURCE)
    try:
        base = handle.endpoint
        print(f"\n=== VM up at {base} ===\n")

        # 1. where.exe socat (Windows native)
        out1 = post_execute(base, ["where", "socat"], shell=True)
        print(f"[where socat] -> {out1}")

        # 2. powershell Get-Command socat
        out2 = post_execute(base, ["powershell", "-Command", "Get-Command socat 2>&1 | Out-String"], shell=False)
        print(f"[Get-Command socat] -> {out2}")

        # 3. directly try to invoke socat -V
        out3 = post_execute(base, ["socat", "-V"], shell=True)
        print(f"[socat -V] -> {out3}")

        # 4. cygwin/git-bash bin paths
        for p in [r"C:\Program Files\Git\usr\bin\socat.exe",
                  r"C:\cygwin64\bin\socat.exe",
                  r"C:\msys64\usr\bin\socat.exe"]:
            o = post_execute(base, ["powershell", "-Command", f"Test-Path '{p}'"], shell=False)
            print(f"[Test-Path {p}] -> {o}")
    finally:
        print("\n=== tearing down ===")
        handle.close()


if __name__ == "__main__":
    main()
