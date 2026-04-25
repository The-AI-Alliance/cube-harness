"""One-shot probe: figure out why chrome's CDP returns 502 on Windows VM.

Spins up one WAA VM, fires the same launch step the eval does
(`start chrome --remote-debugging-port=1337`), then probes from inside the VM
what's actually responding on port 1337 and where the 502 comes from. Tears
down the VM at the end.

Usage:
    uv run recipes/waa/probe_chrome_502.py
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
    image_name_suffix="-kusha",
    source_cache_blob="sources/waa-windows-prepared.qcow2",
)


def post_launch(base_url: str, command: list[str]) -> dict | None:
    """Fire-and-forget launch via guest agent /setup/launch."""
    payload = json.dumps({"command": command, "shell": False})
    try:
        r = requests.post(
            f"{base_url}/setup/launch",
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        return {"status": r.status_code, "body": r.text[:200]}
    except Exception as exc:
        return {"error": str(exc)}


def post_execute(base_url: str, command: list[str], shell: bool = True) -> dict | None:
    """Run a command and return its output via guest agent /execute."""
    payload = json.dumps({"command": command, "shell": shell})
    try:
        r = requests.post(
            f"{base_url}/execute",
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        return {"status_code": r.status_code, "text": r.text[:500]}
    except Exception as exc:
        return {"error": str(exc)}


def show(label: str, result) -> None:
    print(f"\n--- {label} ---")
    if isinstance(result, dict):
        out = result.get("output", "") or ""
        err = result.get("error", "") or ""
        rc = result.get("returncode")
        if rc is not None:
            print(f"  returncode={rc}")
        if out:
            print(f"  stdout: {out[:1000]}")
        if err:
            print(f"  stderr: {err[:500]}")
        if not out and not err:
            print(f"  raw: {result}")
    else:
        print(f"  {result}")


def main() -> None:
    handle = INFRA.launch(WAA_WINDOWS_RESOURCE)
    base = handle.endpoint
    print(f"\n=== VM up at {base} ===")

    try:
        # 0. Sanity — what does Windows say is on port 1337 BEFORE we launch chrome?
        show("netstat 1337 (before chrome)",
             post_execute(base, ["cmd", "/c", "netstat -an | findstr 1337"]))

        # 1. Launch chrome (same command our task config uses)
        print("\n--- launching: start chrome --remote-debugging-port=1337 ---")
        print(f"  {post_launch(base, ['start', 'chrome', '--remote-debugging-port=1337'])}")

        # 2. Wait a few seconds, then poll what's happening
        for delay in (5, 15, 30, 60):
            print(f"\n=== T+{delay}s after chrome launch ===")
            time.sleep(delay if delay == 5 else delay - (5 if delay == 15 else (15 if delay == 30 else 30)))

            show("tasklist | findstr chrome",
                 post_execute(base, ["cmd", "/c", "tasklist | findstr chrome.exe"]))

            show("netstat -an | findstr 1337",
                 post_execute(base, ["cmd", "/c", "netstat -an | findstr 1337"]))

            show("netstat -an | findstr 9222 (socat target?)",
                 post_execute(base, ["cmd", "/c", "netstat -an | findstr 9222"]))

            show("curl -i http://localhost:1337/json/version",
                 post_execute(base, ["cmd", "/c", "curl -i --max-time 5 http://localhost:1337/json/version"]))

            # Also try port 9222 (the socat-forwarded one — if socat is running)
            show("curl -i http://localhost:9222/json/version",
                 post_execute(base, ["cmd", "/c", "curl -i --max-time 5 http://localhost:9222/json/version"]))
    finally:
        print("\n=== tearing down ===")
        handle.close()


if __name__ == "__main__":
    main()
