"""Tighter chrome / socat / Caddy probe — verifies, doesn't assume.

Procedure:
  Phase A — baseline (BEFORE any launch step from us):
      A1. tasklist | findstr chrome.exe
      A2. tasklist | findstr socat
      A3. tasklist | findstr caddy
      A4. netstat | findstr ":1337"
      A5. netstat | findstr ":9222"
      A6. netstat | findstr "LISTENING" (truncated)
      A7. where chrome / where socat / where caddy
      A8. Get-Service caddy / sshd / etc

  Phase B — variation 1: `start chrome --remote-debugging-port=1337` (what we use today)
      B1. fire launch
      B2. wait 30s
      B3. recheck tasklist + netstat 1337/9222
      B4. curl /json/version on 1337 directly + on 9222 (through Caddy)

  Phase C — variation 2: kill chrome, then `chrome.exe --remote-debugging-port=1337` (direct exe)
      C1. taskkill /IM chrome.exe /F (execute, wait for return)
      C2. sleep 3s
      C3. fire launch with direct exe
      C4. wait 30s
      C5. recheck

We do NOT touch socat — we just observe whether it's running and whether 9222 is held by Caddy or socat.

Usage:
    uv run recipes/waa/probe_chrome_502_v2.py
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


def post_execute(base: str, command: list[str], shell: bool = True, timeout: int = 30) -> dict:
    payload = json.dumps({"command": command, "shell": shell})
    try:
        r = requests.post(
            f"{base}/execute",
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json()
        return {"http_status": r.status_code, "text": r.text[:500]}
    except Exception as exc:
        return {"error": str(exc)}


def post_launch(base: str, command: list[str]) -> dict:
    payload = json.dumps({"command": command, "shell": False})
    try:
        r = requests.post(
            f"{base}/setup/launch",
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        return {"http_status": r.status_code, "body": r.text[:200]}
    except Exception as exc:
        return {"error": str(exc)}


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def show(label: str, result: dict) -> None:
    print(f"\n--- {label} ---")
    out = (result or {}).get("output", "") or ""
    err = (result or {}).get("error", "") or ""
    rc = (result or {}).get("returncode")
    if rc is not None:
        print(f"  returncode={rc}")
    if out:
        print(f"  stdout: {out[:1000].rstrip()}")
    if err:
        print(f"  stderr: {err[:300].rstrip()}")
    if not out and not err and rc is None:
        print(f"  raw: {result}")


def baseline(base: str) -> None:
    section("Phase A — baseline (no launch from us yet)")
    show("A1 tasklist | findstr chrome.exe",
         post_execute(base, ["cmd", "/c", "tasklist | findstr chrome.exe"]))
    show("A2 tasklist | findstr socat",
         post_execute(base, ["cmd", "/c", "tasklist | findstr socat"]))
    show("A3 tasklist | findstr caddy",
         post_execute(base, ["cmd", "/c", "tasklist | findstr caddy"]))
    show("A4 netstat | findstr :1337",
         post_execute(base, ["cmd", "/c", "netstat -an | findstr :1337"]))
    show("A5 netstat | findstr :9222",
         post_execute(base, ["cmd", "/c", "netstat -an | findstr :9222"]))
    show("A6 where chrome / chrome.exe",
         post_execute(base, ["cmd", "/c", "where chrome 2>NUL & where chrome.exe 2>NUL"]))
    show("A7 where socat",
         post_execute(base, ["cmd", "/c", "where socat 2>NUL"]))
    show("A8 Get-Service caddy",
         post_execute(base, ["powershell", "-Command", "Get-Service -Name caddy* 2>$null | Format-List Name,Status,DisplayName"], shell=False))
    show("A9 chrome.exe registry path",
         post_execute(base, ["powershell", "-Command", r"Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty '(default)'"], shell=False))


def variation_b(base: str) -> None:
    section("Phase B — `start chrome --remote-debugging-port=1337` (current task config)")
    print(f"  launch: {post_launch(base, ['start', 'chrome', '--remote-debugging-port=1337'])}")
    print("  waiting 30s for chrome to settle...")
    time.sleep(30)
    show("B1 tasklist chrome.exe (after launch)",
         post_execute(base, ["cmd", "/c", "tasklist | findstr chrome.exe"]))
    show("B2 netstat :1337",
         post_execute(base, ["cmd", "/c", "netstat -an | findstr :1337"]))
    show("B3 netstat :9222",
         post_execute(base, ["cmd", "/c", "netstat -an | findstr :9222"]))
    show("B4 curl localhost:1337/json/version",
         post_execute(base, ["cmd", "/c", "curl -i --max-time 5 http://localhost:1337/json/version"]))
    show("B5 curl localhost:9222/json/version (Caddy/socat)",
         post_execute(base, ["cmd", "/c", "curl -i --max-time 5 http://localhost:9222/json/version"]))


def variation_c(base: str) -> None:
    section("Phase C — kill chrome, then `chrome.exe --remote-debugging-port=1337` directly")
    show("C1 taskkill /IM chrome.exe /F",
         post_execute(base, ["cmd", "/c", "taskkill /IM chrome.exe /F /T"]))
    time.sleep(3)
    show("C2 tasklist chrome.exe (post-kill)",
         post_execute(base, ["cmd", "/c", "tasklist | findstr chrome.exe"]))
    # Use cmd /c so we can spawn chrome.exe by name (Windows resolves via App Paths registry)
    print(f"  launch: {post_launch(base, ['cmd', '/c', 'start', '', 'chrome.exe', '--remote-debugging-port=1337'])}")
    print("  waiting 30s for chrome to settle...")
    time.sleep(30)
    show("C3 tasklist chrome.exe (after re-launch)",
         post_execute(base, ["cmd", "/c", "tasklist | findstr chrome.exe"]))
    show("C4 netstat :1337",
         post_execute(base, ["cmd", "/c", "netstat -an | findstr :1337"]))
    show("C5 curl localhost:1337/json/version",
         post_execute(base, ["cmd", "/c", "curl -i --max-time 5 http://localhost:1337/json/version"]))
    show("C6 curl localhost:9222/json/version",
         post_execute(base, ["cmd", "/c", "curl -i --max-time 5 http://localhost:9222/json/version"]))


def main() -> None:
    handle = INFRA.launch(WAA_WINDOWS_RESOURCE)
    base = handle.endpoint
    print(f"\n=== VM up at {base} ===")
    try:
        baseline(base)
        variation_b(base)
        variation_c(base)
    finally:
        print("\n=== tearing down ===")
        handle.close()


if __name__ == "__main__":
    main()
