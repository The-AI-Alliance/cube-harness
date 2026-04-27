"""Probe v3 — test the user-data-dir hypothesis + chrome policy registry.

Hypothesis from probe v2: chrome silently ignores --remote-debugging-port=1337
on this Windows VM. Likely causes:
  1. Default user-data-dir is in some state that suppresses CDP
  2. A Chrome group policy disables remote debugging
  3. Some other invocation issue

This probe tests:
  D0. Query Chrome policy registry — is RemoteDebuggingAllowed disabled?
  D1. Kill chrome, then chrome.exe with --remote-debugging-port=1337
      AND --user-data-dir=C:\\Temp\\cdp1 AND --no-first-run.
  D2. After D1, did 1337 bind? Does /json/version work on 1337?
  D3. Try a third variation if D1 fails: same plus --no-default-browser-check
      and --headless=new.

Usage:
    uv run recipes/waa/probe_chrome_502_v3.py
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


def post_execute(base: str, command: list[str], shell: bool = True) -> dict:
    payload = json.dumps({"command": command, "shell": shell})
    try:
        r = requests.post(f"{base}/execute", data=payload, headers={"Content-Type": "application/json"}, timeout=60)
        return r.json() if r.status_code == 200 else {"http_status": r.status_code, "text": r.text[:300]}
    except Exception as exc:
        return {"error": str(exc)}


def post_launch(base: str, command: list[str]) -> dict:
    payload = json.dumps({"command": command, "shell": False})
    try:
        r = requests.post(
            f"{base}/setup/launch", data=payload, headers={"Content-Type": "application/json"}, timeout=30
        )
        return {"http_status": r.status_code, "body": r.text[:300]}
    except Exception as exc:
        return {"error": str(exc)}


def show(label: str, r: dict) -> None:
    print(f"\n--- {label} ---")
    out = (r or {}).get("output", "") or ""
    err = (r or {}).get("error", "") or ""
    rc = (r or {}).get("returncode")
    if rc is not None:
        print(f"  rc={rc}")
    if out:
        print(f"  out: {out[:1500].rstrip()}")
    if err:
        print(f"  err: {err[:300].rstrip()}")
    if not out and not err and rc is None:
        print(f"  raw: {r}")


def main() -> None:
    handle = INFRA.launch(WAA_WINDOWS_RESOURCE)
    base = handle.endpoint
    print(f"\n=== VM up at {base} ===\n")
    try:
        # D0 — Chrome policy registry
        print("=== D0: Chrome policy registry ===")
        show(
            "HKLM Policies\\Google\\Chrome",
            post_execute(base, ["cmd", "/c", r'reg query "HKLM\SOFTWARE\Policies\Google\Chrome" /s 2>NUL']),
        )
        show(
            "HKLM Policies\\Chromium",
            post_execute(base, ["cmd", "/c", r'reg query "HKLM\SOFTWARE\Policies\Chromium" /s 2>NUL']),
        )
        show(
            "HKCU Policies\\Google\\Chrome",
            post_execute(base, ["cmd", "/c", r'reg query "HKCU\SOFTWARE\Policies\Google\Chrome" /s 2>NUL']),
        )

        # D1 — kill chrome, launch with --user-data-dir
        print("\n=== D1: kill chrome, then chrome.exe with --user-data-dir + CDP ===")
        show("taskkill chrome", post_execute(base, ["cmd", "/c", "taskkill /IM chrome.exe /F /T"]))
        time.sleep(3)
        # Make a fresh user-data-dir
        show(
            "create user-data-dir",
            post_execute(
                base,
                [
                    "cmd",
                    "/c",
                    r"if not exist C:\Temp mkdir C:\Temp & rmdir /s /q C:\Temp\cdp1 2>NUL & mkdir C:\Temp\cdp1",
                ],
            ),
        )
        # Launch chrome.exe DIRECT (no `start`) with the right flags. Use `start /b` to detach.
        launch_args = [
            "cmd",
            "/c",
            "start",
            "",
            "/b",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "--remote-debugging-port=1337",
            r"--user-data-dir=C:\Temp\cdp1",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        print(f"  launch args: {launch_args}")
        print(f"  result: {post_launch(base, launch_args)}")
        time.sleep(15)

        show("D1 chrome.exe processes", post_execute(base, ["cmd", "/c", "tasklist | findstr chrome.exe"]))
        show("D1 netstat :1337", post_execute(base, ["cmd", "/c", "netstat -an | findstr :1337"]))
        show(
            "D1 curl localhost:1337/json/version",
            post_execute(base, ["cmd", "/c", "curl -i --max-time 5 http://localhost:1337/json/version"]),
        )

        # D2 — wait longer (chrome may need more startup time)
        print("\n=== D2: wait another 30s and recheck ===")
        time.sleep(30)
        show("D2 netstat :1337", post_execute(base, ["cmd", "/c", "netstat -an | findstr :1337"]))
        show(
            "D2 curl localhost:1337/json/version",
            post_execute(base, ["cmd", "/c", "curl -i --max-time 5 http://localhost:1337/json/version"]),
        )
        show(
            "D2 curl localhost:9222/json/version (via Caddy)",
            post_execute(base, ["cmd", "/c", "curl -i --max-time 5 http://localhost:9222/json/version"]),
        )

        # D3 — last-resort: try --headless=new with --user-data-dir
        print("\n=== D3: kill chrome, retry with --headless=new ===")
        show("D3 taskkill chrome", post_execute(base, ["cmd", "/c", "taskkill /IM chrome.exe /F /T"]))
        time.sleep(3)
        show(
            "D3 wipe user-data-dir",
            post_execute(base, ["cmd", "/c", r"rmdir /s /q C:\Temp\cdp2 2>NUL & mkdir C:\Temp\cdp2"]),
        )
        launch_args2 = [
            "cmd",
            "/c",
            "start",
            "",
            "/b",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "--remote-debugging-port=1337",
            r"--user-data-dir=C:\Temp\cdp2",
            "--headless=new",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        print(f"  launch args: {launch_args2}")
        print(f"  result: {post_launch(base, launch_args2)}")
        time.sleep(15)
        show("D3 chrome.exe processes", post_execute(base, ["cmd", "/c", "tasklist | findstr chrome.exe"]))
        show("D3 netstat :1337", post_execute(base, ["cmd", "/c", "netstat -an | findstr :1337"]))
        show(
            "D3 curl localhost:1337/json/version",
            post_execute(base, ["cmd", "/c", "curl -i --max-time 5 http://localhost:1337/json/version"]),
        )
    finally:
        print("\n=== tearing down ===")
        handle.close()


if __name__ == "__main__":
    main()
