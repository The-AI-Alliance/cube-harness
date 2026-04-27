#!/usr/bin/env python3
"""Boot the existing Packer overlay and finish the build manually.

Why this exists: Packer's WSMan file uploader runs at ~5 KB/s on this host
because guest powershell.exe cold-starts under load take ~3 minutes per
chunk. Uploading a 15 MB MSI takes ~75 minutes; a 350 MB MSI is hours.

This script bypasses WSMan PUT entirely:
  * Hosts ~/.cube/cache/ + packer/scripts/ over HTTP on the host
  * Boots the existing overlay qcow2 (which has WinRM-bootstrapped Windows)
  * Uses curl on the guest to pull artifacts via QEMU usermode → 250 KB/s
  * Runs each install script via a one-shot Scheduled Task as Docker (admin)
    to get a non-UAC-split token
  * Issues a graceful shutdown
  * The overlay file is now the final prepared image — no commit needed,
    Packer's `output_directory` semantics already give us a standalone
    qcow2 with the WinRM-base as the backing file.

Result: prepared.qcow2 ready for HuggingFace upload.
"""

from __future__ import annotations

import http.server
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import winrm

REPO = Path(__file__).resolve().parent
OVERLAY = REPO / "output-waa-prepared/waa-windows-prepared.qcow2"
CACHE = Path.home() / ".cube/cache"
SCRIPTS = REPO / "scripts"
SSH_PUBKEY = Path.home() / ".ssh/id_ed25519.pub"
PASSWORD = (Path.home() / ".cube/waa-build-admin-password.txt").read_text().strip()

WINRM_PORT = 17585
HTTP_PORT = 17080

PROVISIONER_SCRIPTS = [
    "install-openssh-server.ps1",
    "install-azure-vm-agent.ps1",
    "drop-authorized-keys.ps1",
    "configure-autologon.ps1",
]

CACHED_FILES = ["OpenSSH-Win64.zip", "WindowsAzureVmAgent.msi"]


def log(msg: str) -> None:
    print(f"[manual-finish] {msg}", flush=True)


def start_http_server(serve_dir: Path) -> tuple[socketserver.TCPServer, threading.Thread]:
    def handler(*a: object, **kw: object) -> http.server.SimpleHTTPRequestHandler:
        return http.server.SimpleHTTPRequestHandler(*a, directory=str(serve_dir), **kw)

    httpd = socketserver.ThreadingTCPServer(("0.0.0.0", HTTP_PORT), handler)
    httpd.allow_reuse_address = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t


def stage_serve_dir() -> Path:
    """Build a single dir merging cache + scripts so HTTP_PORT serves both."""
    serve = Path(tempfile.mkdtemp(prefix="manual-finish-serve-"))
    for f in CACHED_FILES:
        src = CACHE / f
        if src.exists():
            (serve / f).symlink_to(src)
    for f in PROVISIONER_SCRIPTS:
        src = SCRIPTS / f
        (serve / f).symlink_to(src)
    serve_pubkey = serve / "id_rsa.pub"
    serve_pubkey.symlink_to(SSH_PUBKEY)
    return serve


def boot_qemu(workdir: Path) -> tuple[subprocess.Popen, subprocess.Popen]:
    tpm_dir = workdir / "tpm"
    tpm_dir.mkdir(parents=True, exist_ok=True)
    tpm_sock = tpm_dir / "sock"
    swtpm = subprocess.Popen(
        [
            "swtpm",
            "socket",
            "--tpmstate",
            f"dir={tpm_dir}",
            "--ctrl",
            f"type=unixio,path={tpm_sock}",
            "--tpm2",
            "--log",
            f"file={tpm_dir}/swtpm.log,level=20",
        ]
    )
    for _ in range(50):
        if tpm_sock.exists():
            break
        time.sleep(0.1)
    else:
        swtpm.kill()
        raise RuntimeError("swtpm sock never appeared")

    ovmf_vars = workdir / "OVMF_VARS.fd"
    shutil.copy("/usr/share/OVMF/OVMF_VARS_4M.ms.fd", ovmf_vars)

    qemu = subprocess.Popen(
        [
            "qemu-system-x86_64",
            "-cpu",
            "host",
            "-enable-kvm",
            "-machine",
            "q35,smm=on",
            "-m",
            "8192",
            "-smp",
            "8",
            "-drive",
            f"file={OVERLAY},format=qcow2,if=virtio,cache=writeback,discard=ignore",
            "-drive",
            "if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.ms.fd",
            "-drive",
            f"if=pflash,format=raw,file={ovmf_vars}",
            "-chardev",
            f"socket,id=chrtpm,path={tpm_sock}",
            "-tpmdev",
            "emulator,id=tpm0,chardev=chrtpm",
            "-device",
            "tpm-tis,tpmdev=tpm0",
            "-netdev",
            f"user,id=u0,hostfwd=tcp:127.0.0.1:{WINRM_PORT}-:5985",
            "-device",
            "virtio-net,netdev=u0",
            "-display",
            "none",
        ]
    )
    return swtpm, qemu


def wait_for_winrm(timeout: int = 600) -> winrm.Session:
    log(f"polling WinRM at 127.0.0.1:{WINRM_PORT} (up to {timeout}s)…")
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            s = winrm.Session(
                f"http://127.0.0.1:{WINRM_PORT}/wsman",
                auth=("Docker", PASSWORD),
                transport="basic",
            )
            # pywinrm Session has no timeout kwarg — relies on protocol defaults.
            # The bottleneck before WinRM is up is just connection refused / 404,
            # both of which fail fast. So we just retry on exception.
            r = s.run_cmd("echo hi")
            if r.status_code == 0 and b"hi" in r.std_out:
                log("WinRM ready")
                return s
        except Exception as exc:
            last_err = exc
        time.sleep(10)
    raise TimeoutError(f"WinRM never came up within {timeout}s; last error: {last_err}")


def guest_curl(s: winrm.Session, name: str) -> None:
    """Download a file from host HTTP into C:\\Windows\\Temp\\<name>."""
    dst = f"C:\\Windows\\Temp\\{name}"
    cmd = f"curl -sf --max-time 1200 -o {dst} http://10.0.2.2:{HTTP_PORT}/{name}"
    log(f"curl-ing {name} → {dst}")
    r = s.run_cmd(cmd)
    if r.status_code != 0:
        raise RuntimeError(f"curl {name} failed rc={r.status_code} stderr={r.std_err[:300]!r}")
    # Verify
    r2 = s.run_cmd(f"dir {dst} 2>NUL | findstr /v Volume | findstr /v Direct")
    log(r2.std_out.decode().strip()[-200:])


def run_elevated_script(s: winrm.Session, script_name: str) -> None:
    """Run a provisioner script via WMI Win32_Process.Create as the Docker
    user (admin in WAA setup), capturing stdout/stderr to log files we then
    read back. WMI fully detaches from Flask's handle chain so subsequent
    WinRM commands aren't blocked.
    """
    import base64

    label = script_name.replace(".ps1", "")
    log(f"=== {label} ===")
    src_url = f"http://10.0.2.2:{HTTP_PORT}/{script_name}"
    local = f"C:\\Windows\\Temp\\{script_name}"
    out_log = f"C:\\Windows\\Temp\\{label}.out"
    done = f"C:\\Windows\\Temp\\{label}.done"
    # Pull script to disk
    r = s.run_cmd(f"curl -sf --max-time 60 -o {local} {src_url}")
    if r.status_code != 0:
        raise RuntimeError(f"curl {script_name} failed: {r.std_err[:300]!r}")
    s.run_cmd(f"del {done} 2>NUL & del {out_log} 2>NUL")
    # Inner PS: set env vars (configure-autologon.ps1 needs ADMIN_USER/PASSWORD),
    # then run the script and write a sentinel with its exit code. Use base64
    # encoding to avoid all the cmdline escape headaches.
    pw_escaped = PASSWORD.replace("'", "''")
    inner = (
        f'$env:ADMIN_USER = "Docker"; '
        f"$env:ADMIN_PASSWORD = '{pw_escaped}'; "
        f'$ErrorActionPreference = "Continue"; '
        f'powershell -NoProfile -ExecutionPolicy Bypass -File "{local}" '
        f'*> "{out_log}"; '
        f'"OK exit=$LASTEXITCODE" | Out-File -Encoding ascii "{done}"'
    )
    encoded = base64.b64encode(inner.encode("utf-16-le")).decode()
    spawn_cmd = f"powershell -NoProfile -NonInteractive -EncodedCommand {encoded}"
    # Wrap the spawn itself in a tiny PS that calls Win32_Process.Create.
    spawn_ps = (
        f'$r = ([WMICLASS]"\\\\.\\ROOT\\CIMV2:Win32_Process").Create("{spawn_cmd}"); '
        f'"PID=$($r.ProcessId) RV=$($r.ReturnValue)"'
    )
    r = s.run_ps(spawn_ps)
    log(f"  spawn rc={r.status_code} out={r.std_out.decode().strip()[:300]}")
    if r.status_code != 0:
        raise RuntimeError(f"spawn failed for {label}: stderr={r.std_err.decode()[:500]!r}")
    # Poll the sentinel
    log(f"  polling for {done}...")
    deadline = time.time() + 1800
    while time.time() < deadline:
        try:
            r = s.run_cmd(f"if exist {done} (type {done}) else (echo pending)")
            status = r.std_out.decode().strip()
        except Exception as exc:
            log(f"  poll transient: {exc}")
            time.sleep(15)
            continue
        if status.startswith("OK"):
            log(f"  done: {status}")
            try:
                r2 = s.run_cmd(f"type {out_log} 2>NUL")
                log(f"  ---{label}.out (last 25 lines)---")
                for line in r2.std_out.decode().splitlines()[-25:]:
                    log(f"  {line}")
            except Exception:
                pass
            # Sanity: zero exit code only
            if "exit=0" not in status:
                raise RuntimeError(f"{label} returned non-zero: {status}")
            return
        time.sleep(15)
    raise TimeoutError(f"{label} never produced sentinel file in 30 min")


def main() -> int:
    if not OVERLAY.exists():
        log(f"OVERLAY missing: {OVERLAY}")
        return 1
    log(f"using overlay: {OVERLAY} ({OVERLAY.stat().st_size // (1024 * 1024)} MiB)")

    serve_dir = stage_serve_dir()
    log(f"serving {serve_dir} on http://0.0.0.0:{HTTP_PORT}")
    httpd, _ = start_http_server(serve_dir)

    workdir = Path(tempfile.mkdtemp(prefix="manual-finish-"))
    log(f"workdir: {workdir}")
    swtpm = qemu = None
    try:
        swtpm, qemu = boot_qemu(workdir)
        log(f"swtpm pid={swtpm.pid}, qemu pid={qemu.pid}")
        s = wait_for_winrm()
        # Pull cached files
        for f in CACHED_FILES:
            guest_curl(s, f)
        # Pull pubkey
        guest_curl(s, "id_rsa.pub")
        # Run provisioner scripts
        for ps in PROVISIONER_SCRIPTS:
            run_elevated_script(s, ps)
        # Graceful shutdown
        log("shutdown /s /f /t 5")
        s.run_cmd("shutdown /s /f /t 5")
        log("waiting for QEMU to exit (up to 5 min)…")
        try:
            qemu.wait(timeout=300)
        except subprocess.TimeoutExpired:
            log("QEMU did not exit cleanly; SIGTERM")
            qemu.terminate()
            qemu.wait(timeout=30)
        log(f"DONE. Final image: {OVERLAY} ({OVERLAY.stat().st_size // (1024 * 1024)} MiB)")
        return 0
    finally:
        if qemu and qemu.poll() is None:
            qemu.kill()
        if swtpm and swtpm.poll() is None:
            swtpm.terminate()
            try:
                swtpm.wait(timeout=10)
            except subprocess.TimeoutExpired:
                swtpm.kill()
        httpd.shutdown()
        shutil.rmtree(serve_dir, ignore_errors=True)
        # Keep workdir for debugging


if __name__ == "__main__":
    sys.exit(main())
