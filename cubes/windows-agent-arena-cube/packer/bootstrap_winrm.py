#!/usr/bin/env python3
"""One-time (idempotent) WinRM enablement for the WAA base qcow2.

The shipped WAA Windows image has no WinRM server running, so Packer can't
connect to it. This script boots the image in QEMU, uses the Flask guest agent
already running on guest port 5000 to execute ``Enable-PSRemoting`` + firewall
rules inside the VM, shuts the VM down cleanly, and commits the overlay back
into the base image.

After running this once, ``packer/run.sh`` can connect over WinRM on every
subsequent build with no interactive step.

Usage::

    export BASE_IMAGE=~/.cube/images/waa-windows-vm.qcow2
    python packer/bootstrap_winrm.py

Safety:
    - Refuses to run without an existing backup under
      ``~/.cube/images/backups/``  — if you're about to modify the base image
      in place, you need a copy.
    - Operates via a qcow2 overlay, then ``qemu-img commit`` merges the changes
      into the base. If anything fails the overlay is discarded, leaving the
      base unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

logger = logging.getLogger("bootstrap_winrm")

DEFAULT_BASE = Path.home() / ".cube" / "images" / "waa-windows-vm.qcow2"
DEFAULT_BACKUPS = Path.home() / ".cube" / "images" / "backups"
OVMF_CODE = Path("/usr/share/OVMF/OVMF_CODE_4M.ms.fd")
OVMF_VARS = Path("/usr/share/OVMF/OVMF_VARS_4M.ms.fd")

# PowerShell that enables WinRM for Packer. Permissive (basic auth, unencrypted)
# — acceptable because (a) it runs on loopback in build VMs only, and (b)
# sysprep /generalize at the end of the Packer build resets these before any
# Azure launch.
#
# Also sets a password on the Docker user. The WAA upstream image leaves Docker
# with an empty password (see unattend-files/dev_win11x64-enterprise-eval.xml),
# but Windows refuses WinRM network logon for blank-password accounts unless
# LimitBlankPasswordUse=0 is set. Setting a real password is simpler and more
# portable; Packer then uses it as PKR_VAR_admin_password.
_PS_TEMPLATE = r"""
$ErrorActionPreference = 'Stop'

Write-Output '--- setting password on Docker user ---'
$pwd = ConvertTo-SecureString '{password}' -AsPlainText -Force
Set-LocalUser -Name 'Docker' -Password $pwd

Write-Output '--- forcing network profile to Private (required for WinRM firewall) ---'
# Public profile blocks AllowUnencrypted and firewall-rule creation on non-
# private profiles. Flip everything to Private for the build; sysprep resets.
Get-NetConnectionProfile | ForEach-Object {{
    Set-NetConnectionProfile -InterfaceIndex $_.InterfaceIndex -NetworkCategory Private
}}

Write-Output '--- enabling WinRM ---'
Enable-PSRemoting -Force -SkipNetworkProfileCheck | Out-Null
Set-Item WSMan:\localhost\Service\Auth\Basic $true
Set-Item WSMan:\localhost\Service\AllowUnencrypted $true
if (-not (Get-NetFirewallRule -Name WinRM-HTTP-In-TCP -ErrorAction SilentlyContinue)) {{
    New-NetFirewallRule -Name WinRM-HTTP-In-TCP -DisplayName 'WinRM HTTP-In' `
        -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 5985 | Out-Null
}}

Write-Output 'winrm-ready'
""".strip()


def render_enable_script(password: str) -> str:
    """Render the WinRM-enable PowerShell with the given Docker-user password.

    The password is embedded as a PowerShell single-quoted literal, so any
    single quote in it is escaped by doubling per PS rules.
    """
    escaped = password.replace("'", "''")
    return _PS_TEMPLATE.format(password=escaped)


def free_port(start: int = 17000, count: int = 200) -> int:
    for port in range(start, start + count):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port in {start}-{start + count}")


def wait_for_agent(endpoint: str, timeout: int = 600) -> None:
    """Poll /screenshot until the guest agent returns a non-empty 200."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{endpoint}/screenshot", timeout=5) as r:
                if r.status == 200 and len(r.read()) > 0:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"guest agent never ready at {endpoint} within {timeout}s")


def guest_execute(endpoint: str, command: str | list[str], shell: bool = False,
                  timeout: int = 120) -> dict:
    """POST to /setup/execute — same contract as SetupController._setup_execute."""
    payload = json.dumps({"command": command, "shell": shell}).encode()
    req = urllib.request.Request(
        f"{endpoint}/setup/execute",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode(errors="replace")
    if r.status != 200:
        raise RuntimeError(f"guest /setup/execute → HTTP {r.status}: {body[:400]}")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def backup_exists(base_image: Path) -> bool:
    if not DEFAULT_BACKUPS.is_dir():
        return False
    return any(DEFAULT_BACKUPS.glob(f"{base_image.stem}*.bak*"))


def start_qemu(base_image: Path, overlay: Path, local_port: int,
               tpm_sock: Path, pflash_vars: Path) -> subprocess.Popen:
    cmd = [
        "qemu-system-x86_64",
        "-machine", "q35,smm=on",
        "-cpu", "host",
        "-enable-kvm",
        "-m", "8G",
        "-smp", "8",
        "-drive", f"if=pflash,format=raw,readonly=on,file={OVMF_CODE}",
        "-drive", f"if=pflash,format=raw,file={pflash_vars}",
        "-chardev", f"socket,id=chrtpm,path={tpm_sock}",
        "-tpmdev", "emulator,id=tpm0,chardev=chrtpm",
        "-device", "tpm-tis,tpmdev=tpm0",
        "-drive", f"file={overlay},format=qcow2,if=virtio",
        "-netdev", f"user,id=net0,hostfwd=tcp:127.0.0.1:{local_port}-:5000",
        "-device", "virtio-net-pci,netdev=net0",
        "-vga", "virtio",
        "-display", "none",
    ]
    logger.info("starting qemu (local:%d → guest:5000, overlay=%s)", local_port, overlay.name)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_swtpm(sock_dir: Path) -> subprocess.Popen:
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock = sock_dir / "sock"
    proc = subprocess.Popen(
        [
            "swtpm", "socket",
            "--tpmstate", f"dir={sock_dir}",
            "--ctrl", f"type=unixio,path={sock}",
            "--tpm2",
            "--log", f"file={sock_dir}/swtpm.log,level=20",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        if sock.exists():
            return proc
        time.sleep(0.1)
    proc.kill()
    raise RuntimeError(f"swtpm socket never appeared at {sock}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[bootstrap_winrm] %(message)s")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=Path(os.environ.get("BASE_IMAGE", DEFAULT_BASE)))
    parser.add_argument("--skip-backup-check", action="store_true",
                        help="Don't require a backup under ~/.cube/images/backups/")
    parser.add_argument("--keep-overlay", action="store_true",
                        help="Don't commit/delete the overlay (for debugging).")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Seconds to wait for guest agent (default 900)")
    parser.add_argument("--admin-password", default=os.environ.get("WAA_BUILD_ADMIN_PASSWORD"),
                        help="Password to set on the Docker user (empty-password blocks "
                             "WinRM). Also accepts WAA_BUILD_ADMIN_PASSWORD env var. "
                             "This is the same value Packer later uses as "
                             "PKR_VAR_admin_password.")
    args = parser.parse_args()

    if not args.admin_password:
        logger.error(
            "--admin-password is required (or set WAA_BUILD_ADMIN_PASSWORD). "
            "The base image ships with a blank Docker-user password, which blocks WinRM."
        )
        return 2

    base: Path = args.base.expanduser().resolve()
    if not base.is_file():
        logger.error("base image not found: %s", base)
        return 2

    if not args.skip_backup_check and not backup_exists(base):
        logger.error(
            "no backup found under %s for %s — this script modifies the base image in place.\n"
            "  Run `cp %s ~/.cube/images/backups/%s.bak` first, or pass --skip-backup-check.",
            DEFAULT_BACKUPS, base.name, base, base.name,
        )
        return 2

    for prog in ("qemu-system-x86_64", "qemu-img", "swtpm"):
        if shutil.which(prog) is None:
            logger.error("%s not installed", prog)
            return 2
    if not OVMF_CODE.exists() or not OVMF_VARS.exists():
        logger.error("OVMF not at %s — install with `apt install ovmf`", OVMF_CODE.parent)
        return 2

    workdir = Path(f"/tmp/waa-bootstrap-winrm-{uuid.uuid4().hex[:8]}")
    workdir.mkdir()
    overlay = workdir / "overlay.qcow2"
    pflash_vars = workdir / "OVMF_VARS.fd"
    tpm_dir = workdir / "tpm"

    logger.info("workdir: %s", workdir)
    shutil.copy(OVMF_VARS, pflash_vars)
    subprocess.run(
        ["qemu-img", "create", "-f", "qcow2",
         "-b", str(base), "-F", "qcow2", str(overlay)],
        check=True, capture_output=True,
    )

    qemu: subprocess.Popen | None = None
    swtpm: subprocess.Popen | None = None
    try:
        swtpm = start_swtpm(tpm_dir)
        port = free_port()
        qemu = start_qemu(base, overlay, port, tpm_dir / "sock", pflash_vars)
        endpoint = f"http://127.0.0.1:{port}"

        logger.info("waiting for guest agent at %s (up to %ds)…", endpoint, args.timeout)
        wait_for_agent(endpoint, timeout=args.timeout)

        logger.info("guest agent ready — setting Docker password + enabling WinRM (idempotent)…")
        ps_script = render_enable_script(args.admin_password)
        result = guest_execute(
            endpoint,
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            shell=False,
            timeout=300,
        )
        output = (result.get("output") or "") + (result.get("error") or "")
        logger.info("guest powershell output: %s", output.strip()[:800])
        if "winrm-ready" not in output:
            raise RuntimeError("guest did not report 'winrm-ready' — see output above")

        logger.info("requesting clean shutdown…")
        try:
            guest_execute(endpoint, "shutdown /s /t 0 /f", shell=True, timeout=30)
        except Exception as exc:
            logger.warning("shutdown command returned error (may be benign): %s", exc)

        logger.info("waiting for qemu to exit…")
        for _ in range(120):  # 2 min
            if qemu.poll() is not None:
                break
            time.sleep(1)
        else:
            logger.warning("qemu did not exit cleanly — sending SIGTERM")
            qemu.terminate()
            qemu.wait(timeout=30)
    finally:
        if qemu and qemu.poll() is None:
            qemu.kill()
        if swtpm and swtpm.poll() is None:
            swtpm.terminate()
            try:
                swtpm.wait(timeout=10)
            except subprocess.TimeoutExpired:
                swtpm.kill()

    if args.keep_overlay:
        logger.info("--keep-overlay set; leaving overlay at %s", overlay)
        return 0

    logger.info("committing overlay → base (%s)…", base)
    subprocess.run(["qemu-img", "commit", str(overlay)], check=True)
    shutil.rmtree(workdir, ignore_errors=True)
    logger.info("done — %s now has WinRM enabled", base)
    return 0


if __name__ == "__main__":
    # Restore default SIGINT handling so Ctrl-C breaks out of wait_for_agent cleanly.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sys.exit(main())
