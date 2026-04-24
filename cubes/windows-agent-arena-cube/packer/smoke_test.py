#!/usr/bin/env python3
"""Boot the Packer-prepared image and verify both endpoints come up:

    * guest agent on :5000  (proves the scheduled task + Flask server still works)
    * OpenSSH on :22        (proves our side-loaded install survived sysprep)

Uses a COW overlay over the prepared qcow2 so the prepared image is not
modified. No admin password needed — we only read endpoints.

Usage::

    uv run python packer/smoke_test.py  # defaults to packer/output-waa-prepared/...
"""

from __future__ import annotations

import argparse
import logging
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

logger = logging.getLogger("smoke_test")

DEFAULT_IMG = Path(__file__).parent / "output-waa-prepared" / "waa-windows-prepared.qcow2"
OVMF_CODE = Path("/usr/share/OVMF/OVMF_CODE_4M.ms.fd")
OVMF_VARS = Path("/usr/share/OVMF/OVMF_VARS_4M.ms.fd")


def free_port(start: int = 18000) -> int:
    for port in range(start, start + 200):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port")


def tcp_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def http_ok(url: str, timeout: float = 5.0) -> tuple[bool, int]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200, len(r.read())
    except Exception:
        return False, 0


def start_swtpm(sock_dir: Path) -> subprocess.Popen:
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock = sock_dir / "sock"
    p = subprocess.Popen(
        ["swtpm", "socket",
         "--tpmstate", f"dir={sock_dir}",
         "--ctrl", f"type=unixio,path={sock}",
         "--tpm2",
         "--log", f"file={sock_dir}/swtpm.log,level=20"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        if sock.exists():
            return p
        time.sleep(0.1)
    p.kill()
    raise RuntimeError("swtpm socket never appeared")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[smoke] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, default=DEFAULT_IMG)
    ap.add_argument("--ssh-key", type=Path, default=Path.home() / ".ssh" / "id_ed25519")
    ap.add_argument("--agent-timeout", type=int, default=900)
    ap.add_argument("--ssh-timeout", type=int, default=300)
    args = ap.parse_args()

    img: Path = args.image.expanduser().resolve()
    if not img.is_file():
        logger.error("image not found: %s", img)
        return 2

    for prog in ("qemu-system-x86_64", "qemu-img", "swtpm"):
        if shutil.which(prog) is None:
            logger.error("%s not installed", prog)
            return 2

    workdir = Path(f"/tmp/waa-smoke-{uuid.uuid4().hex[:8]}")
    workdir.mkdir()
    overlay = workdir / "overlay.qcow2"
    pflash = workdir / "OVMF_VARS.fd"
    tpm_dir = workdir / "tpm"

    logger.info("workdir=%s", workdir)
    shutil.copy(OVMF_VARS, pflash)
    subprocess.run(["qemu-img", "create", "-f", "qcow2",
                    "-b", str(img), "-F", "qcow2", str(overlay)],
                   check=True, capture_output=True)

    port_agent = free_port(18000)
    port_ssh = free_port(18200)
    swtpm: subprocess.Popen | None = None
    qemu: subprocess.Popen | None = None
    try:
        swtpm = start_swtpm(tpm_dir)

        cmd = [
            "qemu-system-x86_64",
            "-machine", "q35,smm=on",
            "-cpu", "host",
            "-enable-kvm",
            "-m", "8G",
            "-smp", "8",
            "-drive", f"if=pflash,format=raw,readonly=on,file={OVMF_CODE}",
            "-drive", f"if=pflash,format=raw,file={pflash}",
            "-chardev", f"socket,id=chrtpm,path={tpm_dir}/sock",
            "-tpmdev", "emulator,id=tpm0,chardev=chrtpm",
            "-device", "tpm-tis,tpmdev=tpm0",
            "-drive", f"file={overlay},format=qcow2,if=virtio",
            "-netdev", f"user,id=n0,hostfwd=tcp:127.0.0.1:{port_agent}-:5000,"
                        f"hostfwd=tcp:127.0.0.1:{port_ssh}-:22",
            "-device", "virtio-net-pci,netdev=n0",
            "-vga", "virtio",
            "-display", "none",
        ]
        logger.info("starting qemu: guest:5000 → local:%d, guest:22 → local:%d",
                    port_agent, port_ssh)
        qemu = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Wait for the guest agent (longer timeout — first boot post-sysprep
        # mini-setup can take up to ~10 min).
        agent_url = f"http://127.0.0.1:{port_agent}/screenshot"
        logger.info("waiting for guest agent at %s (up to %ds)", agent_url, args.agent_timeout)
        agent_deadline = time.time() + args.agent_timeout
        agent_up = False
        while time.time() < agent_deadline:
            ok, nbytes = http_ok(agent_url, timeout=3)
            if ok and nbytes > 0:
                logger.info("GUEST AGENT: UP (%d bytes)", nbytes)
                agent_up = True
                break
            time.sleep(5)
        if not agent_up:
            logger.error("GUEST AGENT: TIMEOUT")

        # Now SSH — should be up once Windows is fully booted.
        logger.info("probing SSH on 127.0.0.1:%d (up to %ds)", port_ssh, args.ssh_timeout)
        ssh_deadline = time.time() + args.ssh_timeout
        ssh_up = False
        while time.time() < ssh_deadline:
            if tcp_reachable("127.0.0.1", port_ssh, timeout=3):
                # Try actually SSH'ing.
                r = subprocess.run(
                    ["ssh", "-i", str(args.ssh_key),
                     "-o", "IdentitiesOnly=yes",
                     "-o", "StrictHostKeyChecking=no",
                     "-o", "UserKnownHostsFile=/dev/null",
                     "-o", "ConnectTimeout=5",
                     "-o", "BatchMode=yes",
                     "-p", str(port_ssh),
                     "Docker@127.0.0.1", "echo SSH_OK"],
                    capture_output=True, text=True, timeout=20)
                if "SSH_OK" in r.stdout:
                    logger.info("SSH: UP + authenticated as Docker")
                    ssh_up = True
                    break
                else:
                    logger.info("SSH port open but auth not ready yet — retrying...")
            time.sleep(5)
        if not ssh_up:
            logger.error("SSH: TIMEOUT")

        return 0 if (agent_up and ssh_up) else 1
    finally:
        if qemu and qemu.poll() is None:
            qemu.terminate()
            try: qemu.wait(timeout=15)
            except subprocess.TimeoutExpired: qemu.kill()
        if swtpm and swtpm.poll() is None:
            swtpm.terminate()
            try: swtpm.wait(timeout=5)
            except subprocess.TimeoutExpired: swtpm.kill()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
