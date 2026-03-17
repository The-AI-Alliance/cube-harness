"""QEMU/KVM VM lifecycle manager for OSWorld.

Replaces desktop_env's Docker provider with a bare QEMU/KVM approach:
- Uses SLIRP user-mode networking with port forwarding (no root / bridge required)
- Snapshot strategy: read-only base qcow2 + per-instance copy-on-write overlay
  (reset = stop QEMU, delete overlay, create new overlay, start QEMU)
- Communicates with running QEMU via QMP Unix socket for clean shutdown

VM image download logic is ported from desktop_env's DockerVMManager.
"""

import json
import logging
import os
import socket
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from time import sleep
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# HuggingFace image URLs (same as desktop_env/providers/docker/manager.py)
UBUNTU_X86_URL = "https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip"
WINDOWS_X86_URL = "https://huggingface.co/datasets/xlangai/windows_osworld/resolve/main/Windows-10-x64.qcow2.zip"

_VM_READY_TIMEOUT = 300  # seconds to wait for VM screenshot endpoint
_VM_READY_POLL_INTERVAL = 2  # seconds between readiness polls
_QMP_TIMEOUT = 10  # seconds to wait for QMP socket


class QEMUConfig:
    """Configuration for a QEMU/KVM virtual machine.

    Parameters
    ----------
    base_image : Path
        Path to the read-only base qcow2 disk image.
    overlay_dir : Path
        Directory where per-instance overlay qcow2 files are created.
    memory : str
        RAM allocation passed to QEMU ``-m`` flag (e.g. ``"4G"``).
    cpus : int
        Number of vCPUs (``-smp``).
    headless : bool
        If True, suppress the graphical display (``-display none``).
    screen_width : int
        Horizontal resolution injected into the guest via kernel cmdline (informational).
    screen_height : int
        Vertical resolution (informational).
    enable_kvm : bool
        Automatically enable KVM hardware acceleration if ``/dev/kvm`` is present.
    """

    def __init__(
        self,
        base_image: Path,
        overlay_dir: Path,
        memory: str = "4G",
        cpus: int = 4,
        headless: bool = True,
        screen_width: int = 1920,
        screen_height: int = 1080,
        enable_kvm: bool = True,
    ) -> None:
        self.base_image = Path(base_image)
        self.overlay_dir = Path(overlay_dir)
        self.memory = memory
        self.cpus = cpus
        self.headless = headless
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.enable_kvm = enable_kvm


class QEMUManager:
    """Manages the full lifecycle of an OSWorld QEMU/KVM virtual machine.

    Usage::

        manager = QEMUManager(config)
        manager.start()          # boot VM (allocates ports, creates overlay)
        manager.reset()          # restore initial state (stop → new overlay → boot)
        manager.stop()           # shut down VM cleanly

    After ``start()`` the following properties are available:

    - :attr:`server_port`   — host port forwarded to guest :5000  (Flask agent)
    - :attr:`chromium_port` — host port forwarded to guest :9222  (Chromium DevTools)
    - :attr:`vnc_port`      — host port forwarded to guest :8006  (VNC/noVNC)
    - :attr:`vlc_port`      — host port forwarded to guest :8080  (VLC HTTP)

    Parameters
    ----------
    config : QEMUConfig
        VM configuration (image paths, memory, CPU, display settings).
    """

    def __init__(self, config: QEMUConfig) -> None:
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._pid_file: Optional[Path] = None
        self._overlay_path: Optional[Path] = None
        self._qmp_socket: Optional[Path] = None

        self._server_port: Optional[int] = None
        self._chromium_port: Optional[int] = None
        self._vnc_port: Optional[int] = None
        self._vlc_port: Optional[int] = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def server_port(self) -> int:
        if self._server_port is None:
            raise RuntimeError("VM not started — call start() first")
        return self._server_port

    @property
    def chromium_port(self) -> int:
        if self._chromium_port is None:
            raise RuntimeError("VM not started — call start() first")
        return self._chromium_port

    @property
    def vnc_port(self) -> int:
        if self._vnc_port is None:
            raise RuntimeError("VM not started — call start() first")
        return self._vnc_port

    @property
    def vlc_port(self) -> int:
        if self._vlc_port is None:
            raise RuntimeError("VM not started — call start() first")
        return self._vlc_port

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Allocate ports, create overlay, launch QEMU, and wait for VM readiness."""
        self.config.overlay_dir.mkdir(parents=True, exist_ok=True)

        self._server_port = _get_free_port(5000)
        self._chromium_port = _get_free_port(9222)
        self._vnc_port = _get_free_port(8006)
        self._vlc_port = _get_free_port(8080)

        self._overlay_path = self.config.overlay_dir / f"overlay_{self._server_port}.qcow2"
        self._qmp_socket = Path(tempfile.gettempdir()) / f"qemu_qmp_{self._server_port}.sock"
        self._pid_file = Path(tempfile.gettempdir()) / f"qemu_{self._server_port}.pid"

        self._create_overlay()
        self._launch_qemu()
        self._wait_for_ready()

    def reset(self) -> None:
        """Restore the VM to its initial state.

        Stops the running instance, deletes the overlay, creates a fresh one,
        and boots the VM again.
        """
        self._stop_qemu()
        if self._overlay_path and self._overlay_path.exists():
            self._overlay_path.unlink()
        self._create_overlay()
        self._launch_qemu()
        self._wait_for_ready()

    def stop(self) -> None:
        """Shut down the VM and clean up overlay and socket files."""
        self._stop_qemu()
        if self._overlay_path and self._overlay_path.exists():
            try:
                self._overlay_path.unlink()
            except OSError:
                pass
        if self._qmp_socket and self._qmp_socket.exists():
            try:
                self._qmp_socket.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_overlay(self) -> None:
        """Create a fresh qcow2 overlay on top of the read-only base image."""
        cmd = [
            "qemu-img",
            "create",
            "-f",
            "qcow2",
            "-b",
            str(self.config.base_image),
            "-F",
            "qcow2",
            str(self._overlay_path),
        ]
        logger.info("Creating overlay: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True)

    def _launch_qemu(self) -> None:
        """Build QEMU command and launch as a background subprocess."""
        qemu_cmd = ["qemu-system-x86_64"]

        # KVM hardware acceleration
        if self.config.enable_kvm and os.path.exists("/dev/kvm"):
            qemu_cmd += ["-enable-kvm"]
            logger.info("KVM acceleration enabled")
        else:
            logger.warning("KVM not available — running without hardware acceleration (slow)")

        # Machine resources
        qemu_cmd += ["-m", self.config.memory, "-smp", str(self.config.cpus)]

        # Disk
        qemu_cmd += ["-drive", f"file={self._overlay_path},format=qcow2,if=virtio"]

        # Network: SLIRP user-mode with port forwarding
        hostfwds = ",".join(
            [
                f"hostfwd=tcp::{self._server_port}-:5000",
                f"hostfwd=tcp::{self._chromium_port}-:9222",
                f"hostfwd=tcp::{self._vnc_port}-:8006",
                f"hostfwd=tcp::{self._vlc_port}-:8080",
            ]
        )
        qemu_cmd += [
            "-netdev",
            f"user,id=net0,{hostfwds}",
            "-device",
            "virtio-net-pci,netdev=net0",
        ]

        # Display
        if self.config.headless:
            qemu_cmd += ["-display", "none"]
        else:
            qemu_cmd += ["-vga", "virtio"]

        # QMP control socket
        qemu_cmd += ["-qmp", f"unix:{self._qmp_socket},server,nowait"]

        # PID file and daemonize
        qemu_cmd += ["-pidfile", str(self._pid_file), "-daemonize"]

        logger.info("Starting QEMU: %s", " ".join(qemu_cmd))
        result = subprocess.run(qemu_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"QEMU failed to start (exit {result.returncode}):\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        logger.info("QEMU launched (server_port=%d)", self._server_port)

    def _wait_for_ready(self) -> None:
        """Poll the guest's /screenshot endpoint until the VM is ready."""
        deadline = time.time() + _VM_READY_TIMEOUT
        url = f"http://localhost:{self._server_port}/screenshot"
        logger.info("Waiting for VM to be ready at %s ...", url)
        while time.time() < deadline:
            try:
                resp = requests.get(url, timeout=(5, 5))
                if resp.status_code == 200:
                    logger.info("VM is ready")
                    return
            except Exception:
                pass
            logger.info("VM not ready yet, retrying in %ds...", _VM_READY_POLL_INTERVAL)
            time.sleep(_VM_READY_POLL_INTERVAL)
        raise TimeoutError(f"VM failed to become ready within {_VM_READY_TIMEOUT}s")

    def _stop_qemu(self) -> None:
        """Send QMP 'quit' to the running QEMU instance and wait for it to exit."""
        if self._qmp_socket and self._qmp_socket.exists():
            try:
                _qmp_quit(str(self._qmp_socket))
                logger.info("Sent QMP quit")
            except Exception as exc:
                logger.warning("QMP quit failed (%s), falling back to SIGTERM", exc)
                self._kill_by_pid()
        elif self._pid_file and self._pid_file.exists():
            self._kill_by_pid()

        # Wait briefly for process to exit
        for _ in range(20):
            if self._pid_file and not self._pid_file.exists():
                break
            time.sleep(0.5)

    def _kill_by_pid(self) -> None:
        """Terminate the QEMU process via the PID file."""
        if not self._pid_file or not self._pid_file.exists():
            return
        try:
            pid = int(self._pid_file.read_text().strip())
            os.kill(pid, 15)  # SIGTERM
            logger.info("Sent SIGTERM to QEMU pid %d", pid)
        except Exception as exc:
            logger.warning("Failed to kill QEMU by pid: %s", exc)


# ------------------------------------------------------------------
# QMP communication
# ------------------------------------------------------------------


def _qmp_quit(socket_path: str) -> None:
    """Connect to a QEMU QMP Unix socket and send the 'quit' command."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(_QMP_TIMEOUT)
        sock.connect(socket_path)

        # Read the QMP greeting
        greeting = _qmp_recv(sock)
        logger.debug("QMP greeting: %s", greeting)

        # Negotiate capabilities
        sock.sendall(json.dumps({"execute": "qmp_capabilities"}).encode())
        _qmp_recv(sock)

        # Send quit
        sock.sendall(json.dumps({"execute": "quit"}).encode())
        try:
            _qmp_recv(sock)
        except (ConnectionResetError, OSError):
            pass  # Connection closed after quit — expected


def _qmp_recv(sock: socket.socket) -> dict:
    """Read a complete JSON object from a QMP socket."""
    data = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
        try:
            return json.loads(data.decode())
        except json.JSONDecodeError:
            continue
    return {}


# ------------------------------------------------------------------
# Port allocation
# ------------------------------------------------------------------


def _get_free_port(start: int = 5000) -> int:
    """Return the first free TCP port at or above ``start``."""
    for port in range(start, 65354):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found starting from {start}")


# ------------------------------------------------------------------
# VM image download (ported from desktop_env DockerVMManager)
# ------------------------------------------------------------------


def ensure_base_image(vm_dir: Path, os_type: str = "Ubuntu") -> Path:
    """Download and extract the OSWorld base qcow2 image if not already present.

    Parameters
    ----------
    vm_dir : Path
        Directory where the qcow2 image should be stored.
    os_type : str
        ``"Ubuntu"`` or ``"Windows"``.

    Returns
    -------
    Path
        Path to the extracted qcow2 file.
    """
    vm_dir = Path(vm_dir)
    vm_dir.mkdir(parents=True, exist_ok=True)

    if os_type == "Ubuntu":
        url = UBUNTU_X86_URL
    elif os_type == "Windows":
        url = WINDOWS_X86_URL
    else:
        raise ValueError(f"Unknown os_type: {os_type!r}")

    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if "hf-mirror.com" in hf_endpoint:
        url = url.replace("huggingface.co", "hf-mirror.com")
        logger.info("Using HF mirror: %s", url)

    zip_name = url.split("/")[-1]
    qcow2_name = zip_name[:-4] if zip_name.endswith(".zip") else zip_name
    qcow2_path = vm_dir / qcow2_name

    if qcow2_path.exists():
        logger.info("Base image already present: %s", qcow2_path)
        return qcow2_path

    zip_path = vm_dir / zip_name
    _download_file(url, zip_path)

    if zip_name.endswith(".zip"):
        logger.info("Extracting %s ...", zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(vm_dir)
        logger.info("Extracted to %s", vm_dir)

    return qcow2_path


def _download_file(url: str, dest: Path) -> None:
    """Download a file with resumable support and a progress bar."""
    downloaded_size = 0
    while True:
        headers: dict = {}
        if dest.exists():
            downloaded_size = dest.stat().st_size
            headers["Range"] = f"bytes={downloaded_size}-"

        with requests.get(url, headers=headers, stream=True) as resp:
            if resp.status_code == 416:
                logger.info("File already fully downloaded: %s", dest)
                return

            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            with (
                open(dest, "ab") as fp,
                tqdm(
                    desc=dest.name,
                    total=total,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                    initial=downloaded_size,
                    ascii=True,
                ) as bar,
            ):
                try:
                    for chunk in resp.iter_content(chunk_size=1024):
                        size = fp.write(chunk)
                        bar.update(size)
                    return  # success
                except (requests.RequestException, IOError) as exc:
                    logger.error("Download interrupted: %s — retrying", exc)
                    sleep(5)
