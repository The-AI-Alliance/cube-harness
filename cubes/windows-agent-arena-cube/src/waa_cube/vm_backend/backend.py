"""WAADockerVMBackend — VMBackend that runs windowsarena/winarena:latest Docker image.

The Docker image contains a QEMU-hosted Windows 11 VM with named snapshots
pre-baked (e.g. "vscode", "libreoffice_calc"). The snapshot restoration uses
QMP loadvm (not container restart) which is fast (~5-10s) and matches WAA's
original behavior.

Reset strategy: QMP loadvm <name> on the running container (SNAPSHOT isolation).
Ports:
    5000 → Windows guest Flask agent (via QEMU hostfwd inside container)
    7200 → QEMU QMP TCP socket (inside container)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import docker
import docker.errors
import requests
from cube.vm import VM, VMBackend, VMConfig
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

WAA_DOCKER_IMAGE = "windowsarena/winarena:latest"

_VM_READY_TIMEOUT = 900  # seconds to wait on normal boot
_VM_READY_TIMEOUT_FIRST = 1800  # seconds to wait on first boot (Windows doing one-time setup)
_VM_READY_POLL_INTERVAL = 5  # seconds between readiness polls
_QMP_TIMEOUT = 30  # seconds to wait for QMP response after loadvm


# ---------------------------------------------------------------------------
# Port reservation helpers (adapted from cube_vm_backend.qemu_manager)
# ---------------------------------------------------------------------------


def _reserve_free_port() -> tuple[int, str]:
    """Bind an OS-assigned TCP port and return (port, lock_path).

    The port is held by a file lock until released. Callers must call
    _release_port_reservation(lock_path) when done.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    lock_path = os.path.join(tempfile.gettempdir(), f"waa_port_{port}.lock")
    with open(lock_path, "w") as f:
        f.write(str(port))
    return port, lock_path


def _release_port_reservation(lock_path: str) -> None:
    """Remove port lock file."""
    try:
        os.remove(lock_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Per-container overlay storage helpers
# ---------------------------------------------------------------------------

_METADATA_SUFFIXES = {".base", ".boot", ".mac", ".mode", ".ver", ".rom", ".tpm", ".vars"}


def _create_overlay_storage(base_storage: str, container_name: str) -> str:
    """Create a per-container storage directory backed by the golden disk image.

    Each parallel container gets its own qcow2 overlay of the base data.img so
    they don't share a writable disk. Small metadata files are copied as-is.

    Returns the path to the overlay directory.
    """
    base = Path(base_storage)
    overlay_dir = base / "containers" / container_name
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Create a sparse raw copy of the base disk image.
    # The WAA container hardcodes format=raw in its QEMU invocation so we cannot
    # use a qcow2 backing-file overlay — it would be read as raw bytes and corrupt
    # the VM. A sparse copy uses only the allocated blocks of the source on disk.
    for name in ("data.qcow2", "data.img"):
        base_disk = base / name
        if base_disk.exists():
            overlay_disk = overlay_dir / name
            logger.info("Creating sparse disk copy %s (this may take ~30s)…", overlay_disk)
            subprocess.run(
                ["cp", "--sparse=always", str(base_disk), str(overlay_disk)],
                check=True,
            )
            logger.debug("Sparse copy complete: %s", overlay_disk)
            break

    # Copy small metadata files (skip any we lack permission to read)
    for f in base.iterdir():
        if f.is_file() and f.suffix in _METADATA_SUFFIXES:
            try:
                shutil.copy2(str(f), str(overlay_dir / f.name))
            except PermissionError:
                logger.debug("Skipping unreadable metadata file: %s", f.name)

    return str(overlay_dir)


def _remove_overlay_storage(overlay_dir: str) -> None:
    """Delete a per-container overlay storage directory."""
    try:
        shutil.rmtree(overlay_dir)
        logger.debug("Removed overlay storage: %s", overlay_dir)
    except Exception as exc:
        logger.warning("Failed to remove overlay storage %s: %s", overlay_dir, exc)


# ---------------------------------------------------------------------------
# QMP client (adapted from WindowsAgentArena/src/win-arena-container/client/
#              desktop_env/controllers/vm.py)
# ---------------------------------------------------------------------------


class QMPConnection:
    """Context manager for a QMP TCP socket connection."""

    def __init__(self, host: str, port: int, timeout: float = _QMP_TIMEOUT) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    def __enter__(self) -> "QMPConnection":
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        greeting = self._read_response()
        if not greeting or "QMP" not in greeting:
            raise ConnectionError("Invalid QMP greeting")
        self._send_command("qmp_capabilities")
        return self

    def __exit__(self, *args: object) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _send_command(self, command: str, arguments: Optional[dict] = None) -> dict:
        if self._sock is None:
            raise ConnectionError("Not connected to QMP")
        cmd: dict = {"execute": command}
        if arguments:
            cmd["arguments"] = arguments
        self._sock.send(json.dumps(cmd).encode("utf-8") + b"\n")
        return self._read_response()

    def _read_response(self) -> dict:
        if self._sock is None:
            raise ConnectionError("Not connected to QMP")
        chunks: list[bytes] = []
        while True:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("QMP connection closed")
            chunks.append(chunk)
            data = b"".join(chunks)
            try:
                decoded = data.decode("latin-1")
                if decoded.strip().endswith("}"):
                    break
            except UnicodeError:
                continue
        messages = [json.loads(m) for m in decoded.split("\n") if m.strip()]
        for msg in reversed(messages):
            if "event" not in msg:
                return msg
        return messages[-1]

    def loadvm(self, name: str) -> None:
        """Restore VM to a named snapshot via QMP loadvm."""
        response = self._send_command("loadvm", {"name": name})
        if "error" in response:
            raise RuntimeError(f"QMP loadvm '{name}' failed: {response['error']}")

    def query_snapshots(self) -> list[dict]:
        """Return list of available snapshot dicts."""
        response = self._send_command("query-snapshots")
        return response.get("return", [])


# ---------------------------------------------------------------------------
# WAADockerManager
# ---------------------------------------------------------------------------


class WAADockerManager:
    """Manages the lifecycle of a windowsarena/winarena Docker container.

    Unlike OSWorld's DockerManager, we do NOT restart the container on
    restore_snapshot — instead we call QMP loadvm. The container stays running
    across tasks.
    """

    def __init__(
        self,
        image: str,
        storage_path: str,
        ram_size: str,
        cpu_cores: int,
        screen_width: int,
        screen_height: int,
        pull_policy: str,
    ) -> None:
        self.image = image
        self.storage_path = storage_path
        self.ram_size = ram_size
        self.cpu_cores = cpu_cores
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.pull_policy = pull_policy

        self._client = docker.from_env()
        self._container: Optional[docker.models.containers.Container] = None
        self._container_name: str = f"waa-cube-{uuid.uuid4().hex[:8]}"
        self._overlay_storage_path: str = ""

        # Reserved host ports
        self._server_port: int = 0
        self._server_port_lock: str = ""
        self._qmp_port: int = 0
        self._qmp_port_lock: str = ""
        self._chromium_port: int = 0
        self._chromium_port_lock: str = ""
        self._vlc_port: int = 0
        self._vlc_port_lock: str = ""

    @property
    def server_port(self) -> int:
        return self._server_port

    @property
    def qmp_port(self) -> int:
        return self._qmp_port

    @property
    def chromium_port(self) -> int:
        return self._chromium_port

    @property
    def vlc_port(self) -> int:
        return self._vlc_port

    def start(self) -> None:
        """Pull image if needed, reserve ports, start container, wait for VM."""
        self._pull_image_if_needed()
        self._stop_stale_containers()
        self._overlay_storage_path = _create_overlay_storage(self.storage_path, self._container_name)
        self._reserve_ports()
        self._start_container()
        self._wait_for_ready()
        self._start_proxies()
        self._verify_host_connectivity()

    def _container_status(self) -> str:
        """Return the current Docker status for the managed container."""
        if self._container is None:
            return "missing"
        try:
            self._container.reload()
            return self._container.status
        except docker.errors.NotFound:
            return "removed"

    def is_alive(self) -> bool:
        """Return True if the underlying Docker container is still running."""
        return self._container_status() == "running"

    def _require_running_container(self, action: str) -> None:
        """Raise a clear error if the managed container is not available for exec."""
        status = self._container_status()
        if status != "running":
            raise RuntimeError(
                f"Cannot {action}: WAA container '{self._container_name}' is not running "
                f"(status={status}). Relaunch the VM before continuing."
            )

    def _stop_stale_containers(self) -> None:
        """Stop stale WAA containers that hold the base storage path directly.

        Only targets containers whose mount source is exactly the base storage path
        (the legacy single-container case). Per-container overlay directories under
        {base_storage}/containers/ are used by legitimate parallel workers and must
        not be stopped.
        """
        try:
            for container in self._client.containers.list():
                for mount in container.attrs.get("Mounts", []):
                    source = mount.get("Source", "")
                    is_same_storage = source == self.storage_path
                    if is_same_storage and container.name != self._container_name:
                        logger.warning(
                            "Stopping stale WAA container '%s' that holds storage %s",
                            container.name,
                            source,
                        )
                        try:
                            container.stop(timeout=10)
                        except Exception as exc:
                            logger.warning("Error stopping stale container '%s': %s", container.name, exc)
        except Exception as exc:
            logger.warning("Could not check for stale containers: %s", exc)

    def _start_proxies(self) -> None:
        """No-op: the dockurr/windows base image handles port forwarding natively.

        The base image's networking scripts set up iptables DNAT rules from the
        container's interface to the Windows VM at 20.20.20.21.  Docker port
        mapping (host → container) combined with the base image's DNAT
        (container → VM) provides end-to-end connectivity without any extra
        forwarders.
        """

    def _verify_host_connectivity(self) -> None:
        """Verify the host can reach the VM Flask server through the port-forwarding chain.

        Raises RuntimeError if the probe fails after retries.
        """
        url = f"http://localhost:{self._server_port}/probe"
        for attempt in range(12):
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    logger.info("Host→VM connectivity verified via %s", url)
                    return
                logger.debug("Probe returned %d (attempt %d/12)", resp.status_code, attempt + 1)
            except Exception:
                logger.debug("Probe failed (attempt %d/12), retrying in 5s…", attempt + 1)
            time.sleep(5)
        raise RuntimeError(
            f"Cannot reach WAA VM from host via {url} after 60s.\n"
            "Port forwarding from host → Docker container → VM is not working.\n"
            "Check that the container is running and the VM Flask server is up."
        )

    def _pull_image_if_needed(self) -> None:
        if self.pull_policy == "always":
            logger.info("Pulling Docker image: %s", self.image)
            self._client.images.pull(self.image)
            return
        try:
            self._client.images.get(self.image)
            logger.debug("Docker image already present: %s", self.image)
        except docker.errors.ImageNotFound:
            if self.pull_policy == "never":
                raise RuntimeError(
                    f"Docker image '{self.image}' not found and pull_policy='never'.\n"
                    "Build the image with:\n"
                    "  cd WindowsAgentArena/scripts && ./build-container-image.sh\n"
                    "  ./run-local.sh --prepare-image true"
                )
            logger.info("Pulling Docker image: %s", self.image)
            self._client.images.pull(self.image)

    def _reserve_ports(self) -> None:
        self._server_port, self._server_port_lock = _reserve_free_port()
        self._qmp_port, self._qmp_port_lock = _reserve_free_port()
        self._chromium_port, self._chromium_port_lock = _reserve_free_port()
        self._vlc_port, self._vlc_port_lock = _reserve_free_port()
        logger.debug(
            "Reserved ports — server=%d qmp=%d chromium=%d vlc=%d",
            self._server_port,
            self._qmp_port,
            self._chromium_port,
            self._vlc_port,
        )

    def _start_container(self) -> None:
        logger.info(
            "Starting WAA Docker container '%s' (overlay: %s)", self._container_name, self._overlay_storage_path
        )
        self._container = self._client.containers.run(
            self.image,
            # Pass as a single-element list so Docker does not split it.
            # The image ENTRYPOINT is ["/bin/bash", "-c"], so Docker runs:
            #   bash -c "/entry.sh --prepare-image false --start-client false"
            # If we pass a plain string, the SDK splits it into a list and
            # the args become bash positional params instead of entry.sh args.
            command=["/entry.sh --prepare-image false --start-client false"],
            name=self._container_name,
            detach=True,
            cap_add=["NET_ADMIN"],
            devices=["/dev/kvm:/dev/kvm"] if Path("/dev/kvm").exists() else [],
            volumes={
                self._overlay_storage_path: {"bind": "/storage", "mode": "rw"},
            },
            ports={
                "5000/tcp": self._server_port,
                "7200/tcp": self._qmp_port,
                "9222/tcp": self._chromium_port,
                "8080/tcp": self._vlc_port,
            },
            environment={
                "RAM_SIZE": self.ram_size,
                "CPU_CORES": str(self.cpu_cores),
                "XRES": str(self.screen_width),
                "YRES": str(self.screen_height),
                "ARGUMENTS": "-qmp tcp:0.0.0.0:7200,server,nowait",
            },
        )

    def _wait_for_ready(self) -> None:
        """Poll /probe endpoint until the Windows guest Flask server responds.

        The Flask server runs on the QEMU guest at 20.20.20.21:5000, which is
        only reachable from inside the container (TAP network). We probe via
        docker exec rather than from the host.

        Uses a longer timeout on first boot (windows.boot marker absent) since
        Windows performs one-time hardware setup tasks after installation.
        """
        boot_marker = Path(self._overlay_storage_path) / "windows.boot"
        first_boot = not boot_marker.exists()
        timeout = _VM_READY_TIMEOUT_FIRST if first_boot else _VM_READY_TIMEOUT
        if first_boot:
            logger.info("First boot detected (no windows.boot marker) — using extended timeout of %ds", timeout)
        deadline = time.monotonic() + timeout
        logger.info("Waiting for WAA VM to be ready (probing 20.20.20.21:5000/probe inside container)…")
        while time.monotonic() < deadline:
            if self._container is not None:
                status = self._container_status()
                if status == "removed":
                    raise RuntimeError("WAA container was removed before the VM became ready.")
                if status not in ("running", "created"):
                    try:
                        logs = self._container.logs(tail=50).decode("utf-8", errors="replace")
                    except Exception:
                        logs = "(logs unavailable — container already removed)"
                    raise RuntimeError(
                        f"WAA container exited with status '{status}' before VM became ready.\n"
                        f"Last 50 lines of container logs:\n{logs}"
                    )
                try:
                    result = self._container.exec_run(
                        "curl -sf --max-time 3 http://20.20.20.21:5000/probe",
                        demux=False,
                    )
                    if result.exit_code == 0:
                        logger.info("WAA VM is ready")
                        return
                except Exception:
                    pass
            time.sleep(_VM_READY_POLL_INTERVAL)
        raise TimeoutError(f"WAA VM did not become ready within {_VM_READY_TIMEOUT}s")

    def restore_snapshot(self, name: str) -> None:
        """Reset VM state by closing all open applications via the guest HTTP API.

        WAA does not use QMP loadvm for task resets — snapshots are not pre-baked
        into the disk image. Instead, the original WAA client calls /setup/close_all
        on the Windows guest Flask server to close all applications between tasks.
        """
        logger.info("Resetting WAA VM state (close_all) for snapshot '%s'", name)
        self._require_running_container("reset the VM state")
        try:
            self._container.exec_run(
                "curl -sf -X POST --max-time 30 http://20.20.20.21:5000/setup/close_all",
                demux=False,
            )
        except Exception as exc:
            logger.warning("close_all request failed (non-fatal): %s", exc)
        time.sleep(3)

    def stop(self) -> None:
        """Stop and remove the Docker container, release port reservations, clean up overlay."""
        if self._container is not None:
            try:
                self._container.stop(timeout=10)
            except docker.errors.NotFound:
                pass
            except docker.errors.APIError as exc:
                logger.warning("Error stopping WAA container: %s", exc)
            try:
                self._container.remove(force=True)
            except docker.errors.NotFound:
                pass
            except docker.errors.APIError as exc:
                logger.warning("Error removing WAA container: %s", exc)
            self._container = None

        for lock_path in (
            self._server_port_lock,
            self._qmp_port_lock,
            self._chromium_port_lock,
            self._vlc_port_lock,
        ):
            _release_port_reservation(lock_path)

        if self._overlay_storage_path:
            _remove_overlay_storage(self._overlay_storage_path)
            self._overlay_storage_path = ""


# ---------------------------------------------------------------------------
# WAADockerVM
# ---------------------------------------------------------------------------


class WAADockerVM(VM):
    """Runtime handle to a WAA QEMU-in-Docker VM.

    Task resets call /setup/close_all on the Windows guest Flask server
    (matching WAA's original behavior). The container stays running across tasks.
    """

    def __init__(self, manager: WAADockerManager) -> None:
        self._manager = manager

    @property
    def endpoint(self) -> str:
        """Base URL of the Windows guest Flask agent: http://localhost:<port>."""
        return f"http://localhost:{self._manager.server_port}"

    @property
    def server_port(self) -> int:
        """Host port mapped to the Windows guest Flask server (5000 inside)."""
        return self._manager.server_port

    @property
    def qmp_port(self) -> int:
        """Host port mapped to QEMU QMP (7200 inside container)."""
        return self._manager.qmp_port

    @property
    def chromium_port(self) -> int:
        """Host port mapped to Chrome DevTools Protocol (9222 inside guest)."""
        return self._manager.chromium_port

    @property
    def vlc_port(self) -> int:
        """Host port mapped to VLC HTTP interface (8080 inside guest)."""
        return self._manager.vlc_port

    def restore_snapshot(self, name: str) -> None:
        """Restore the VM to a named QEMU snapshot via QMP loadvm."""
        self._manager.restore_snapshot(name)

    def refresh_proxies(self) -> None:
        """No-op: the base image's iptables DNAT rules persist across snapshot restores.

        Snapshot restores only call /setup/close_all on the Windows guest, which
        does not affect the container's network namespace or iptables rules.
        """

    def is_alive(self) -> bool:
        """Return True if the underlying Docker container is still running."""
        return self._manager.is_alive()

    def stop(self) -> None:
        """Stop the Docker container and release port reservations."""
        self._manager.stop()


# ---------------------------------------------------------------------------
# WAADockerVMBackend
# ---------------------------------------------------------------------------


WAA_VM_STORAGE_ENV = "WAA_VM_STORAGE"
_DEFAULT_STORAGE_PATH = os.path.expanduser("~/.cube/waa/storage")


class WAADockerVMBackend(VMBackend):
    """VMBackend that runs windowsarena/winarena:latest Docker image.

    The Docker image contains QEMU + a Windows 11 VM with named snapshots
    pre-baked (e.g. 'vscode', 'libreoffice_calc').

    Unlike LocalDockerVMBackend, restore_snapshot() uses QMP loadvm rather
    than container restart, providing fast SNAPSHOT isolation (~5-10s).

    Prerequisites:
        1. Build the Docker image and prepare the golden storage directory:
               cd WindowsAgentArena/scripts
               ./build-container-image.sh
               ./run-local.sh --prepare-image true
           This installs Windows 11 and creates all task snapshots (~20 min).
           The resulting storage directory contains data.qcow2.
        2. Set WAA_VM_STORAGE=/path/to/vm/storage, or pass storage_path=.
        3. /dev/kvm must be present for KVM acceleration (recommended).
    """

    waa_image: str = WAA_DOCKER_IMAGE
    """Docker image to run. Defaults to windowsarena/winarena:latest."""

    storage_path: str | None = None
    """Host directory mounted as /storage inside the container.
    Must contain data.qcow2 (the prepared Windows disk image with snapshots).
    Falls back to WAA_VM_STORAGE env var, then ~/.cube/waa/storage."""

    ram_size: str = "8G"
    """RAM allocation for the Windows VM (e.g. '8G')."""

    cpu_cores: int = 8
    """Number of vCPUs for the Windows VM."""

    screen_width: int = 1920
    """QEMU display width passed as XRES. Must be 1920 — the Windows accessibility
    API returns an empty tree when QEMU starts at the snapshot resolution (1280×800).
    The mismatch forces a display reinit on snapshot restore that wakes the UI
    Automation framework. The actual VM resolution after restore is 1280×800."""

    screen_height: int = 1080
    """QEMU display height passed as YRES. See screen_width note."""

    pull_policy: str = "missing"
    """Docker image pull policy: 'missing' (default), 'always', 'never'."""

    setup_iso_path: str | None = None
    """Optional path to a Windows setup ISO to mount as /custom.iso inside the container.
    Only used during install(). Falls back to the WAA_SETUP_ISO env var."""

    def _resolve_storage_path(self) -> str:
        """Return the resolved storage path, checking env var and default."""
        if self.storage_path:
            return self.storage_path
        env_val = os.environ.get(WAA_VM_STORAGE_ENV)
        if env_val:
            return env_val
        return _DEFAULT_STORAGE_PATH

    def install(self) -> None:
        """Prepare the Windows VM disk image if not already built.

        Runs the WAA Docker container with --prepare-image true, which installs
        Windows from the provided ISO and saves the resulting disk image to the
        storage directory. This is a one-time operation (~20 min).

        Requires a Windows 11 Enterprise Evaluation ISO — the container does not
        support automatic download. Set WAA_SETUP_ISO or pass setup_iso_path=.

        Idempotent: skips if data.qcow2 already exists in the storage directory.
        """
        storage = self._resolve_storage_path()
        storage_dir = Path(storage)
        disk_image = next(
            (storage_dir / name for name in ("data.qcow2", "data.img") if (storage_dir / name).exists()),
            None,
        )

        if disk_image is not None:
            logger.info("WAA disk image already present at %s — skipping install", disk_image)
            return

        logger.info("Creating WAA storage directory: %s", storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)

        client = docker.from_env()
        try:
            client.images.get(self.waa_image)
        except docker.errors.ImageNotFound:
            logger.info("Pulling Docker image: %s", self.waa_image)
            client.images.pull(self.waa_image)

        iso_path = self.setup_iso_path or os.environ.get("WAA_SETUP_ISO")
        if not iso_path or not Path(iso_path).exists():
            raise FileNotFoundError(
                "A Windows 11 Enterprise Evaluation ISO is required to prepare the WAA disk image.\n\n"
                "The WAA container does not support automatic ISO download.\n"
                "Download the ISO manually from:\n"
                "  https://www.microsoft.com/en-us/evalcenter/evaluate-windows-11-enterprise\n\n"
                "Then either:\n"
                "  - Set WAA_SETUP_ISO=/path/to/Win11_Eval.iso\n"
                "  - Pass setup_iso_path='/path/to/Win11_Eval.iso' to WAADockerVMBackend\n"
                "  - Place it at WindowsAgentArena/src/win-arena-container/vm/image/setup.iso"
            )

        volumes: dict[str, dict[str, str]] = {storage: {"bind": "/storage", "mode": "rw"}}
        volumes[str(Path(iso_path).resolve())] = {"bind": "/custom.iso", "mode": "ro"}
        logger.info("Mounting setup ISO: %s", iso_path)

        devices = ["/dev/kvm:/dev/kvm"] if Path("/dev/kvm").exists() else []

        logger.info(
            "Starting WAA image preparation (this takes ~20 min)...\n  Image: %s\n  Storage: %s",
            self.waa_image,
            storage,
        )
        container = client.containers.run(
            self.waa_image,
            command=["-c", "/entry.sh --prepare-image true --start-client false"],
            entrypoint=["/bin/bash"],
            detach=True,
            cap_add=["NET_ADMIN"],
            devices=devices,
            volumes=volumes,
            environment={
                "RAM_SIZE": self.ram_size,
                "CPU_CORES": str(self.cpu_cores),
            },
        )
        exit_code = -1
        try:
            for log_bytes in container.logs(stream=True, follow=True):
                logger.info("[waa-install] %s", log_bytes.decode("utf-8", errors="replace").rstrip())
            result = container.wait()
            exit_code = result.get("StatusCode", -1)
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass

        if exit_code != 0:
            raise RuntimeError(
                f"WAA image preparation failed with exit code {exit_code}.\nCheck the logs above for details."
            )
        final_image = next(
            (storage_dir / name for name in ("data.qcow2", "data.img") if (storage_dir / name).exists()),
            None,
        )
        if final_image is None:
            raise RuntimeError(
                f"WAA image preparation completed but no disk image found in {storage_dir}.\n"
                "Check the logs above for details."
            )
        logger.info("WAA image preparation complete — disk image: %s", final_image)

    def ensure_resource(self, config: VMConfig) -> None:
        """Validate Docker image and storage directory are present."""
        storage = self._resolve_storage_path()
        if not Path(storage).is_dir():
            raise FileNotFoundError(
                f"WAA storage directory not found: {storage}\n\n"
                "Prepare the Windows VM image with:\n"
                "  cd WindowsAgentArena/scripts\n"
                "  ./build-container-image.sh\n"
                "  ./run-local.sh --prepare-image true\n\n"
                "Then set WAA_VM_STORAGE=/path/to/vm/storage (or pass storage_path= "
                "to WAADockerVMBackend).\n"
                f"The default location is {_DEFAULT_STORAGE_PATH}."
            )

        client = docker.from_env()
        try:
            client.images.get(self.waa_image)
        except docker.errors.ImageNotFound:
            if self.pull_policy == "never":
                raise RuntimeError(
                    f"Docker image '{self.waa_image}' not found.\n\n"
                    "Build the WAA golden image with:\n"
                    "  cd WindowsAgentArena/scripts\n"
                    "  ./build-container-image.sh\n"
                    "  ./run-local.sh --prepare-image true\n\n"
                    "This will install Windows 11 and create all task snapshots (~20 min)."
                )
            # Will be pulled in WAADockerManager._pull_image_if_needed()

    def cleanup_stale_overlays(self) -> None:
        """Remove overlay directories left by crashed runs.

        Call once before starting any workers. Safe because no workers are running yet,
        so there are no races with ongoing overlay creation.
        """
        storage = self._resolve_storage_path()
        containers_dir = Path(storage) / "containers"
        if not containers_dir.is_dir():
            return
        client = docker.from_env()
        running_names = {c.name for c in client.containers.list()}
        for overlay in containers_dir.iterdir():
            if overlay.is_dir() and overlay.name not in running_names:
                logger.info("Removing stale overlay for defunct container '%s'", overlay.name)
                _remove_overlay_storage(str(overlay))

    def launch(self, config: VMConfig) -> WAADockerVM:
        """Start the Docker container and return a live handle.

        Blocks until the Windows guest Flask server is reachable.
        """
        self.ensure_resource(config)
        storage = self._resolve_storage_path()

        manager = WAADockerManager(
            image=self.waa_image,
            storage_path=storage,
            ram_size=self.ram_size,
            cpu_cores=self.cpu_cores,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            pull_policy=self.pull_policy,
        )
        manager.start()
        logger.info(
            "WAA Docker VM launched — server=%d qmp=%d chromium=%d vlc=%d",
            manager.server_port,
            manager.qmp_port,
            manager.chromium_port,
            manager.vlc_port,
        )
        return WAADockerVM(manager)

    def close(self) -> None:
        """No-op — each VM is stopped individually via vm.stop()."""
