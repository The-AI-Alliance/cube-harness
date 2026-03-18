"""VM backend for OSWorld.

Provides OSWorldQEMUVMBackend — a LocalQEMUVMBackend subclass that
auto-downloads the OSWorld qcow2 image from HuggingFace on first use.
"""

import logging
import os
import zipfile
from pathlib import Path
from time import sleep

import requests
from cube.vm import VM, VMBackend, VMConfig
from cube_vm_backend import LocalQEMUVM, LocalQEMUVMBackend
from cube_vm_backend.qemu_manager import QEMUConfig, QEMUManager
from tqdm import tqdm

logger = logging.getLogger(__name__)

# HuggingFace image URLs for OSWorld VM images
_UBUNTU_X86_URL = "https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip"
_WINDOWS_X86_URL = "https://huggingface.co/datasets/xlangai/windows_osworld/resolve/main/Windows-10-x64.qcow2.zip"

# Backwards-compatibility alias — old code referenced VMInstance as QEMUManager
VMInstance = QEMUManager


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
                    return
                except (requests.RequestException, IOError) as exc:
                    logger.error("Download interrupted: %s — retrying", exc)
                    sleep(5)


def ensure_base_image(vm_dir: Path, os_type: str = "Ubuntu") -> Path:
    """Download and extract the OSWorld base qcow2 image if not already present."""
    vm_dir = Path(vm_dir)
    vm_dir.mkdir(parents=True, exist_ok=True)

    if os_type == "Ubuntu":
        url = _UBUNTU_X86_URL
    elif os_type == "Windows":
        url = _WINDOWS_X86_URL
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


class OSWorldQEMUVMBackend(LocalQEMUVMBackend):
    """LocalQEMUVMBackend that auto-downloads OSWorld VM images from HuggingFace."""

    def ensure_resource(self, config: VMConfig) -> None:
        if self.path_to_vm is not None:
            logger.info("Using explicit VM image: %s", self.path_to_vm)
            return
        vm_dir = Path(self.cache_dir)
        base_image = ensure_base_image(vm_dir, config.os_type)
        logger.info("Base image ready: %s", base_image)

    def launch(self, config: VMConfig) -> LocalQEMUVM:
        self.ensure_resource(config)
        if self.path_to_vm is None:
            self.path_to_vm = str(ensure_base_image(Path(self.cache_dir), config.os_type))
        return super().launch(config)


__all__ = [
    "VM",
    "VMBackend",
    "VMConfig",
    "VMInstance",
    "LocalQEMUVM",
    "LocalQEMUVMBackend",
    "OSWorldQEMUVMBackend",
    "QEMUConfig",
    "QEMUManager",
    "ensure_base_image",
]
