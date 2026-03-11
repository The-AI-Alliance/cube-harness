"""OSWorld-specific VM constants and download utilities."""

import logging
import zipfile
from pathlib import Path

from cube import get_cache_dir

logger = logging.getLogger(__name__)

# OSWorld Docker image tag (bundles QEMU + Flask control server)
OSWORLD_DOCKER_IMAGE = "happysixd/osworld-docker"

# OSWorld VM image download URL (HuggingFace dataset)
OSWORLD_VM_ZIP_URL = "https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip"
OSWORLD_VM_QCOW2_NAME = "Ubuntu.qcow2"

# Pinned OSWorld commit for reproducibility
OSWORLD_COMMIT = "e695a10"

# Default OSWorld directories
OSWORLD_BASE_DIR = Path.home() / ".agentlab2" / "benchmarks" / "osworld"
OSWORLD_REPO_DIR = OSWORLD_BASE_DIR / "OSWorld"


def get_osworld_vm_image() -> str:
    """Download the OSWorld Ubuntu qcow2 disk image if not cached and return its path.

    Downloads ``Ubuntu.qcow2.zip`` from HuggingFace, extracts it, and caches the
    result under ``~/.cache/cube/osworld/``.  Subsequent calls return immediately.

    Returns:
        Absolute path to the local ``Ubuntu.qcow2`` file.
    """
    import requests
    from tqdm import tqdm

    cache_dir = Path(get_cache_dir("osworld"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    qcow2_path = cache_dir / OSWORLD_VM_QCOW2_NAME
    if qcow2_path.exists():
        logger.info("OSWorld VM image already cached at: %s", qcow2_path)
        return str(qcow2_path)

    zip_path = cache_dir / "Ubuntu.qcow2.zip"
    logger.info("Downloading OSWorld VM image from %s …", OSWORLD_VM_ZIP_URL)

    # Resumable download: use Range header if partial file exists
    downloaded = zip_path.stat().st_size if zip_path.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

    with requests.get(OSWORLD_VM_ZIP_URL, headers=headers, stream=True) as resp:
        if resp.status_code == 416:
            # Already fully downloaded
            logger.info("Zip already fully downloaded.")
        else:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(zip_path, "ab") as f, tqdm(
                desc="Ubuntu.qcow2.zip",
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                initial=downloaded,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    bar.update(len(chunk))

    logger.info("Extracting %s …", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    logger.info("OSWorld VM image ready at: %s", qcow2_path)
    return str(qcow2_path)
