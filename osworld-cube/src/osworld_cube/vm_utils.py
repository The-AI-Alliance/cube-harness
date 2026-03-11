"""OSWorld-specific VM constants and download utilities."""

from pathlib import Path

from cube import get_cache_dir

# OSWorld Docker image tag (bundles QEMU + Flask control server)
OSWORLD_DOCKER_IMAGE = "happysixd/osworld-docker"

# OSWorld HuggingFace repository
OSWORLD_HF_REPO = "xlangai/ubuntu_osworld"

# Pinned OSWorld commit for reproducibility
OSWORLD_COMMIT = "e695a10"

# Default OSWorld directories
OSWORLD_BASE_DIR = Path.home() / ".agentlab2" / "benchmarks" / "osworld"
OSWORLD_REPO_DIR = OSWORLD_BASE_DIR / "OSWorld"


def get_osworld_vm_image() -> str:
    """Download the OSWorld Ubuntu qcow2 disk image from HuggingFace if not cached.

    Returns:
        Absolute path to the local qcow2 file.

    Requires:
        huggingface-hub installed (listed in osworld-cube dependencies).
    """
    from huggingface_hub import hf_hub_download

    cache = get_cache_dir("osworld")
    path = hf_hub_download(
        repo_id=OSWORLD_HF_REPO,
        filename="ubuntu.qcow2",
        cache_dir=str(cache),
    )
    return path
