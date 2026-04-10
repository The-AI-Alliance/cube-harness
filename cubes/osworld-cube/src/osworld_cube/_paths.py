"""OSWorld path constants shared across benchmark and computer modules."""

import cube

OSWORLD_BASE_DIR = cube.get_cache_dir("osworld-cube")
OSWORLD_REPO_DIR = OSWORLD_BASE_DIR / "OSWorld"
OSWORLD_VM_DIR = OSWORLD_BASE_DIR / "vm_data"
OSWORLD_CACHE_DIR = OSWORLD_BASE_DIR / "cache"
