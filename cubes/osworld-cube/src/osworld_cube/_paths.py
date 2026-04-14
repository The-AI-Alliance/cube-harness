"""OSWorld path constants — isolated here to avoid circular imports in __init__.py.

benchmark.py and computer.py import from this module directly so that __init__.py
can keep all of its imports at the top level (satisfying ruff E402).
"""

import cube

OSWORLD_BASE_DIR = cube.get_cache_dir("osworld-cube")
OSWORLD_REPO_DIR = OSWORLD_BASE_DIR / "OSWorld"
OSWORLD_VM_DIR = OSWORLD_BASE_DIR / "vm_data"
OSWORLD_CACHE_DIR = OSWORLD_BASE_DIR / "cache"
