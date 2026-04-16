"""Canonical DockerServiceConfig declarations for WebArena-Verified sites.

Each constant represents one WAV Docker stack.  Pass the desired resource(s) via
``resources=[...]`` when constructing :class:`WebArenaVerifiedBenchmark` with
``infra=<InfraConfig>``.

Usage::

    from webarena_verified_cube.resources import WEBARENA_SHOPPING_ADMIN
    from webarena_verified_cube.debug import get_debug_benchmark

    benchmark = get_debug_benchmark(infra=my_infra)   # uses WEBARENA_SHOPPING_ADMIN internally
"""

from __future__ import annotations

from pathlib import Path

from cube.resource import DockerServiceConfig

_SCRIPTS_DIR = Path(__file__).parent / "scripts"


def _script(name: str) -> str:
    """Return the contents of a launch script from the scripts/ directory."""
    return (_SCRIPTS_DIR / name).read_text()


# ── shopping_admin ────────────────────────────────────────────────────────────
# Magento-based admin portal.  Host ports: 7780 (web UI), 7781 (env-ctrl API).
WEBARENA_SHOPPING_ADMIN = DockerServiceConfig(
    name="webarena-shopping-admin",
    scope="benchmark",
    docker_images=["am1n3e/webarena-verified-shopping_admin"],
    services={
        "shopping_admin": 7780,  # web UI  (container port 80)
        "shopping_admin_ctrl": 7781,  # env-ctrl (container port 8877)
    },
    endpoint_to_site={
        "shopping_admin": "shopping_admin",
        # shopping_admin_ctrl is not a browser site — no mapping
    },
    launch_script=_script("shopping_admin_launch.sh"),
)
