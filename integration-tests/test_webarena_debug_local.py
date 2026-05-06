"""
Integration test: LocalInfraConfig + DockerServiceConfig lifecycle.

Uses the real webarena shopping_admin image to validate the full LocalInfraConfig
lifecycle:

    provision (docker pull) → launch → endpoint reachable → close → container gone

On macOS with Podman (Rosetta 2), PHP OPcache crashes in php-fpm.  The launch
script patches this by disabling OPcache immediately after container start and
restarting php-fpm — the production launch script is left unchanged.

Run:
    cd cube-harness/integration-tests
    uv run python test_webarena_debug_local.py

To skip re-pulling (reuse existing provision entry):
    SKIP_PROVISION=1 uv run python test_webarena_debug_local.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

import requests
from cube.infra_local import LocalInfraConfig, _load_active
from cube.resource import DockerServiceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INFRA = LocalInfraConfig()

# Local variant of the shopping_admin resource.
# Identical to WEBARENA_SHOPPING_ADMIN except the launch script includes a
# Rosetta 2 workaround: OPcache is disabled immediately after container start
# so that php-fpm does not crash under x86_64 emulation on Apple Silicon.
# The production script (used on cloud VMs running native amd64) is unchanged.
_SHOPPING_ADMIN_PORTS = [7780, 7781]
RESOURCE = DockerServiceConfig(
    name="webarena-shopping-admin-local",
    scope="benchmark",
    docker_images=["am1n3e/webarena-verified-shopping_admin"],
    services={
        "shopping_admin": 7780,
        "shopping_admin_ctrl": 7781,
    },
    endpoint_to_site={"shopping_admin": "shopping_admin"},
    launch_script="""\
#!/usr/bin/env bash
# Local launch script for webarena-verified shopping_admin.
# Differs from the cloud version: OPcache is disabled immediately after start
# to work around a Rosetta 2 / php-fpm SIGSEGV on Apple Silicon Macs.
set -euo pipefail

docker run -d \\
    --name webarena_shopping_admin \\
    -p 7780:80 \\
    -p 7781:8877 \\
    am1n3e/webarena-verified-shopping_admin

# Rosetta 2 fix: OPcache uses AVX instructions not supported under emulation.
# Disable it before supervisord/php-fpm fully initialises.
docker exec webarena_shopping_admin bash -c "
    mv /usr/local/etc/php/conf.d/docker-php-ext-opcache.ini \\
       /usr/local/etc/php/conf.d/docker-php-ext-opcache.ini.disabled 2>/dev/null || true
    supervisorctl restart php-fpm
" || true   # non-fatal on native Linux where OPcache works fine

# Magento can take 3-5 min to initialise (up to 300s).
healthy=0
for i in $(seq 1 150); do
    curl -sf http://localhost:7780/ > /dev/null 2>&1 && echo "healthy" && healthy=1 && break
    sleep 2
done
if [ "$healthy" -eq 0 ]; then
    echo "ERROR: shopping_admin did not become healthy after 300s" >&2
    exit 1
fi

# Extra warmup: Magento PHP caches take ~30s after first curl success before
# serving full pages to Playwright.
sleep 30
""",
)

SKIP_PROVISION = os.environ.get("SKIP_PROVISION", "").strip() in ("1", "true", "yes")
_REQUIRED_PORTS = [7780, 7781]


def _containers_on_ports(ports: list[int]) -> list[str]:
    """Return names of all containers (running or created) with any of the given
    host ports bound.

    Uses ``docker port <name>`` per container because Podman's ``{{.Ports}}``
    template omits host-side bindings.
    """
    port_set = {str(p) for p in ports}
    all_names = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    names = []
    for cname in all_names:
        cname = cname.strip()
        if not cname:
            continue
        port_out = subprocess.run(["docker", "port", cname], capture_output=True, text=True).stdout
        for line in port_out.splitlines():
            if " -> " in line:
                host_port = line.split(" -> ")[-1].rsplit(":", 1)[-1]
                if host_port in port_set:
                    names.append(cname)
                    break
    return names


def main() -> None:
    log.info("=== integration test: LocalInfraConfig + DockerServiceConfig ===")
    log.info("infra: %s", INFRA.fingerprint())
    log.info("resource: %s  images=%s", RESOURCE.name, RESOURCE.docker_images)

    # ── Step 1: clean up any stale container on required ports ──────────────
    conflicting = _containers_on_ports(_REQUIRED_PORTS)
    if conflicting:
        log.info("Step 1: stopping stale container(s) on ports %s: %s", _REQUIRED_PORTS, conflicting)
        for cname in conflicting:
            subprocess.run(["docker", "stop", cname], check=False, capture_output=True)
            subprocess.run(["docker", "rm", cname], check=False, capture_output=True)
    else:
        log.info("Step 1: no stale containers")

    # ── Step 2: provision (docker pull nginx:alpine) ─────────────────────────
    if not SKIP_PROVISION:
        log.info("Step 2: provisioning (docker pull) …")
        INFRA.provision(RESOURCE)
        log.info("Step 2: provisioned ✓  status=%s", INFRA.provision_status(RESOURCE))
    else:
        log.info("Step 2: SKIP_PROVISION=1 — reusing existing provision entry")
        if INFRA.provision_status(RESOURCE) != "ready":
            log.error("No provision entry — run without SKIP_PROVISION=1 first")
            sys.exit(1)

    # ── Step 3: launch ────────────────────────────────────────────────────────
    log.info("Step 3: launching %r …", RESOURCE.name)
    handle = INFRA.launch(RESOURCE)
    log.info("Step 3: launched ✓  run_id=%s", handle.run_id[:8])
    log.info("  endpoints: %s", handle.endpoints)

    try:
        # ── Step 4: verify endpoint is reachable ──────────────────────────────
        log.info("Step 4: verifying endpoint …")
        shopping_url = handle.endpoints.get("shopping_admin")
        if shopping_url is None:
            log.error("Step 4: FAILED — 'shopping_admin' endpoint missing from handle.endpoints")
            sys.exit(1)
        # No OPcache on Rosetta 2, so PHP is slow — allow up to 60s per request.
        resp = requests.get(shopping_url, timeout=60)
        if resp.status_code != 200:
            log.error("Step 4: FAILED — expected HTTP 200, got %d", resp.status_code)
            sys.exit(1)
        log.info("Step 4: endpoint %s → HTTP %d ✓", shopping_url, resp.status_code)

        # ── Step 5: verify container tracking ────────────────────────────────
        log.info("Step 5: verifying container tracking …")
        if not handle._container_ids:
            log.error("Step 5: FAILED — no containers tracked in handle")
            sys.exit(1)
        log.info("Step 5: handle tracks %d container(s) ✓", len(handle._container_ids))

        # ── Step 6: verify active.json registration ───────────────────────────
        log.info("Step 6: verifying active.json entry …")
        active = _load_active()
        entry = next(
            (e for e in active.values() if e.get("run_id") == handle.run_id),
            None,
        )
        if entry is None:
            log.error("Step 6: FAILED — run_id %s not found in active.json", handle.run_id[:8])
            sys.exit(1)
        log.info("Step 6: active.json entry present  type=%s ✓", entry.get("type"))

    finally:
        # ── Step 7: close handle ──────────────────────────────────────────────
        log.info("Step 7: closing handle …")
        handle.close()
        log.info("Step 7: handle closed ✓")

    # ── Step 8: verify container is gone and active.json entry removed ───────
    log.info("Step 8: verifying cleanup …")
    still_up = _containers_on_ports(_REQUIRED_PORTS)
    if still_up:
        log.error("Step 8: FAILED — container(s) still running after close(): %s", still_up)
        sys.exit(1)

    active_after = _load_active()
    if any(e.get("run_id") == handle.run_id for e in active_after.values()):
        log.error("Step 8: FAILED — active.json entry not removed after close()")
        sys.exit(1)

    log.info("Step 8: container gone, active.json cleaned up ✓")
    log.info("=== SUCCESS — LocalInfraConfig + DockerServiceConfig lifecycle works ===")


if __name__ == "__main__":
    main()
