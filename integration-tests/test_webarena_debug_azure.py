"""
Integration test: WebArenaVerifiedBenchmark + AzureInfraConfig auto-provisioning.

Provisions a Docker-host gallery image (shopping_admin only, ~13 min first time),
launches a VM, runs the WAV debug suite (tasks 0 + 1 via run_debug_suite), then
unprovisions.

Run:
    cd cube-harness/integration-tests
    uv run --group azure python test_webarena_debug_azure.py

To skip reprovisioning (reuse an already-provisioned gallery image):
    SKIP_PROVISION=1 uv run --group azure python test_webarena_debug_azure.py
"""

from __future__ import annotations

import logging
import os
import sys
import types

from cube.testing import run_debug_suite
from cube_infra_azure import AzureInfraConfig
from webarena_verified_cube.debug import get_debug_benchmark, make_debug_agent
from webarena_verified_cube.resources import WEBARENA_SHOPPING_ADMIN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("azure.core.pipeline.policies.http_logging_policy", "azure.identity", "urllib3.connectionpool"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger(__name__)

INFRA = AzureInfraConfig(
    resource_group="ui_assist",
    storage_account="cubeexpvhd",
    vnet_name="vnet-westus2",
    nsg_name="osworld-nsg",
    image_name_suffix="-test",
)

RESOURCE = WEBARENA_SHOPPING_ADMIN
SKIP_PROVISION = os.environ.get("SKIP_PROVISION", "").strip() in ("1", "true", "yes")


def main() -> None:
    log.info("=== integration test: WebArenaVerifiedBenchmark + AzureInfraConfig ===")
    log.info("infra: %s", INFRA.fingerprint())
    log.info("resource: %s  images=%s", RESOURCE.name, RESOURCE.docker_images)

    # ── Step 1: clean up stale VMs and orphaned resources ────────────────────
    active = INFRA.list_active()
    if active:
        log.info("Step 1: found %d active VM(s) — cleaning up", len(active))
        for h in active:
            h.close()
    else:
        log.info("Step 1: no active VMs")
    INFRA.cleanup_orphaned_resources()

    # ── Step 2+3: provision (or skip) ────────────────────────────────────────
    if not SKIP_PROVISION:
        if INFRA.provision_status(RESOURCE) == "ready":
            log.info("Step 2: unprovisioning stale test image …")
            INFRA.unprovision(RESOURCE)
        else:
            log.info("Step 2: no stale image — skipping")
        log.info("Step 3: provisioning (~13 min) …")
        INFRA.provision(RESOURCE)
        log.info("Step 3: provisioned ✓  status=%s", INFRA.provision_status(RESOURCE))
    else:
        log.info("Step 2+3: SKIP_PROVISION=1 — reusing existing gallery image")
        if INFRA.provision_status(RESOURCE) != "ready":
            log.error("No provisioned image found — run without SKIP_PROVISION=1 first")
            sys.exit(1)

    # ── Step 4: run debug suite via get_debug_benchmark(infra=INFRA) ─────────
    # The benchmark auto-launches the VM in setup() and closes it in close().
    debug_module = types.SimpleNamespace(
        get_debug_benchmark=lambda: get_debug_benchmark(infra=INFRA),
        make_debug_agent=make_debug_agent,
    )
    log.info("Step 4: running WAV debug suite (tasks 0 + 1) …")
    results = run_debug_suite("webarena-verified-cube", debug_module)

    # ── Step 5: unprovision ───────────────────────────────────────────────────
    log.info("Step 5: unprovisioning test image …")
    INFRA.unprovision(RESOURCE)
    log.info("Step 5: unprovisioned ✓")

    failed = [r for r in results if r["error"] or r["reward"] != 1.0]
    if failed:
        log.error("FAILED — %d episode(s) did not pass: %s", len(failed), failed)
        sys.exit(1)
    log.info("=== SUCCESS — WebArenaVerifiedBenchmark + AzureInfraConfig works ===")


if __name__ == "__main__":
    main()
