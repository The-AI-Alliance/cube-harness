"""
Integration test: WebArenaVerifiedBenchmark + AWSInfraConfig auto-provisioning.

Provisions a Docker-host AMI (shopping_admin only, ~15 min first time),
launches an EC2 instance, runs the WAV debug suite (tasks 0 + 1 via run_debug_suite),
then unprovisions.

Run:
    cd cube-harness/integration-tests
    uv run --group aws python test_webarena_debug_aws.py

To skip reprovisioning (reuse an already-provisioned AMI):
    SKIP_PROVISION=1 uv run --group aws python test_webarena_debug_aws.py
"""

from __future__ import annotations

import logging
import os
import sys
import types

from cube.testing import run_debug_suite
from cube_infra_aws import AWSInfraConfig
from webarena_verified_cube.debug import get_debug_benchmark, make_debug_agent
from webarena_verified_cube.resources import WEBARENA_SHOPPING_ADMIN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("botocore", "urllib3.connectionpool"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger(__name__)

INFRA = AWSInfraConfig(image_name_suffix="-test")
RESOURCE = WEBARENA_SHOPPING_ADMIN
SKIP_PROVISION = os.environ.get("SKIP_PROVISION", "").strip() in ("1", "true", "yes")


def main() -> None:
    log.info("=== integration test: WebArenaVerifiedBenchmark + AWSInfraConfig ===")
    log.info("infra: %s", INFRA.fingerprint())
    log.info("resource: %s  images=%s", RESOURCE.name, RESOURCE.docker_images)

    # ── Step 1: clean up stale instances from previous runs ──────────────────
    active = INFRA.list_active()
    if active:
        log.info("Step 1: found %d active instance(s) — cleaning up", len(active))
        for h in active:
            h.close()
    else:
        log.info("Step 1: no active instances")

    # ── Step 2+3: provision (or skip) ────────────────────────────────────────
    if not SKIP_PROVISION:
        if INFRA.provision_status(RESOURCE) == "ready":
            log.info("Step 2: unprovisioning stale test AMI …")
            INFRA.unprovision(RESOURCE)
        else:
            log.info("Step 2: no stale AMI — skipping")
        log.info("Step 3: provisioning (~15 min) …")
        INFRA.provision(RESOURCE)
        log.info("Step 3: provisioned ✓  status=%s", INFRA.provision_status(RESOURCE))
    else:
        log.info("Step 2+3: SKIP_PROVISION=1 — reusing existing AMI")
        if INFRA.provision_status(RESOURCE) != "ready":
            log.error("No provisioned AMI found — run without SKIP_PROVISION=1 first")
            sys.exit(1)

    # ── Step 4: run debug suite via get_debug_benchmark(infra=INFRA) ─────────
    debug_module = types.SimpleNamespace(
        get_debug_benchmark=lambda: get_debug_benchmark(infra=INFRA),
        make_debug_agent=make_debug_agent,
    )
    log.info("Step 4: running WAV debug suite (tasks 0 + 1) …")
    results = run_debug_suite("webarena-verified-cube", debug_module)

    # ── Step 5: unprovision ───────────────────────────────────────────────────
    log.info("Step 5: unprovisioning test AMI …")
    INFRA.unprovision(RESOURCE)
    log.info("Step 5: unprovisioned ✓")

    failed = [r for r in results if r["error"] or r["reward"] != 1.0]
    if failed:
        log.error("FAILED — %d episode(s) did not pass: %s", len(failed), failed)
        sys.exit(1)
    log.info("=== SUCCESS — WebArenaVerifiedBenchmark + AWSInfraConfig works ===")


if __name__ == "__main__":
    main()
