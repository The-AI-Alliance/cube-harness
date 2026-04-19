"""
Integration test: provision + run_debug_episode on OSWorld with AzureInfraConfig.

Uses image_name_suffix="-test" so the test creates its own gallery image definition
("osworld-ubuntu-vm-test"), leaving the team's "osworld-ubuntu-vm/1.0.0" untouched.

Steps:
  1. Clean up any stale "-test" VMs from previous runs
  2. Unprovision the test image if it exists (delete gallery image + ProvisionStore entry)
  3. Provision from scratch: full ~40-min bootstrap pipeline
  4. get_debug_benchmark(infra) → first task → run_debug_episode
  5. Unprovision: clean up the test gallery image after the run

Run:
    cd cube-resources/cube-infra-azure
    uv run python test_run_debug_agent.py
"""

from __future__ import annotations

import json
import logging
import sys

from cube.testing import run_debug_episode
from cube_infra_azure import AzureInfraConfig
from osworld_cube.debug import get_debug_benchmark, make_debug_agent

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
    # isolates test from the team's "osworld-ubuntu-vm/1.0.0"
    image_name_suffix="-test",
)


def main() -> None:
    log.info("=== integration test: provision + run_debug_episode (Azure) ===")
    log.info("infra: %s", INFRA.fingerprint())

    resources = get_debug_benchmark(infra=INFRA).resources
    log.info("benchmark resources: %s", [r.name for r in resources])

    # ── Step 1: clean up any stale test VMs + orphaned NICs/IPs/disks ───────────
    active = INFRA.list_active()
    if active:
        log.info("Step 1: found %d active VM(s) — cleaning up", len(active))
        for handle in active:
            log.info("  closing run_id=%s", handle.run_id[:8])
            handle.close()
    else:
        log.info("Step 1: no active VMs")
    INFRA.cleanup_orphaned_resources()

    # ── Step 2: unprovision test image (clean slate for full reprovision) ──────
    for resource in resources:
        if INFRA.provision_status(resource) == "ready":
            log.info("Step 2: unprovisioning stale test image for %s …", resource.name)
            INFRA.unprovision(resource)

    # ── Step 3: provision from scratch (~40 min bootstrap pipeline) ───────────
    for resource in resources:
        log.info("Step 3: provisioning %s-test (this takes ~40 min) …", resource.name)
        INFRA.provision(resource)

    # ── Step 4: run a full debug episode via get_debug_benchmark ──────────────
    log.info("Step 4: running debug episode …")
    result = None
    try:
        benchmark = get_debug_benchmark(infra=INFRA)
        benchmark.install()
        benchmark.setup()

        task_configs = list(benchmark.get_task_configs())
        if not task_configs:
            log.error("No debug tasks found — aborting")
            sys.exit(1)

        tc = task_configs[0]
        task = tc.make()
        agent = make_debug_agent(tc.task_id)

        result = run_debug_episode(task, agent)
        log.info("=== Episode report ===")
        print(json.dumps(result, indent=2, default=str))

        benchmark.close()
    finally:
        # ── Step 5: unprovision test image (cleanup) — always runs ────────────
        for resource in resources:
            log.info("Step 5: cleaning up test image for %s …", resource.name)
            INFRA.unprovision(resource)

    if result is not None and result.get("reward", 0) > 0 and not result.get("error"):
        log.info("SUCCESS — Azure infra + OSWorld debug episode passed.")
    else:
        log.error("FAILED — see report above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
