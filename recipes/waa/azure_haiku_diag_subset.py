"""WAA n=50 eval on a 50-task subset, with dead-VM diagnostic capture enabled.

Companion to `azure_haiku.py`. Goal: reproduce the dead-Flask failure mode
(~32% rate at n=50) on a small enough corpus that we can iterate fast and
get diagnostic dumps from any VM that fails the health gate.

Diagnostic dumps land in `/tmp/dead-flask-eval-diag/<task_id>_<vm_name>.txt`
and are written by `WAATask._dump_dead_vm_diagnostics` automatically when
the health gate fast-fails or times out.

Usage:
    uv run recipes/waa/azure_haiku_diag_subset.py
"""

import logging
import os
from datetime import datetime

from cube_infra_azure import AzureInfraConfig
from dotenv import load_dotenv
from waa_cube.benchmark import WAABenchmark
from waa_cube.computer import ComputerConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import BudgetConfig, GennyConfig
from cube_harness.exp_runner import run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
for _noisy in ("azure.core.pipeline.policies.http_logging_policy", "azure.identity", "urllib3.connectionpool"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

INFRA = AzureInfraConfig(
    resource_group=os.environ.get("AZURE_RESOURCE_GROUP") or "ui_assist",
    storage_account=os.environ.get("AZURE_STORAGE_ACCOUNT") or "cubeexpvhd",
    vnet_name="vnet-westus2",
    nsg_name="osworld-nsg",
    windows_admin_username="Docker",
    image_name_suffix="-kusha-lo",
    source_cache_blob="sources/waa-windows-prepared-lo.qcow2",
)

WAA_SYSTEM_PROMPT = """\
You are a desktop automation agent controlling a real Windows 11 computer.

## Environment
- OS: Windows 11
- Today's date: {today}

## Observations
Each step you receive:
1. A screenshot of the current screen (1280×800)
2. An element table listing interactive UI elements

## Actions
You control the computer by calling run_pyautogui(code) with valid Python/pyautogui code.

### Ending the task
- Call fail() if the task CANNOT be completed
- Call done() when the task is successfully COMPLETE\
"""


def main() -> None:
    today = datetime.today().strftime("%A, %B %d, %Y")
    system_prompt = WAA_SYSTEM_PROMPT.format(today=today)

    output_dir = make_experiment_output_dir("genny_azure_haiku_diag_subset", "waa-cube")

    llm_config = LLMConfig(model_name="claude-haiku-4-5-20251001", temperature=1.0)
    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=system_prompt,
        budget=BudgetConfig(max_actions=100),
        enable_summarize=False,
    )

    tool_config = ComputerConfig(
        action_space="pyautogui",
        require_a11y_tree=True,
        require_obs_winagent=True,
        observe_after_action=True,
    )

    print("--- pre-run cleanup ---")
    if os.environ.get("WAA_CLEAN_START") == "1":
        stale_vms = INFRA.cleanup_stale(max_age_seconds=60)
        if stale_vms:
            print(f"WAA_CLEAN_START=1: deleted {len(stale_vms)} stale VM(s) older than 60s")
    pre_deleted = INFRA.cleanup_orphaned_resources()
    if pre_deleted:
        n = sum(len(v) for v in pre_deleted.values())
        print(f"Cleaned up {n} orphaned resource(s) from prior runs")

    bench_config = WAABenchmark(
        tool_config=tool_config,
        infra=INFRA,
    )

    # Subset to first 50 tasks (sorted by id for determinism). Goal: 50 tasks
    # at n=50 = every worker gets ~1 task = round-1 finishes fast = dead-Flask
    # retries fire ~10-15 min in instead of ~hour-plus on the full 152 corpus.
    full_ids = sorted(bench_config.task_metadata.keys())
    subset_ids = full_ids[:50]
    bench_config.task_ids = subset_ids
    logging.info("DIAG SUBSET: %d tasks (of %d) at n=50", len(subset_ids), len(full_ids))

    exp = Experiment(
        name="waa_haiku_diag_subset",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=bench_config,
        infra=INFRA,
        max_steps=100,
    )

    try:
        print(f"\nHAIKU DIAG SUBSET — output: {output_dir}")
        print("Dead-VM diagnostic dumps will land in: /tmp/dead-flask-eval-diag/")
        run_with_ray(exp, n_cpus=50)
    finally:
        print("\n--- post-run cleanup ---")
        leftover = INFRA.cleanup_orphaned_resources()
        if leftover:
            n = sum(len(v) for v in leftover.values())
            print(f"Cleaned up {n} orphaned resource(s) from this run")


if __name__ == "__main__":
    main()
