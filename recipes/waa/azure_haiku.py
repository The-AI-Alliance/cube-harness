"""WAA full-corpus eval — Claude Haiku 4.5 on the full 152-task Windows corpus.

Companion to `eval_azure_waa_paper_repro.py` (GPT-4o-mini). Same Genny+axtree
setup, same full corpus, running on the LO-enabled image (waa-windows-vm-kusha-lo).

The paper doesn't have a Haiku row in Table 4, so this is exploratory rather
than a direct reproduction. Closest comparison points from the paper Table 4
(OneOCR + ✓UIA, no Navi grounding pipeline — closest to our setup):
    GPT-4o-mini → 7.3% overall
    GPT-4o      → 13.3% overall

Usage:
    uv run recipes/waa/eval_azure_waa_kusha_haiku_full.py
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
2. An element table listing interactive UI elements with columns:
   index, tag, name, text, x, y, w, h
3. The active window title
4. A list of all open windows
5. Clipboard contents (if any)

Where (x, y) is the top-left corner and (w, h) is the size of each element.
To click the center of element at row i: center_x = x + w//2, center_y = y + h//2

Prefer the element table for precise coordinates; use the screenshot for visual context.
Use the window title and window list to track which application is in focus.
You will see the last 3 observations in context — use this history to track progress.

## Actions
You control the computer by calling run_pyautogui(code) with valid Python/pyautogui code.

### Common pyautogui commands
- pyautogui.click(x, y)                       — left-click at pixel coordinates
- pyautogui.rightClick(x, y)                  — right-click at pixel coordinates
- pyautogui.doubleClick(x, y)                 — double-click at pixel coordinates
- pyautogui.typewrite('text', interval=0.05)  — type text character by character
- pyautogui.hotkey('ctrl', 'c')               — press key combination
- pyautogui.press('enter')                    — press a single key
- pyautogui.scroll(x, y, clicks=-3)           — scroll (negative clicks = down)
- pyautogui.dragTo(x, y, button='left')       — drag to coordinates

### Ending the task
- Call fail() if the task CANNOT be completed (infeasible tasks)
- Call done() when the task is successfully COMPLETE

## Strategy
1. Study the element table carefully to find the element you need to interact with
2. Calculate center coordinates: center_x = x + w//2, center_y = y + h//2
3. If an unexpected dialog or popup is blocking your task, dismiss it before proceeding
4. If the task is clearly impossible (missing app, contradictory requirements), call fail() immediately
5. Prefer hotkey shortcuts over mouse clicks when practical
6. Do NOT ask for clarification — always proceed with available information
7. After completing the task, verify by checking the next observation then call done()
8. Do not loop — if an action has no effect after 2 attempts, try a completely different approach\
"""


def main() -> None:
    today = datetime.today().strftime("%A, %B %d, %Y")
    system_prompt = WAA_SYSTEM_PROMPT.format(today=today)

    output_dir = make_experiment_output_dir("genny_azure_kusha_haiku_full", "waa-cube")

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

    # Pre-run cleanup. Two phases:
    #
    #   1. (opt-in via WAA_CLEAN_START=1) cleanup_stale(60s) deletes any
    #      cube-* VM older than 60s. Use ONLY when no other eval is running
    #      in this resource group — it doesn't distinguish "stranded VM
    #      from a prior crashed run" from "VM in active use by a parallel
    #      eval", and at this account scale we sometimes intentionally
    #      run two evals concurrently. Off by default to avoid friendly-fire.
    #   2. cleanup_orphaned_resources() — always safe. Only sweeps NICs /
    #      IPs / disks that aren't attached to a VM. Anything in active use
    #      is by definition attached, so concurrent evals stay untouched.
    #
    # If killed runs are leaving stranded VMs, manually:
    #   WAA_CLEAN_START=1 uv run recipes/waa/azure_haiku.py
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

    # Full 152-task corpus on the LO-enabled image.
    logging.info("Haiku full eval: %d tasks", len(bench_config.task_metadata))

    exp = Experiment(
        name="waa_haiku_full",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=bench_config,
        infra=INFRA,
        max_steps=100,
    )

    try:
        print(f"\nHAIKU FULL EVAL — output: {output_dir}")
        run_with_ray(exp, n_cpus=50)
    finally:
        # Post-run: same safety stance as pre-run. Don't unconditionally call
        # cleanup_stale — concurrent evals' VMs would be in scope. Just sweep
        # the unattached debris from this run's per-task handle.close() calls.
        # If user wants total annihilation post-run they re-run with
        # WAA_CLEAN_START=1 next time, or run cleanup_stale manually.
        print("\n--- post-run cleanup ---")
        leftover = INFRA.cleanup_orphaned_resources()
        if leftover:
            n = sum(len(v) for v in leftover.values())
            print(f"Cleaned up {n} orphaned resource(s) from this run")


if __name__ == "__main__":
    main()
