"""WAA eval on Azure — Genny agent with GPT-5-mini and accessibility tree observations.

Uses AzureInfraConfig to launch fresh Windows 11 VMs per task. Mirrors eval_waa.py
except for the infra swap, and requires that the prepared Windows image
(OpenSSH Server + Azure VM Agent + sysprep — see waa-cube's Packer pipeline)
has been uploaded to the source_url on WAA_WINDOWS_RESOURCE.

Prerequisites:
    See cube-resources/cube-infra-azure/README.md for Azure setup.
    WAA_WINDOWS_ADMIN_PASSWORD must be set — Azure enforces 12–72 chars,
    3 of 4 character classes.

Usage:
    # Debug mode (1 task, sequential)
    uv run recipes/waa/eval_azure_waa.py debug

    # Eval mode (all tasks, sequential)
    uv run recipes/waa/eval_azure_waa.py
"""

import logging
import os
import sys

from cube_infra_azure import AzureInfraConfig
from dotenv import load_dotenv
from waa_cube.benchmark import WAABenchmark
from waa_cube.computer import ComputerConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import GennyConfig
from cube_harness.exp_runner import run_sequentially
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
for _noisy in ("azure.core.pipeline.policies.http_logging_policy", "azure.identity", "urllib3.connectionpool"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


_ADMIN_PASSWORD = os.environ.get("WAA_WINDOWS_ADMIN_PASSWORD")
if not _ADMIN_PASSWORD:
    raise SystemExit(
        "WAA_WINDOWS_ADMIN_PASSWORD is required for Azure Windows VMs.\n"
        "Set it in .env (12–72 chars, 3 of {lower, upper, digit, special})."
    )

# Standard_D8s_v3: 8 vCPU, 32 GB RAM — meets WAA_WINDOWS_RESOURCE's 8-core/8-GB
# floor and supports Trusted Launch (required for UEFI + vTPM).
INFRA = AzureInfraConfig(
    resource_group=os.environ.get("AZURE_RESOURCE_GROUP") or "ui_assist",
    storage_account=os.environ.get("AZURE_STORAGE_ACCOUNT") or "cubeexpvhd",
    vnet_name="vnet-westus2",
    nsg_name="osworld-nsg",
    vm_size="Standard_D8s_v3",
    windows_admin_password=_ADMIN_PASSWORD,
)

WAA_SYSTEM_PROMPT = """\
You are a desktop automation agent controlling a real Windows 11 computer.

## Observations
Each step you receive an element table listing interactive UI elements with columns:
index, tag, name, text, x, y, w, h

Where (x, y) is the top-left corner and (w, h) is the size of each element.
To click the center of element at row i: center_x = x + w//2, center_y = y + h//2

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
3. If the task is clearly impossible, call fail() immediately
4. Prefer hotkey shortcuts over mouse clicks when practical
5. After completing the task, verify by checking the next observation then call done()
6. Do not loop — if an action has no effect after 2 attempts, try a different approach\
"""


def main(debug: bool) -> None:
    output_dir = make_experiment_output_dir("genny_azure", "waa-cube")

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=WAA_SYSTEM_PROMPT,
        max_actions=15,
        render_last_n_obs=1,
        enable_summarize=False,
        tools_as_text=False,
    )

    tool_config = ComputerConfig(
        action_space="pyautogui",
        require_a11y_tree=True,
        observe_after_action=True,
    )

    benchmark = WAABenchmark(
        default_tool_config=tool_config,
        infra=INFRA,
    )
    benchmark.install()
    benchmark.setup()

    # Exclude chrome/msedge tasks — CDP setup has a timing issue (see eval_waa.py).
    keep_ids = [
        tid
        for tid, meta in benchmark.task_metadata.items()
        if meta.extra_info.get("domain") not in ("chrome", "msedge")
    ]
    benchmark = benchmark.subset_from_list(keep_ids)

    # Provision the Windows VM image into the Compute Gallery (idempotent).
    for resource in benchmark.resources:
        INFRA.provision(resource)

    exp = Experiment(
        name="waa_genny_gpt5_azure",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=15,
    )

    try:
        if debug:
            print("\n" + "=" * 60)
            print("DEBUG MODE: Running 1 WAA task sequentially on Azure")
            print("=" * 60)
            print(f"Output directory: {output_dir}")
            print(f"Model: {llm_config.model_name}")
            print(f"Infra: {INFRA.fingerprint()}")
            print("=" * 60 + "\n")
            run_sequentially(exp, debug_limit=1)
        else:
            print("\n" + "=" * 60)
            print("EVAL MODE: Running WAA tasks sequentially on Azure")
            print("=" * 60)
            print(f"Output directory: {output_dir}")
            print(f"Model: {llm_config.model_name}")
            print(f"Infra: {INFRA.fingerprint()}")
            print("=" * 60 + "\n")
            run_sequentially(exp)
    finally:
        deleted = INFRA.cleanup_orphaned_resources()
        if deleted:
            print(f"Cleaned up {len(deleted)} orphaned VM(s): {deleted}")


if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
    main(debug)
