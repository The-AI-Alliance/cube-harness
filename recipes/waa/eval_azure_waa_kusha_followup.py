"""WAA Azure follow-up eval — targeted test of the port-forwarding fix.

Verifies the forwarded_ports addition to VMResourceConfig + the new multi-tunnel
loop in cube-infra-azure: chrome/msedge tasks that previously hit
ECONNREFUSED 127.0.0.1:9222 should now connect via the host-side SSH tunnel.

Task selection: first 5 chrome and first 5 msedge tasks (10 total).

Usage:
    uv run recipes/waa/eval_azure_waa_kusha_followup.py
"""

import logging
import os
from datetime import datetime

from cube_infra_azure import AzureInfraConfig
from dotenv import load_dotenv
from waa_cube.benchmark import WAABenchmark
from waa_cube.computer import ComputerConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import GennyConfig
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

# First 5 chrome and 5 msedge tasks — exercise the port-forwarding fix.
CHROME_IDS = [
    "030eeff7-b492-4218-b312-701ec99ee0cc-wos",
    "06fe7178-4491-4589-810f-2e2bc9502122-wos",
    "121ba48f-9e17-48ce-9bc6-a4fb17a7ebba-wos",
    "2ae9ba84-3a0d-4d4c-8338-3a1478dc5fe3-wos",
    "35253b65-1c19-4304-8aa4-6884b8218fc0-wos",
]
MSEDGE_IDS = [
    "004587f8-6028-4656-94c1-681481abbc9c-wos",
    "049d3788-c979-4ea6-934d-3a35c4630faf-WOS",
    "1376d5e7-deb7-471a-9ecc-c5d4e155b0c8-wos",
    "1a1ec621-b675-4099-96a9-f702dc27afb4-wos",
    "1c9d2c6c-ae4b-4359-9a93-9d3c42f48417-wos",
]

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

    output_dir = make_experiment_output_dir("genny_azure_kusha_haiku_followup", "waa-cube")

    llm_config = LLMConfig(model_name="claude-haiku-4-5-20251001", temperature=1.0)
    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=system_prompt,
        max_actions=100,
        render_last_n_obs=3,
        enable_summarize=False,
        tools_as_text=False,
    )

    tool_config = ComputerConfig(
        action_space="pyautogui",
        require_a11y_tree=True,
        require_obs_winagent=True,
        observe_after_action=True,
    )

    benchmark = WAABenchmark(
        default_tool_config=tool_config,
        infra=INFRA,
    )
    benchmark.setup()

    keep_ids = CHROME_IDS + MSEDGE_IDS
    available = set(benchmark.task_metadata.keys())
    missing = [tid for tid in keep_ids if tid not in available]
    if missing:
        logging.warning("Task IDs not in benchmark.task_metadata: %s", missing)
    keep_ids = [tid for tid in keep_ids if tid in available]
    benchmark = benchmark.subset_from_list(keep_ids)
    logging.info("Follow-up eval: %d tasks (%d chrome, %d msedge)", len(keep_ids), len(CHROME_IDS), len(MSEDGE_IDS))

    exp = Experiment(
        name="waa_azure_kusha_haiku_followup",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=100,
    )

    try:
        print(f"\nFOLLOW-UP EVAL — parallel, output: {output_dir}")
        run_with_ray(exp, n_cpus=20)
    finally:
        deleted = INFRA.cleanup_orphaned_resources()
        if deleted:
            print(f"Cleaned up orphaned resources: {deleted}")


if __name__ == "__main__":
    main()
