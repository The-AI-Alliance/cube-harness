"""WAA LO smoke eval — 10 LibreOffice tasks (5 calc + 5 writer) on the new
LO-enabled image.

Uses the new image_name_suffix=-kusha-lo / source_cache_blob=...-lo.qcow2 to
keep the old -kusha image side-by-side. Image was verified good (all 8 apps
present via PowerShell Test-Path).

Usage:
    uv run recipes/waa/eval_azure_waa_kusha_lo_smoke.py
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

CALC_IDS = [
    "01b269ae-2111-4a07-81fd-3fcd711993b0-WOS",
    "035f41ba-6653-43ab-aa63-c86d449d62e5-WOS",
    "04d9aeaf-7bed-4024-bedb-e10e6f00eb7f-WOS",
    "0a2e43bf-b26c-4631-a966-af9dfa12c9e5-WOS",
    "0acbd372-ca7a-4507-b949-70673120190f-WOS",
]
WRITER_IDS = [
    "0810415c-bde4-4443-9047-d5f70165a697-WOS",
    "0a0faba3-5580-44df-965d-f562a99b291c-WOS",
    "0b17a146-2934-46c7-8727-73ff6b6483e8-WOS",
    "0e47de2a-32e0-456c-a366-8c607ef7a9d2-WOS",
    "0e763496-b6bb-4508-a427-fad0b6c3e195-WOS",
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

    output_dir = make_experiment_output_dir("genny_azure_kusha_haiku_lo_smoke", "waa-cube")

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

    keep_ids = CALC_IDS + WRITER_IDS
    available = set(benchmark.task_metadata.keys())
    missing = [tid for tid in keep_ids if tid not in available]
    if missing:
        logging.warning("Task IDs not in benchmark.task_metadata: %s", missing)
    keep_ids = [tid for tid in keep_ids if tid in available]
    benchmark = benchmark.subset_from_list(keep_ids)
    logging.info("LO smoke eval: %d tasks (%d calc, %d writer)", len(keep_ids), len(CALC_IDS), len(WRITER_IDS))

    exp = Experiment(
        name="waa_azure_kusha_haiku_lo_smoke",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=100,
    )

    try:
        print(f"\nLO SMOKE EVAL — parallel, output: {output_dir}")
        run_with_ray(exp, n_cpus=10)
    finally:
        deleted = INFRA.cleanup_orphaned_resources()
        if deleted:
            print(f"Cleaned up orphaned resources: {deleted}")


if __name__ == "__main__":
    main()
