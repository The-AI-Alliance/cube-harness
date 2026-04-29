"""WAA SoM smoke — 1 task with Set-of-Marks visual prompting.

Verifies the SoM pipeline end-to-end:
  - WAATask._postprocess_som() draws numbered red boxes on the screenshot
    using the accessibility tree, and replaces the axtree with a coords-free
    `index/tag/name/text` table.
  - Computer.update_marks() registers [x,y,w,h] per tag so run_pyautogui()
    prepends `tag_N = (cx, cy)` lines, letting the agent write
    `pyautogui.click(*tag_3)` instead of computing pixel centres.

Task chosen: 7c70e16b-…-WOS (file_explorer, "Sort files by date modified").
Haiku solved this in 2 agent steps without SoM, so SoM should solve it too —
making it a clean signal on whether the visual marks confuse the model.

If it works, next steps:
  - Run the full corpus with use_som=True for direct comparison vs. Haiku full
  - Compare per-domain win rates (chrome/file_explorer/settings/etc.)

Usage:
    uv run recipes/waa/eval_azure_waa_kusha_haiku_som_smoke.py
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

# 2-step file_explorer task that Haiku solved cleanly without SoM.
SMOKE_TASK_ID = "7c70e16b-e14f-4baa-b046-3e022b2d0305-WOS"

# SoM-aware system prompt. Differences vs the non-SoM prompt:
#  - Element table has columns `index/tag/name/text` (no x,y,w,h)
#  - Screenshot has numbered red boxes around each element
#  - Agent clicks via `pyautogui.click(*tag_N)` — tag_N is auto-injected by
#    run_pyautogui() with the box centre, no math needed
WAA_SOM_SYSTEM_PROMPT = """\
You are a desktop automation agent controlling a real Windows 11 computer.

## Environment
- OS: Windows 11
- Today's date: {today}

## Observations
Each step you receive:
1. A screenshot of the current screen (1280×800), annotated with NUMBERED RED
   BOUNDING BOXES around interactive UI elements. Each box has a small black
   tag with a white number at its bottom-left corner — that is the element's
   index.
2. An element table listing those same numbered elements with columns:
   index, tag, name, text
3. The active window title
4. A list of all open windows
5. Clipboard contents (if any)

The element table tells you WHAT each numbered box is. The screenshot tells you
WHERE it is visually.

You will see the last 3 observations in context — use this history to track
progress.

## Actions
You control the computer by calling run_pyautogui(code) with valid Python/pyautogui code.

### Set-of-Marks variables
For each numbered box you see, a variable `tag_N` is automatically defined as
the (x, y) centre of that box. So you can click element 5 with simply:
    pyautogui.click(*tag_5)
You do NOT need to compute pixel coordinates yourself. Always prefer
`pyautogui.click(*tag_N)` over `pyautogui.click(x, y)`.

### Common pyautogui commands
- pyautogui.click(*tag_N)                     — click element N
- pyautogui.rightClick(*tag_N)                — right-click element N
- pyautogui.doubleClick(*tag_N)               — double-click element N
- pyautogui.typewrite('text', interval=0.05)  — type text
- pyautogui.hotkey('ctrl', 'c')               — keyboard shortcut
- pyautogui.press('enter')                    — single key
- pyautogui.scroll(*tag_N, clicks=-3)         — scroll over an element

### Ending the task
- Call fail() if the task CANNOT be completed (infeasible tasks)
- Call done() when the task is successfully COMPLETE

## Strategy
1. Look at the numbered boxes in the screenshot to find the element you need
2. Cross-reference with the element table to confirm what it is
3. Click via `pyautogui.click(*tag_N)` — never compute pixel coords
4. If a dialog blocks your task, dismiss it before proceeding
5. If the task is clearly impossible, call fail() immediately
6. Prefer hotkey shortcuts over mouse clicks when practical
7. Do NOT ask for clarification — proceed with available information
8. After completing the task, verify in the next observation, then call done()
9. Do not loop — if an action has no effect after 2 attempts, change approach\
"""


def main() -> None:
    today = datetime.today().strftime("%A, %B %d, %Y")
    system_prompt = WAA_SOM_SYSTEM_PROMPT.format(today=today)

    output_dir = make_experiment_output_dir("genny_azure_kusha_haiku_som_smoke", "waa-cube")

    llm_config = LLMConfig(model_name="claude-haiku-4-5-20251001", temperature=1.0)
    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=system_prompt,
        max_actions=20,  # smoke task should solve in <10
        render_last_n_obs=3,
        enable_summarize=False,
        tools_as_text=False,
    )

    tool_config = ComputerConfig(
        action_space="pyautogui",
        require_a11y_tree=True,  # SoM needs the axtree to know where to draw boxes
        require_obs_winagent=True,
        observe_after_action=True,
    )

    benchmark = WAABenchmark(
        default_tool_config=tool_config,
        infra=INFRA,
        use_som=True,
    )
    benchmark.setup()

    # Filter to just the smoke task by overriding task_metadata.
    if SMOKE_TASK_ID not in benchmark.task_metadata:
        raise SystemExit(f"Smoke task '{SMOKE_TASK_ID}' not in metadata")
    only = {SMOKE_TASK_ID: benchmark.task_metadata[SMOKE_TASK_ID]}
    object.__setattr__(benchmark, "task_metadata", only)
    logging.info("SoM smoke: 1 task -> %s", SMOKE_TASK_ID)

    exp = Experiment(
        name="waa_haiku_som_smoke",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=20,
    )

    try:
        print(f"\nSOM SMOKE — output: {output_dir}")
        run_with_ray(exp, n_cpus=1)
    finally:
        deleted = INFRA.cleanup_orphaned_resources()
        if deleted:
            print(f"Cleaned up orphaned resources: {deleted}")


if __name__ == "__main__":
    main()
