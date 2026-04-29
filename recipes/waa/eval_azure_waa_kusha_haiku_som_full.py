"""WAA full-corpus eval — Claude Haiku 4.5 with Set-of-Marks visual prompting.

Companion to `eval_azure_waa_kusha_haiku_full.py` (axtree-only, no SoM).
Same Genny + same VM image; the only difference is `use_som=True` on the
benchmark, which annotates each screenshot with numbered red bounding boxes
and replaces the axtree text with a coords-free `index/tag/name/text`
table. The agent clicks via `pyautogui.click(*tag_N)` instead of computing
pixel centres from x,y,w,h.

Validated end-to-end via eval_azure_waa_kusha_haiku_som_smoke.py (1 task,
won in 2 steps with reward 1.0). This run measures whether SoM helps,
hurts, or is neutral vs. the axtree-only baseline (which scored 36.2%).

Usage:
    uv run recipes/waa/eval_azure_waa_kusha_haiku_som_full.py
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

WAA_SYSTEM_PROMPT = """\
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
    system_prompt = WAA_SYSTEM_PROMPT.format(today=today)

    output_dir = make_experiment_output_dir("genny_azure_kusha_haiku_som_full", "waa-cube")

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
        use_som=True,
    )
    benchmark.setup()

    # Full 152-task corpus on the LO-enabled image, with SoM annotations.
    logging.info("Haiku SoM full eval: %d tasks", len(benchmark.task_metadata))

    exp = Experiment(
        name="waa_haiku_som_full",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=100,
    )

    try:
        print(f"\nHAIKU SOM FULL EVAL — output: {output_dir}")
        run_with_ray(exp, n_cpus=10)
    finally:
        deleted = INFRA.cleanup_orphaned_resources()
        if deleted:
            print(f"Cleaned up orphaned resources: {deleted}")


if __name__ == "__main__":
    main()
