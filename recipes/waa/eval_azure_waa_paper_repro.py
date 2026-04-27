"""WAA paper-reproduction recipe.

Targets the apples-to-apples row from Table 4 of the WindowsAgentArena paper
(arXiv 2409.08264): "OneOCR + Proprietary models, ✓ UIA" — i.e. axtree + screenshot
+ a proprietary LLM. Our Genny agent is the closest match (we feed the agent the
UIA-derived element table + a screenshot — no Omniparser/Grounding-DINO).

Paper Table 4, OneOCR + ✓UIA row:
    GPT-4o-mini  →  7.3% overall  (Web 7.3%, Browser 20.8%, Windows-System 8.3%,
                                   Coding 9.8%, Office/Media 0%)
    GPT-4o       →  13.3% overall (Web 20.0%, Browser 29.2%, Windows-System 9.1%,
                                   Coding 25.3%, Office/Media 0%)

We filter LibreOffice (Office bucket) because our Windows image is missing LO.
Paper's score on Office is 0% for both models, so excluding it doesn't lower the
per-task pass rate — but we should report against the non-Office subset:

    Non-LO target ≈ (overall%) × 152 / (152 − 43 LO tasks) ≈ overall × 1.39

    GPT-4o-mini   → expect ~10% on the 109-task non-LO subset
    GPT-4o        → expect ~18% on the 109-task non-LO subset

Usage:
    uv run recipes/waa/eval_azure_waa_paper_repro.py             # gpt-4o
    WAA_MODEL=gpt-4o-mini uv run recipes/waa/eval_azure_waa_paper_repro.py
"""

import logging
import os
import sys
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

# WAA_MODEL env var lets you swap between gpt-4o (paper config) and gpt-4o-mini
# (cheaper for iteration). LiteLLM-style "azure/<deployment-name>" pattern.
_MODEL = os.environ.get("WAA_MODEL", "gpt-4o")
_MODEL_NAME = f"azure/{_MODEL}"

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

    output_dir = make_experiment_output_dir(f"genny_azure_kusha_paper_repro_{_MODEL.replace('/', '_')}", "waa-cube")

    llm_config = LLMConfig(model_name=_MODEL_NAME, temperature=1.0)
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

    # Run the FULL 152-task corpus including LibreOffice. LibreOffice tasks
    # currently fail (LO not installed in our Windows image) — they'll evaluate
    # to 0 trivially. Reported scores will mirror the paper's where Office=0%
    # for both GPT-4o and GPT-4o-mini, so this still gives a directly comparable
    # overall number. Once the image is rebuilt with LO, these tasks become
    # the new investigation surface.
    logging.info("Paper repro: full 152-task corpus, model=%s", _MODEL_NAME)

    exp = Experiment(
        name=f"waa_paper_repro_{_MODEL.replace('/', '_')}",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=100,
    )

    try:
        print(f"\nPAPER-REPRO EVAL — model={_MODEL_NAME}, output: {output_dir}")
        run_with_ray(exp, n_cpus=20)
    finally:
        deleted = INFRA.cleanup_orphaned_resources()
        if deleted:
            print(f"Cleaned up orphaned resources: {deleted}")


if __name__ == "__main__":
    main()
