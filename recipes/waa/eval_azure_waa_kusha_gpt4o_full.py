"""WAA full-corpus eval — Azure GPT-4o on the full 152-task Windows corpus.

Companion to `eval_azure_waa_kusha_haiku_full.py` (Haiku 4.5) and
`eval_azure_waa_kusha_gpt5mini_full.py` (GPT-5-mini). Same Genny+axtree
setup, same full corpus, running on the LO-enabled image (waa-windows-vm-kusha-lo).

Model: `azure/gpt-4o` (`gpt-4o-2024-11-20` deployment, added to the
ui-assist Azure OpenAI resource by collaborator). Direct comparison point
to the paper's GPT-4o row (Table 4, OneOCR + ✓UIA): 13.3% overall.

Concurrency: n_cpus=10. n_cpus=20 still triggers persistent /setup/upload 502s
(separate from the port-collision fix already landed) — diagnosing that is a
separate workstream.

Prompt note: GPT-4o was returning empty action lists ~22% of the time at
n_cpus=20 — model produced text without tool calls and the agent treated
that as "stop". Prompt updated below to demand a tool call every step.

Usage:
    uv run recipes/waa/eval_azure_waa_kusha_gpt4o_full.py
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

**CRITICAL: Every response MUST include exactly one tool call. Never reply with
plain text only — there is no human reading your messages between steps. The
only ways to advance are run_pyautogui(code), wait(), done(), or fail().
If you are unsure, call run_pyautogui with your best guess; never stop without
calling done() or fail().**

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
8. Do not loop — if an action has no effect after 2 attempts, try a completely different approach
9. NEVER respond with text only — every step MUST end in a tool call (run_pyautogui, wait, done, or fail)\
"""


def main() -> None:
    today = datetime.today().strftime("%A, %B %d, %Y")
    system_prompt = WAA_SYSTEM_PROMPT.format(today=today)

    output_dir = make_experiment_output_dir("genny_azure_kusha_gpt4o_full", "waa-cube")

    # GPT-4o was returning empty action lists ~22% of the time at n=20 —
    # model produced text without tool calls and the agent treated as stop.
    # We rely on the strengthened system prompt (see WAA_SYSTEM_PROMPT) to
    # push toward tool calls; tool_choice="auto" so a genuinely stuck model
    # can still call fail() instead of looping to max_steps.
    llm_config = LLMConfig(model_name="azure/gpt-4o", temperature=1.0)
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

    # Full 152-task corpus on the LO-enabled image.
    logging.info("GPT-4o full eval: %d tasks", len(benchmark.task_metadata))

    exp = Experiment(
        name="waa_gpt4o_full",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=100,
    )

    try:
        print(f"\nGPT-4O FULL EVAL — output: {output_dir}")
        run_with_ray(exp, n_cpus=10)
    finally:
        deleted = INFRA.cleanup_orphaned_resources()
        if deleted:
            print(f"Cleaned up orphaned resources: {deleted}")


if __name__ == "__main__":
    main()
