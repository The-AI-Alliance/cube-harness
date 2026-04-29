"""Re-run the 2 chrome-CDP-failed tasks from the full Haiku eval.

Background: of 152 episodes, 2 failed at `_chrome_open_tabs_setup` after 30
retries × 5s — root cause traced to a port-allocation race in
cube/infra_utils.py:free_port (now fixed). The 7 episodes that recovered
within 2-3 retries are real successes and not retried here.

Mechanism: Experiment(resume=True) reuses the original output_dir and runs
only episodes that have an episode_config.json but no trajectory data
(i.e. setup never finished). retry_failed=True wouldn't help here because
it requires episode.metadata.json to exist, which our 2 setup-failures
never wrote.

Usage:
    uv run recipes/waa/eval_azure_waa_kusha_haiku_retry_failed.py
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from cube_infra_azure import AzureInfraConfig
from dotenv import load_dotenv
from waa_cube.benchmark import WAABenchmark
from waa_cube.computer import ComputerConfig

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

# Output dir of the full eval whose failed tasks we want to retry.
ORIGINAL_OUTPUT_DIR = Path(
    "/Users/kusha.sareen/cube_harness_results/20260427_143250_genny_azure_kusha_haiku_full_waa-cube"
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

    if not ORIGINAL_OUTPUT_DIR.exists():
        raise SystemExit(f"Original output dir not found: {ORIGINAL_OUTPUT_DIR}")

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

    exp = Experiment(
        name="waa_haiku_full",
        output_dir=ORIGINAL_OUTPUT_DIR,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=100,
        resume=True,
    )

    # Sanity-check: how many will be retried?
    to_run = exp.get_episodes_to_run()
    print(f"\nHAIKU RETRY — output: {ORIGINAL_OUTPUT_DIR}")
    print(f"Will retry {len(to_run)} episode(s):")
    for ep in to_run:
        print(f"  - {ep.task_config.task.id if hasattr(ep, 'task_config') else ep}")

    try:
        run_with_ray(exp, n_cpus=2)
    finally:
        deleted = INFRA.cleanup_orphaned_resources()
        if deleted:
            print(f"Cleaned up orphaned resources: {deleted}")


if __name__ == "__main__":
    main()
