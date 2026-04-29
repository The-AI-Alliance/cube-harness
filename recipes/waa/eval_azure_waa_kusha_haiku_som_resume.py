"""Resume the paused Haiku SoM full eval at n_cpus=20.

The probe (`probe_n20_502.py`) ran 20 VMs in parallel and saw 0/20 stuck — no
upload-502s. That contradicts the earlier hypothesis that we needed to stay
at n=10 to avoid 502s. We're testing whether the actual fix was the
port-allocation race (already landed in `cube/infra_utils.free_port`).

If this resume run lands cleanly at n=20, we can update RESULTS.md to retire
the "stay at n=10" recommendation.

State at pause: 75/152 finalized, 23 wins (30.7%), $38.03.
Resume picks up the 77 unstarted episodes via `resume=True`.

Usage:
    uv run recipes/waa/eval_azure_waa_kusha_haiku_som_resume.py
"""

import logging
import os
import time
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

ORIGINAL_OUTPUT_DIR = Path(
    "/Users/kusha.sareen/cube_harness_results/20260429_101553_genny_azure_kusha_haiku_som_full_waa-cube"
)

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


def wait_for_quota(infra: AzureInfraConfig, vms_needed: int, vcpus_per_vm: int = 8,
                   poll_s: float = 15.0, max_wait_s: float = 600.0) -> None:
    """Block until enough Standard DSv3 quota is free for `vms_needed` D8s_v3 VMs."""
    needed = vms_needed * vcpus_per_vm
    compute = infra._compute()
    location = "westus2"
    t0 = time.time()
    while True:
        current = limit = 0
        for u in compute.usage.list(location):
            if u.name and u.name.value == "standardDSv3Family":
                current, limit = int(u.current_value or 0), int(u.limit or 0)
                break
        free = limit - current
        if free >= needed:
            logging.info("quota OK: %d/%d used, %d free, need %d", current, limit, free, needed)
            return
        elapsed = time.time() - t0
        if elapsed > max_wait_s:
            raise RuntimeError(f"quota wait exceeded {max_wait_s:.0f}s; have {free}, need {needed}")
        logging.info("quota wait: %d/%d used, %d free, need %d (waited %.0fs)",
                     current, limit, free, needed, elapsed)
        time.sleep(poll_s)


def main() -> None:
    today = datetime.today().strftime("%A, %B %d, %Y")
    system_prompt = WAA_SOM_SYSTEM_PROMPT.format(today=today)

    if not ORIGINAL_OUTPUT_DIR.exists():
        raise SystemExit(f"Original output dir not found: {ORIGINAL_OUTPUT_DIR}")

    n_cpus = 20
    wait_for_quota(INFRA, vms_needed=n_cpus)

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

    exp = Experiment(
        name="waa_haiku_som_full",
        output_dir=ORIGINAL_OUTPUT_DIR,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=100,
        resume=True,  # picks up unstarted episodes only (already-finalized are kept)
    )

    print(f"\nHAIKU SOM RESUME @ n_cpus={n_cpus} — output: {ORIGINAL_OUTPUT_DIR}")
    try:
        run_with_ray(exp, n_cpus=n_cpus)
    finally:
        deleted = INFRA.cleanup_orphaned_resources()
        if deleted and any(deleted.values()):
            print(f"Cleaned up orphaned resources: {deleted}")


if __name__ == "__main__":
    main()
