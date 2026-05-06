"""WAA eval on Azure using Kusha's pre-built Specialized Windows image.

Uses a Specialized (non-sysprepped) Windows 11 image with UEFI + TPM.
SSH access is via VMAccessAgent injecting your local pubkey into the
Docker user's administrators_authorized_keys at launch time.

Prerequisites:
    - az login
    - Set AZURE_RESOURCE_GROUP (defaults to "ui_assist")
    - Set AZURE_STORAGE_ACCOUNT (defaults to "cubeexpvhd")

First run will provision the gallery image from HuggingFace (~30-90 min).
Subsequent runs skip provisioning and go straight to eval.

Usage:
    # Debug mode (sequential)
    uv run recipes/waa/eval_azure_waa_kusha.py debug

    # Eval mode (full benchmark, parallel)
    uv run recipes/waa/eval_azure_waa_kusha.py
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
from cube_harness.exp_runner import run_sequentially, run_with_ray
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
    image_name_suffix="-kusha",
    source_cache_blob="sources/waa-windows-prepared.qcow2",
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
- pyautogui.click(x, y)
- pyautogui.rightClick(x, y)
- pyautogui.doubleClick(x, y)
- pyautogui.typewrite('text', interval=0.05)
- pyautogui.hotkey('ctrl', 'c')
- pyautogui.press('enter')
- pyautogui.scroll(x, y, clicks=-3)

### Ending the task
- Call fail() if the task CANNOT be completed
- Call done() when the task is successfully COMPLETE
"""


def main(debug: bool) -> None:
    output_dir = make_experiment_output_dir("genny_azure_kusha", "waa-cube")

    llm_config = LLMConfig(model_name="claude-haiku-4-5-20251001", temperature=1.0)
    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=WAA_SYSTEM_PROMPT,
        max_actions=100,
        render_last_n_obs=3,
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

    if debug:
        # Limit to a single task to avoid creating 150+ episode dirs.
        first_id = next(iter(WAABenchmark.task_metadata))
        WAABenchmark.task_metadata = {first_id: WAABenchmark.task_metadata[first_id]}

    benchmark.setup()
    limit = 20
    type(benchmark).task_metadata = dict(list(type(benchmark).task_metadata.items())[:limit])
    logging.info("Trimmed WAABenchmark to first %d tasks for this run", limit)

    exp = Experiment(
        name="waa_azure_kusha_haiku",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=100,
    )

    try:
        if debug:
            print(f"\nDEBUG MODE — sequential, output: {output_dir}")
            run_sequentially(exp)
        else:
            print(f"\nEVAL MODE — parallel, output: {output_dir}")
            run_with_ray(exp, n_cpus=20)
    finally:
        deleted = INFRA.cleanup_orphaned_resources()
        if deleted:
            print(f"Cleaned up orphaned resources: {deleted}")


if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
    main(debug)
