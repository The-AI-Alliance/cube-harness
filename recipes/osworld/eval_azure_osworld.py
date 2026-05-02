"""OSWorld eval on Azure — Genny agent with GPT-5-mini and accessibility tree observations.

Uses AzureInfraConfig to launch fresh VMs per task. Mirrors the non-Azure
OSWorld recipe configuration while using Azure-backed VM provisioning.

Prerequisites:
    See cube-resources/cube-infra-azure/README.md for full setup instructions.

Usage:
    # Debug mode (debug_tasks.json, sequential)
    uv run recipes/osworld/eval_azure_osworld.py debug

    # Eval mode (test_small, 3 parallel workers)
    uv run recipes/osworld/eval_azure_osworld.py
"""

import logging
import os
import sys

from cube_infra_azure import AzureInfraConfig
from dotenv import load_dotenv
from osworld_cube.benchmark import OSWorldBenchmarkConfig
from osworld_cube.computer import ComputerConfig

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
    image_name_suffix="-generalized",
)

OSWORLD_SYSTEM_PROMPT_PYAUTOGUI_AXTREE = """\
You are a desktop automation agent controlling a real Ubuntu computer.

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
    # Resume support: pass `OSWORLD_RESUME_DIR=/path/to/prior/eval/output` to
    # pick up where a previous run left off (re-runs anything not COMPLETED /
    # MAX_STEPS_REACHED). Otherwise creates a fresh timestamped output dir.
    resume_from = os.environ.get("OSWORLD_RESUME_DIR")
    if resume_from:
        from pathlib import Path
        output_dir = Path(resume_from)
        print(f"RESUMING from existing output dir: {output_dir}")
    else:
        output_dir = make_experiment_output_dir("genny_azure", "osworld-cube")

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=OSWORLD_SYSTEM_PROMPT_PYAUTOGUI_AXTREE,
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

    benchmark_config = OSWorldBenchmarkConfig(
        tool_config=tool_config,
        use_som=False,
    )
    OSWorldBenchmarkConfig.install()
    # Subset selectable via env var so we can iterate fast on test_small (39)
    # before scaling to test_nogdrive (360). Default to test_small. We always
    # filter out gdrive-dependent tasks (test_small ∩ test_nogdrive) since
    # those need credentials we don't reliably have wired up.
    subset = os.environ.get("OSWORLD_SUBSET", "test_small")
    benchmark_config = benchmark_config.named_subset(subset)
    if subset in ("test_small", "test_all"):
        nogdrive_ids = {
            tid for tid, tm in OSWorldBenchmarkConfig.task_metadata.items()
            if "test_nogdrive" in (getattr(tm, "test_sets", None) or [])
        }
        kept = [tid for tid in benchmark_config.task_ids or list(benchmark_config.task_metadata)
                if tid in nogdrive_ids]
        n_dropped = len(benchmark_config.task_ids or benchmark_config.task_metadata) - len(kept)
        benchmark_config.task_ids = kept
        if n_dropped:
            logging.info("Filtered out %d gdrive task(s) from %s", n_dropped, subset)
    logging.info("OSWorld eval: subset=%s, %d tasks", subset, len(benchmark_config.task_ids or benchmark_config.task_metadata))

    print("--- pre-run cleanup ---")
    pre_deleted = INFRA.cleanup_orphaned_resources()
    if pre_deleted:
        n = sum(len(v) for v in pre_deleted.values())
        print(f"Cleaned up {n} orphaned resource(s) from prior runs")

    # Pre-warm the Azure CLI token cache on disk before Ray workers spawn.
    # Each Ray worker is a fresh Python process that calls `az account
    # get-access-token` on first auth — at n_cpus=100+ that subprocess storm
    # collides on `~/.azure/msal_token_cache.bin` and ~all workers fail with
    # CredentialUnavailableError. By running `az` once here, the cache is
    # populated; worker `az` calls then read the warm cache and return fast
    # without hitting login.microsoftonline.com.
    print("--- pre-warm Azure CLI token cache ---")
    from cube_infra_azure.azure import _get_cached_cred
    cred = _get_cached_cred()
    tok = cred.get_token("https://management.azure.com/.default")
    print(f"Pre-warmed token, expires in {(tok.expires_on - __import__('time').time())/60:.1f}min")

    exp = Experiment(
        name="osworld_azure_gpt5_mini",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=INFRA,
        max_steps=100,
        resume=bool(resume_from),
    )

    try:
        if debug:
            print("\n" + "=" * 60)
            print("DEBUG MODE: Running test_small sequentially on Azure")
            print("=" * 60)
            print(f"Output directory: {output_dir}")
            print(f"Model: {llm_config.model_name}")
            print(f"Infra: {INFRA.fingerprint()}")
            print("=" * 60 + "\n")
            run_sequentially(exp)
        else:
            print("\n" + "=" * 60)
            print("EVAL MODE: Running OSWorld TEST_SMALL on Azure")
            print("=" * 60)
            print(f"Output directory: {output_dir}")
            print(f"Model: {llm_config.model_name}")
            print(f"Infra: {INFRA.fingerprint()}")
            print("Parallelism: n_cpus=50")
            print("=" * 60 + "\n")
            run_with_ray(exp, n_cpus=50)
    finally:
        # Sweep any VMs orphaned by Ray force-kills or worker crashes.
        # Normal completions are already cleaned up by task.close() in episode.py.
        deleted = INFRA.cleanup_orphaned_resources()
        if deleted:
            print(f"Cleaned up {len(deleted)} orphaned VM(s): {deleted}")


if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
    main(debug)
