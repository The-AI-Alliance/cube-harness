"""OSWorld full evaluation on Azure — Genny agent with pyautogui + AXTree observations.

Usage:
    uv run meta_agent/recipes/osworld_azure_full.py                     # test_small, 3 workers
    uv run meta_agent/recipes/osworld_azure_full.py hints               # + task-specific hints
    uv run meta_agent/recipes/osworld_azure_full.py debug               # sequential, debug tasks
    uv run meta_agent/recipes/osworld_azure_full.py retry /path/to/exp  # retry crashed episodes

Prerequisites:
    See cube-resources/cube-infra-azure/README.md for full setup instructions.
"""

import logging
import os
import sys
from pathlib import Path

from cube_infra_azure import AzureInfraConfig
from dotenv import load_dotenv
from osworld_cube.benchmark import OSWorldBenchmark
from osworld_cube.computer import ComputerConfig

# meta_agent/ is not a Python package — add it to sys.path so we can import osworld_hints.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env so credentials are available even when the shell didn't source ~/.zshrc.
# Ray workers inherit the parent process env, so this must run before ray.init().
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

from osworld_hints import OSWORLD_TASK_HINTS, OSWORLD_TASK_PRECISION

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import GennyConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig

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

OSWORLD_SYSTEM_PROMPT = """\
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


def run(
    debug: bool,
    use_hints: bool,
    retry_dir: Path | None,
) -> None:
    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)

    tool_config = ComputerConfig(
        action_space="pyautogui",
        require_a11y_tree=True,
        observe_after_action=True,
    )

    benchmark = OSWorldBenchmark(
        default_tool_config=tool_config,
        use_som=False,
        infra=INFRA,
    )
    benchmark.install()
    benchmark.setup()

    benchmark = benchmark.named_subset("test_small")

    suffix = "hints" if use_hints else "nohints"

    if retry_dir is not None:
        output_dir = retry_dir
        retry_failed = True
        resume = True
        agent_config = None  # reloaded from persisted EpisodeConfig
    else:
        output_dir = make_experiment_output_dir("genny_azure", f"osworld-{suffix}")
        retry_failed = False
        resume = False
        agent_config = GennyConfig(
            llm_config=llm_config,
            system_prompt=OSWORLD_SYSTEM_PROMPT,
            max_actions=100,
            render_last_n_obs=3,
            enable_summarize=False,
            tools_as_text=False,
            task_hints=OSWORLD_TASK_HINTS if use_hints else {},
            task_precision=OSWORLD_TASK_PRECISION if use_hints else {},
        )

    exp = Experiment(
        name=f"osworld-azure-{suffix}",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=15,
        retry_failed=retry_failed,
        resume=resume,
    )

    label = f"RETRY {retry_dir}" if retry_dir else ("WITH hints" if use_hints else "NO hints")
    mode = "DEBUG sequential (2 tasks)" if debug else "EVAL 3 workers"
    print(f"\n{'=' * 60}")
    print(f"OSWorld Azure | {label} | {mode}")
    print(f"Output: {output_dir}")
    print(f"Model:  {llm_config.model_name}")
    print(f"Infra:  {INFRA.fingerprint()}")
    print("=" * 60 + "\n")

    try:
        if debug:
            run_sequentially(exp, debug_limit=2)
        else:
            run_with_ray(exp, n_cpus=40)
    finally:
        deleted = INFRA.cleanup_orphaned_resources()
        if deleted:
            print(f"Cleaned up {len(deleted)} orphaned VM(s): {deleted}")


def main() -> None:
    args = sys.argv[1:]
    args_set = set(args)

    debug = "debug" in args_set
    use_hints = "hints" in args_set

    retry_dir: Path | None = None
    if "retry" in args_set:
        retry_idx = args.index("retry")
        if retry_idx + 1 < len(args):
            retry_dir = Path(args[retry_idx + 1])
        else:
            print("ERROR: 'retry' flag requires a path argument", file=sys.stderr)
            sys.exit(1)

    run(debug=debug, use_hints=use_hints, retry_dir=retry_dir)


if __name__ == "__main__":
    main()
