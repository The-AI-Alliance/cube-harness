"""WAA Eval — Genny agent with GPT-5 and accessibility tree observations.

Uses the Genny agent (explicit context management, rolling summaries) with
the linearized accessibility tree for element coordinates, without screenshots
or Set-of-Marks scaffolding.

Prerequisites:
    Windows 11 Enterprise Evaluation ISO (see waa-cube README)
    WAA_SETUP_ISO=/path/to/Win11_Eval.iso  (or set in cubes/windows-agent-arena-cube/.env)

Usage:
    # Debug mode (1 task, sequential)
    uv run recipes/waa/eval_waa.py debug

    # Eval mode (all tasks, 3 workers)
    uv run recipes/waa/eval_waa.py
"""

import sys
from pathlib import Path

import waa_cube
from waa_cube.benchmark import WAABenchmark
from waa_cube.computer import ComputerConfig
from waa_cube.vm_backend.backend import WAADockerVMBackend

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import GennyConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig

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
    output_dir = make_experiment_output_dir("genny", "waa-cube")

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=WAA_SYSTEM_PROMPT,
        max_actions=15,
        render_last_n_obs=1,
        enable_summarize=False,
        tools_as_text=False,
    )

    tool_config = ComputerConfig(
        action_space="pyautogui",
        require_a11y_tree=True,
        observe_after_action=True,
    )

    # tasks_file = str(Path(waa_cube.__file__).parent / "debug_tasks.json") if debug else None
    benchmark = WAABenchmark(
        default_tool_config=tool_config,
        vm_backend=WAADockerVMBackend(
            cpu_cores=4,  # Reduced for 16-core machine with 2 parallel workers
        ),
        # tasks_file=tasks_file,
        test_set_name="test_small.json",
    )
    benchmark.install()
    benchmark.setup()

    # Exclude chrome/msedge tasks: Chrome DevTools (CDP) setup has a timing
    # issue — Playwright gets a 502 connecting to the CDP port before Chrome
    # is fully ready.  Until the setup_controller adds a CDP-readiness retry,
    # skip these domains to avoid false failures.
    keep_ids = [
        tid
        for tid, meta in benchmark.task_metadata.items()
        if meta.extra_info.get("domain") not in ("chrome", "msedge")
    ]
    benchmark = benchmark.subset_from_list(keep_ids)

    exp = Experiment(
        name="waa_genny_gpt5",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=15,
    )

    if debug:
        print("\n" + "=" * 60)
        print("DEBUG MODE: Running debug task sequentially")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Model: {llm_config.model_name}")
        print("=" * 60 + "\n")
        run_sequentially(exp)
    else:
        print("\n" + "=" * 60)
        print("EVAL MODE: Running WAA tasks with Ray")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Model: {llm_config.model_name}")
        print("Parallelism: 2 workers")
        print("=" * 60 + "\n")
        run_with_ray(exp, n_cpus=2)


if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
    main(debug)
