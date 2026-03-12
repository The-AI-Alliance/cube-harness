"""OSWorld Eval — Claude Haiku with screenshot + axtree observations, rolling 3-step context.

Uses the Genny agent with Claude Haiku (claude-haiku-4-5-20251001) as a multimodal agent.
Observations include a screenshot and a linearized accessibility tree element table.
Unlike eval_osworld_haiku.py (which keeps only the last observation), this recipe keeps
the last 3 observations in context, giving the agent more history to reason from.

Prerequisites:
    OSWorld repo cloned to ~/.agentlab2/OSWorld/
    (auto-cloned on first run if missing)

Usage:
    # Debug mode (debug_tasks.json, sequential)
    uv run recipes/eval_osworld_haiku_3obs.py debug

    # Eval mode (test_small without gdrive, 3 workers)
    uv run recipes/eval_osworld_haiku_3obs.py
"""

import sys
from pathlib import Path

import osworld_cube
from osworld_cube.benchmark import OSWorldBenchmark, OSWorldTestSet
from osworld_cube.computer import ComputerConfig

from agentlab2 import make_experiment_output_dir
from agentlab2.agents.genny import GennyConfig
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig

GDRIVE_TASK_IDS = {
    "4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc",
    "897e3b53-5d4d-444b-85cb-2cdc8a97d903",
    "46407397-a7d5-4c6b-92c6-dbe038b1457b",
}

HAIKU_SYSTEM_PROMPT = """\
You are a desktop automation agent controlling a real Ubuntu computer.

## Observations
Each step you receive:
1. A screenshot of the current screen (1920×1080)
2. An element table listing interactive UI elements with columns:
   index, tag, name, text, x, y, w, h

Where (x, y) is the top-left corner and (w, h) is the size of each element.
To click the center of element at row i: center_x = x + w//2, center_y = y + h//2

Prefer the element table for precise coordinates; use the screenshot for visual context.
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
- pyautogui.scroll(x, y, clicks=-3)           — scroll (negative = down)
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
    output_dir = make_experiment_output_dir("genny_haiku_3obs", "osworld-cube")

    llm_config = LLMConfig(model_name="claude-haiku-4-5-20251001", temperature=1.0)
    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=HAIKU_SYSTEM_PROMPT,
        max_actions=15,
        render_last_n_obs=3,
        enable_summarize=False,
        tools_as_text=False,
    )

    tool_config = ComputerConfig(
        provider="docker",
        action_space="pyautogui",
        headless=True,
        require_a11y_tree=True,
        screen_size=(1920, 1080),
        observe_after_action=True,
    )

    tasks_file = str(Path(osworld_cube.__file__).parent / "debug_tasks.json") if debug else None
    benchmark = OSWorldBenchmark(
        default_tool_config=tool_config,
        use_som=False,
        tasks_file=tasks_file,
        test_set_name=OSWorldTestSet.TEST_SMALL,
    )
    benchmark.setup()
    keep_ids = [tid for tid in benchmark.task_metadata if tid not in GDRIVE_TASK_IDS]
    benchmark = benchmark.subset_from_list(keep_ids)

    exp = Experiment(
        name="osworld_genny_haiku_3obs",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=15,
    )

    if debug:
        print("\n" + "=" * 60)
        print("DEBUG MODE: Running debug_tasks.json sequentially")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Model: {llm_config.model_name}")
        print(f"Provider: {tool_config.provider}")
        print("=" * 60 + "\n")
        run_sequentially(exp)
    else:
        print("\n" + "=" * 60)
        print("EVAL MODE: Running OSWorld tasks with Ray")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Model: {llm_config.model_name}")
        print(f"Provider: {tool_config.provider}")
        print("Parallelism: 3 workers")
        print("=" * 60 + "\n")
        run_with_ray(exp, n_cpus=3)


if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
    main(debug)
