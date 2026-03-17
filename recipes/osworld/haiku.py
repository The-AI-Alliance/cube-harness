# /// script
# dependencies = [
#   "osworld-cube",
#   "cube-harness",
# ]
# ///
"""OSWorld Eval — Claude Haiku with screenshot + axtree observations, rolling 3-step context, 100 actions.

Uses the Genny agent with Claude Haiku (claude-haiku-4-5-20251001) as a multimodal agent.
Observations include a screenshot and a linearized accessibility tree element table.
Unlike eval_osworld_haiku.py (which keeps only the last observation), this recipe keeps
the last 3 observations in context, giving the agent more history to reason from.
Unlike eval_osworld_haiku_3obs.py, this recipe allows up to 100 actions per episode.

v2: Updated system prompt — adds sudo password, environment context, and tips from
the OSWorld reference prompt (home dir, curl vs wget, zoom, large-output handling, date).

Prerequisites:
    OSWorld repo cloned to ~/.agentlab2/OSWorld/
    (auto-cloned on first run if missing)

Usage:
    # Debug mode (debug_tasks.json, sequential)
    uv run recipes/osworld/haiku.py debug

    # Eval mode (test_small without gdrive, 3 workers)
    uv run recipes/osworld/haiku.py
"""

import sys
from datetime import datetime
from pathlib import Path

import osworld_cube
from osworld_cube.benchmark import OSWorldBenchmark, OSWorldTestSet
from osworld_cube.computer import ComputerConfig
from osworld_cube.vm_backend import LocalQEMUVMBackend

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import GennyConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig

GDRIVE_TASK_IDS = {
    "4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc",
    "897e3b53-5d4d-444b-85cb-2cdc8a97d903",
    "46407397-a7d5-4c6b-92c6-dbe038b1457b",
}

HAIKU_SYSTEM_PROMPT = """\
You are a desktop automation agent controlling a real Ubuntu (x86_64) computer with internet access.

## Environment
- OS: Ubuntu, home directory is `/home/user`
- Browser: Google Chrome — click the Chrome icon to open it
- For sudo commands, the password is `password`
- Use `curl` instead of `wget` for downloads
- Today's date: {today}

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

### Modifier-key clicks (correct pattern)
- pyautogui.keyDown('shift'); pyautogui.click(x, y); pyautogui.keyUp('shift')

### Ending the task
- Call fail() if the task CANNOT be completed (infeasible tasks)
- Call done() when the task is successfully COMPLETE

## Strategy
1. Study the element table carefully to find the element you need to interact with
2. Calculate center coordinates: center_x = x + w//2, center_y = y + h//2
3. If an unexpected dialog or popup is blocking your task, dismiss it before proceeding
4. If the task is clearly impossible (missing app, contradictory requirements), call fail() immediately
5. Prefer hotkey shortcuts over mouse clicks when practical
6. When viewing a web page, zoom out (pyautogui.hotkey('ctrl', '-')) if content seems cut off
7. When a terminal command produces large output, redirect to a file:
   pyautogui.typewrite('command > /tmp/out.txt', interval=0.02)
   then read it with grep/head/tail
8. Do NOT ask for clarification — always proceed with available information
9. After completing the task, verify by checking the next observation then call done()
10. Do not loop — if an action has no effect after 2 attempts, try a completely different approach\
"""


def main(debug: bool) -> None:
    today = datetime.today().strftime("%A, %B %d, %Y")
    system_prompt = HAIKU_SYSTEM_PROMPT.format(today=today)

    output_dir = make_experiment_output_dir("genny_haiku_3obs_100actions_v2", "osworld-cube")

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
        observe_after_action=True,
    )

    tasks_file = str(Path(osworld_cube.__file__).parent / "debug_tasks.json") if debug else None
    benchmark = OSWorldBenchmark(
        default_tool_config=tool_config,
        use_som=False,
        tasks_file=tasks_file,
        test_set_name=OSWorldTestSet.TEST_SMALL,
        vm_backend=LocalQEMUVMBackend(),
    )
    benchmark.setup()
    keep_ids = [tid for tid in benchmark.task_metadata if tid not in GDRIVE_TASK_IDS]
    benchmark = benchmark.subset_from_list(keep_ids)

    exp = Experiment(
        name="osworld_genny_haiku_3obs_100actions_v2",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=100,
    )

    if debug:
        print("\n" + "=" * 60)
        print("DEBUG MODE: Running debug_tasks.json sequentially")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Model: {llm_config.model_name}")
        print("=" * 60 + "\n")
        run_sequentially(exp)
    else:
        print("\n" + "=" * 60)
        print("EVAL MODE: Running OSWorld tasks with Ray")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Model: {llm_config.model_name}")
        print("Parallelism: 3 workers")
        print("=" * 60 + "\n")
        run_with_ray(exp, n_cpus=3)


if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
    main(debug)
