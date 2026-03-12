"""OSWorld Eval — Claude Haiku with screenshot + axtree observations, rolling 3-step context, 100 actions, computer_13 action space.

Uses the Genny agent with Claude Haiku (claude-haiku-4-5-20251001) as a multimodal agent.
Observations include a screenshot and a linearized accessibility tree element table.
Keeps the last 3 observations in context, giving the agent more history to reason from.
Allows up to 100 actions per episode.

Unlike eval_osworld_haiku_3obs_100actions.py (which uses pyautogui code execution),
this recipe uses the computer_13 action space: 13 discrete mouse/keyboard primitives
(click, double_click, right_click, typing, press, hotkey, etc.) instead of raw Python.

Prerequisites:
    OSWorld repo cloned to ~/.agentlab2/OSWorld/
    (auto-cloned on first run if missing)

Usage:
    # Debug mode (debug_tasks.json, sequential)
    uv run recipes/eval_osworld_haiku_3obs_100actions_computer13.py debug

    # Eval mode (test_small without gdrive, 3 workers)
    uv run recipes/eval_osworld_haiku_3obs_100actions_computer13.py
"""

import sys
from datetime import datetime
from pathlib import Path

import osworld_cube
from osworld_cube.benchmark import OSWorldBenchmark, OSWorldTestSet
from osworld_cube.computer import ActionSpace, ComputerConfig

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
You control the computer through 13 discrete mouse/keyboard primitives:

### Mouse
- click(x, y)                     — left-click at coordinates
- click(button="right", x, y)     — right-click at coordinates
- double_click(x, y)              — double-click at coordinates
- mouse_down() / mouse_up()       — press/release the left mouse button
- move_to(x, y)                   — move cursor without clicking
- drag_to(x, y)                   — click-drag from current position to (x, y)
- scroll(dx, dy)                  — scroll; dy > 0 scrolls down, dy < 0 scrolls up

### Keyboard
- typing(text)                    — type a string into the focused element
- press(key)                      — press one key: "enter", "esc", "tab", "backspace", "space", etc.
- key_down(key) / key_up(key)     — hold/release a modifier: "ctrl", "shift", "alt"
- hotkey(keys)                    — simultaneous combo, joined by '+': "ctrl+c", "ctrl+shift+t"

### Task signals
- wait()   — wait one step without acting
- done()   — call when the task is SUCCESSFULLY complete
- fail()   — call if the task is INFEASIBLE or impossible

## Strategy
1. Study the element table carefully; calculate center_x = x + w//2, center_y = y + h//2
2. If an unexpected dialog or popup is blocking your task, dismiss it before proceeding
3. If the task is clearly impossible (missing app, contradictory requirements), call fail() immediately
4. Prefer hotkey shortcuts over multi-step mouse navigation when practical
5. When viewing a web page, zoom out (hotkey("ctrl+-")) if content seems cut off
6. When a terminal command produces large output, redirect to a file and read selectively:
   `command > /tmp/out.txt` then `grep` or `head`/`tail` the file
7. Do NOT ask for clarification — always proceed with available information
8. After completing the task, verify the result in the next observation, then call done()
9. Do not loop — if an action has no visible effect after 2 attempts, try a completely different approach\
"""


def main(debug: bool) -> None:
    today = datetime.today().strftime("%A, %B %d, %Y")
    system_prompt = HAIKU_SYSTEM_PROMPT.format(today=today)

    output_dir = make_experiment_output_dir("genny_haiku_3obs_100actions_computer13", "osworld-cube")

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
        provider="docker",
        action_space=ActionSpace.COMPUTER_13,
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
        name="osworld_genny_haiku_3obs_100actions_computer13",
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
        print(f"Provider: {tool_config.provider}")
        print(f"Action space: {tool_config.action_space}")
        print("=" * 60 + "\n")
        run_sequentially(exp)
    else:
        print("\n" + "=" * 60)
        print("EVAL MODE: Running OSWorld tasks with Ray")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Model: {llm_config.model_name}")
        print(f"Provider: {tool_config.provider}")
        print(f"Action space: {tool_config.action_space}")
        print("Parallelism: 3 workers")
        print("=" * 60 + "\n")
        run_with_ray(exp, n_cpus=3)


if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
    main(debug)
