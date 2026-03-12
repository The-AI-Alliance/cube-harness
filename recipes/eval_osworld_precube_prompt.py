"""OSWorld Eval — Pre-cube system prompt on tasks that regressed after CUBE integration.

This recipe tests the 6 tasks that passed in the pre-cube run but now fail in post-cube
runs. The hypothesis is that the regression is caused by the weaker system prompt adopted
during CUBE integration, not by the CUBE wrapper itself.

Pre-cube prompt improvements over the current post-cube prompt:
  - Explicit coordinate calculation formula (center_x = x + w//2)
  - Curated pyautogui command reference with reliable primitives
  - Explicit fail() / done() termination semantics
  - Anti-loop rule: "if an action has no effect after 2 attempts, try a different approach"

Regressed tasks (pass pre-cube, fail post-cube):
  0810415c  libreoffice_writer  — line spacing task
  0ed39f63  vs_code             — VS Code task
  357ef137  libreoffice_calc    — calc cell navigation
  5ea617a3  os                  — OS-level task
  716a6079  multi_apps          — multi-app coordination
  7b6c7e24  chrome              — Chrome task

Reference runs:
  pre-cube (PASS):  agentlab_results/al2/osworld_genny_gpt5_20260306_192921
  post-cube (FAIL): agentlab2_results/20260305_224521_genny_osworld-cube
                    agentlab2_results/20260306_235115_genny_osworld-cube

Usage:
    # Run all 6 regressed tasks sequentially
    uv run recipes/eval_osworld_precube_prompt.py

    # Debug mode: run just 2 tasks
    uv run recipes/eval_osworld_precube_prompt.py debug
"""

import sys

from osworld_cube.benchmark import OSWorldBenchmark, OSWorldTestSet
from osworld_cube.computer import ComputerConfig

from agentlab2 import make_experiment_output_dir
from agentlab2.agents.genny import GennyConfig
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig

# ---------------------------------------------------------------------------
# Pre-cube system prompt — the one used in osworld_genny_gpt5_20260306_192921
# that achieved 30.6% vs 16.7% / 11.4% for post-cube runs.
# ---------------------------------------------------------------------------

PRECUBE_SYSTEM_PROMPT = """\
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

# ---------------------------------------------------------------------------
# The 6 tasks that regressed: pass in pre-cube, fail in both post-cube runs.
# ---------------------------------------------------------------------------

REGRESSED_TASK_IDS = [
    "0810415c-bde4-4443-9047-d5f70165a697",  # libreoffice_writer
    "0ed39f63-6049-43d4-ba4d-5fa2fe04a951",  # vs_code
    "357ef137-7eeb-4c80-a3bb-0951f26a8aff",  # libreoffice_calc
    "5ea617a3-0e86-4ba6-aab2-dac9aa2e8d57",  # os
    "716a6079-22da-47f1-ba73-c9d58f986a38",  # multi_apps
    "7b6c7e24-c58a-49fc-a5bb-d57b80e5b4c3",  # chrome
]


def main(debug: bool) -> None:
    output_dir = make_experiment_output_dir("genny_precube_prompt", "osworld-cube")

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=PRECUBE_SYSTEM_PROMPT,
        max_actions=15,
        render_last_n_obs=1,
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

    benchmark = OSWorldBenchmark(
        default_tool_config=tool_config,
        use_som=False,
        test_set_name=OSWorldTestSet.TEST_SMALL,
    )
    benchmark.setup()
    benchmark = benchmark.subset_from_list(REGRESSED_TASK_IDS)

    exp = Experiment(
        name="genny_precube_prompt_regressed",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=15,
    )

    print("\n" + "=" * 60)
    print(f"{'DEBUG' if debug else 'EVAL'} MODE: Pre-cube prompt on regressed tasks")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Model: {llm_config.model_name}")
    print(f"Tasks: {REGRESSED_TASK_IDS}")
    print("=" * 60 + "\n")

    if debug:
        run_sequentially(exp, debug_limit=2)
    else:
        run_with_ray(exp, n_cpus=4)


if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
    main(debug)
