"""OSWorld Eval — Genny agent with GPT-5 and accessibility tree observations.

Uses the Genny agent (explicit context management, rolling summaries) with
the linearized accessibility tree for element coordinates, without screenshots
or Set-of-Marks scaffolding.

Prerequisites:
    OSWorld repo cloned to ~/.agentlab2/OSWorld/
    (auto-cloned on first run if missing)

Usage:
    # Debug mode (2 tasks, sequential)
    uv run recipes/eval_osworld.py debug

    # Eval mode (all tasks, 3 workers)
    uv run recipes/eval_osworld.py

    # Custom task subset via tasks_file
    TASKS_FILE=/path/to/tasks.json uv run recipes/eval_osworld.py
"""

import sys

from agentlab2 import make_experiment_output_dir
from agentlab2.agents.genny import GennyConfig
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from osworld_cube.benchmark import OSWorldBenchmark, OSWorldTestSet
from osworld_cube.computer import ComputerConfig

OSWORLD_SYSTEM_PROMPT_PYAUTOGUI_AXTREE = """\
You are an expert computer use agent. You control a desktop computer via pyautogui.
You receive the accessibility tree of the current screen as a tab-separated table \
with columns: tag, name, text, class, description, position (top-left x&y), size (w&h).
Use it to identify interactive elements and their coordinates, then write Python code
using pyautogui to interact with them.

Think step by step: what does the current screen show, what is the next action toward
the goal, and which element should you interact with. Then call run_pyautogui with
the appropriate code."""


def main(debug: bool) -> None:
    output_dir = make_experiment_output_dir("genny", "osworld-cube")

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=OSWORLD_SYSTEM_PROMPT_PYAUTOGUI_AXTREE,
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

    exp = Experiment(
        name="osworld_genny_gpt5",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=15,
    )

    if debug:
        print("\n" + "=" * 60)
        print("DEBUG MODE: Running 2 tasks sequentially")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Model: {llm_config.model_name}")
        print(f"Provider: {tool_config.provider}")
        print("=" * 60 + "\n")
        run_sequentially(exp, debug_limit=2)
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
