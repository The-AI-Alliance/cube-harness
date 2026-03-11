"""OSWorld PyAutoGUI Eval - Runs a sampled fraction of OSWorld tasks using pyautogui + SoM.

Uses the PyAutoGUI action space with Set-of-Marks screenshot annotation.
The agent executes Python/pyautogui code and references tagged elements via tag_N variables.

Prerequisites:
    make install-osworld
    OSWorld repo cloned to ~/.agentlab2/benchmarks/osworld/OSWorld/
    (auto-cloned on first run if missing)

Usage:
    # Run eval (10% of tasks per domain, 5 workers)
    uv run recipes/eval_osworld_pyautogui.py

    # Debug mode (2 tasks sequential)
    uv run recipes/eval_osworld_pyautogui.py debug

    # Custom fraction
    SAMPLE_FRACTION=0.05 uv run recipes/eval_osworld_pyautogui.py
"""

import os
import random
import sys
import time
from collections import defaultdict

from agentlab2.agents.react import ReactAgentConfig
from osworld_cube import OSWorldBenchmark; from osworld_cube.benchmark import OSWORLD_SYSTEM_PROMPT_COMPUTER_13 as OSWORLD_SYSTEM_PROMPT_PYAUTOGUI
from osworld_cube import OSWorldTask
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from osworld_cube import OSWorldComputerConfig as ComputerConfig


class SampledOSWorldBenchmark(OSWorldBenchmark):
    """OSWorldBenchmark that samples a fixed fraction of tasks per domain."""

    sample_fraction: float = 0.10  # fraction of tasks to keep per domain

    def load_tasks(self) -> list[OSWorldTask]:
        all_tasks = super().load_tasks()
        all_tasks = [t for t in all_tasks]

        # Group by domain
        by_domain: dict[str, list[OSWorldTask]] = defaultdict(list)
        for task in all_tasks:
            by_domain[task.domain].append(task)

        sampled: list[OSWorldTask] = []
        rng = random.Random(self.shuffle_seed)
        for domain, tasks in sorted(by_domain.items()):
            k = max(1, round(len(tasks) * self.sample_fraction))
            chosen = rng.sample(tasks, min(k, len(tasks)))
            sampled.extend(chosen)
            print(f"  {domain}: {len(chosen)}/{len(tasks)} tasks sampled")

        print(f"Total sampled: {len(sampled)} tasks across {len(by_domain)} domains")
        return sampled


def main(debug: bool):
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir_base = os.environ.get("OUTPUT_DIR", str(__import__("pathlib").Path.home() / "agentlab_results" / "al2"))
    output_dir = __import__("pathlib").Path(output_dir_base) / f"osworld_eval_pyautogui_{current_datetime}"

    sample_fraction = float(os.environ.get("SAMPLE_FRACTION", "0.1"))

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = ReactAgentConfig(
        llm_config=llm_config,
        max_actions=15,
        system_prompt=OSWORLD_SYSTEM_PROMPT_PYAUTOGUI,
    )

    tool_config = ComputerConfig(
        provider="docker",
        action_space="pyautogui",
        headless=True,
        require_a11y_tree=True,
        screen_size=(1920, 1080),
        observe_after_action=True,
    )

    benchmark = SampledOSWorldBenchmark(
        tool_config=tool_config,
        domain="all",
        shuffle=True,
        shuffle_seed=42,
        use_som=True,
        max_turns=15,
        sample_fraction=sample_fraction,
    )

    exp = Experiment(
        name="osworld_eval_pyautogui",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    if debug:
        print("\n" + "=" * 60)
        print("DEBUG MODE: Running 2 tasks sequentially")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Provider: {tool_config.provider}")
        print(f"Sample fraction: {sample_fraction:.0%} per domain")
        print("=" * 60 + "\n")
        run_sequentially(exp, debug_limit=2)
    else:
        print("\n" + "=" * 60)
        print("EVAL MODE: Running sampled OSWorld tasks with Ray (pyautogui + SoM)")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Provider: {tool_config.provider}")
        print(f"Sample fraction: {sample_fraction:.0%} per domain")
        print(f"Parallelism: 3 workers")
        print("=" * 60 + "\n")
        run_with_ray(exp, n_cpus=3)


if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
    main(debug)
