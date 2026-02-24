"""OSWorld Small Eval - Runs a sampled fraction of OSWorld tasks in parallel.

Loads tasks from the full OSWorld repo (not the debug tasks file) and samples
a configurable fraction of tasks from each domain. Runs with Ray for parallelism.

Prerequisites:
    make install-osworld
    OSWorld repo cloned to ~/.agentlab2/benchmarks/osworld/OSWorld/
    (auto-cloned on first run if missing)

Usage:
    # Run eval (10% of tasks per domain, 5 workers)
    uv run recipes/eval_osworld.py

    # Debug mode (2 tasks sequential)
    uv run recipes/eval_osworld.py debug

    # Custom fraction
    SAMPLE_FRACTION=0.05 uv run recipes/eval_osworld.py
"""

import os
import random
import sys
import time
from collections import defaultdict

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.osworld.benchmark import OSWorldBenchmark
from agentlab2.benchmarks.osworld.task import OSWorldTask
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from agentlab2.tools.computer import ComputerConfig


class SampledOSWorldBenchmark(OSWorldBenchmark):
    """OSWorldBenchmark that samples a fixed fraction of tasks per domain."""

    sample_fraction: float = 0.10  # fraction of tasks to keep per domain

    def load_tasks(self) -> list[OSWorldTask]:
        all_tasks = super().load_tasks()

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
    output_dir = __import__("pathlib").Path(output_dir_base) / f"osworld_eval_{current_datetime}"

    sample_fraction = float(os.environ.get("SAMPLE_FRACTION", "0.50"))

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = ReactAgentConfig(llm_config=llm_config)

    tool_config = ComputerConfig(
        provider="docker",
        headless=True,
        require_a11y_tree=True,
        screen_size=(1920, 1080),
        observe_after_action=False,
    )

    benchmark = SampledOSWorldBenchmark(
        tool_config=tool_config,
        domain="all",
        shuffle=True,
        shuffle_seed=42,
        max_turns=15,
        sample_fraction=sample_fraction,
    )

    exp = Experiment(
        name="osworld_eval",
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
        print("EVAL MODE: Running sampled OSWorld tasks with Ray")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Provider: {tool_config.provider}")
        print(f"Sample fraction: {sample_fraction:.0%} per domain")
        print(f"Parallelism: 5 workers")
        print("=" * 60 + "\n")
        run_with_ray(exp, n_cpus=5)


if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
    main(debug)
