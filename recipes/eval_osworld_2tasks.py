"""OSWorld 2-task eval - runs the first 2 tasks from SampledOSWorldBenchmark (seed=42).

Identical to eval_osworld.py but caps task list to 2 tasks for quick smoke testing.
The 2 tasks are deterministic: same seed → same shuffle → first 2.

Usage:
    uv run recipes/eval_osworld_2tasks.py
"""

import os
import random
import time
from collections import defaultdict
from pathlib import Path

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.osworld.benchmark import OSWORLD_SYSTEM_PROMPT_COMPUTER_13, OSWorldBenchmark
from agentlab2.benchmarks.osworld.task import OSWorldTask
from agentlab2.exp_runner import run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from agentlab2.tools.computer import ComputerConfig


class SampledOSWorldBenchmark(OSWorldBenchmark):
    """OSWorldBenchmark that samples a fixed fraction of tasks per domain."""

    sample_fraction: float = 0.10

    def load_tasks(self) -> list[OSWorldTask]:
        all_tasks = super().load_tasks()
        all_tasks = [t for t in all_tasks if t.domain != "chrome"]

        by_domain: dict[str, list[OSWorldTask]] = defaultdict(list)
        for task in all_tasks:
            by_domain[task.domain].append(task)

        sampled: list[OSWorldTask] = []
        rng = random.Random(self.shuffle_seed)
        for domain, tasks in sorted(by_domain.items()):
            k = max(1, round(len(tasks) * self.sample_fraction))
            chosen = rng.sample(tasks, min(k, len(tasks)))
            sampled.extend(chosen)

        return sampled[:2]


def main() -> None:
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir_base = os.environ.get("OUTPUT_DIR", str(Path.home() / "agentlab_results" / "al2"))
    output_dir = Path(output_dir_base) / f"osworld_2tasks_{current_datetime}"

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = ReactAgentConfig(
        llm_config=llm_config,
        max_actions=15,
        system_prompt=OSWORLD_SYSTEM_PROMPT_COMPUTER_13,
    )

    tool_config = ComputerConfig(
        provider="docker",
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
        max_turns=15,
        sample_fraction=0.10,
        use_som=True,  # Annotate screenshots with numbered bounding boxes (Set-of-Marks)
    )

    exp = Experiment(
        name="osworld_2tasks",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    print("\n" + "=" * 60)
    print("Running 2 OSWorld tasks in parallel")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("=" * 60 + "\n")
    run_with_ray(exp, n_cpus=2)


if __name__ == "__main__":
    main()
