"""
Run MiniWoB benchmark using BrowserGym tool.

This recipe demonstrates using the BrowserGym tool wrapper instead of the
direct Playwright tool for running MiniWoB tasks.

Usage:
    uv run recipes/hello_miniwob_bgym.py        # Full run with Ray
    uv run recipes/hello_miniwob_bgym.py debug  # Debug mode (2 tasks, sequential)
"""

import sys
import time
from pathlib import Path

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.miniwob.benchmark import MiniWobBenchmark
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from agentlab2.tools.browsergym import BrowsergymConfig


def main(debug: bool) -> None:
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path.home() / "agentlab_results" / "al2" / f"miniwob_bgym_{current_datetime}"

    llm_config = LLMConfig(model_name="azure/gpt-5-mini")
    agent_config = ReactAgentConfig(llm_config=llm_config)

    # Use BrowserGym tool instead of Playwright
    tool_config = BrowsergymConfig(
        use_screenshot=True,
        use_html=True,
        use_axtree=False,
    )
    benchmark = MiniWobBenchmark(tool_config=tool_config)
    exp = Experiment(
        name="miniwob_bgym",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=10,
    )
    if debug:
        run_sequentially(exp, debug_limit=2)
    else:
        run_with_ray(exp, n_cpus=4)


if __name__ == "__main__":
    debug = sys.argv[-1] == "debug"
    main(debug)
