"""
Run MiniWoB benchmark using BrowserGym tool.

This recipe demonstrates using the BrowserGym tool wrapper instead of the
direct Playwright tool for running MiniWoB tasks.

Usage:
    uv run recipes/hello_miniwob_bgym.py        # Full run with Ray
    uv run recipes/hello_miniwob_bgym.py debug  # Debug mode (2 tasks, sequential)
"""

import sys

from miniwob_cube.benchmark import MiniWobBenchmark

from agentlab2 import make_experiment_output_dir
from agentlab2.agents.react import ReactAgentConfig
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from agentlab2.tools.browsergym import BrowsergymConfig


def main(debug: bool) -> None:
    output_dir = make_experiment_output_dir("react", "miniwob_browsergym")

    llm_config = LLMConfig(model_name="azure/gpt-5-mini")
    agent_config = ReactAgentConfig(llm_config=llm_config)

    benchmark = MiniWobBenchmark(
        default_tool_config=BrowsergymConfig(
            use_screenshot=True,
            use_html=True,
            use_axtree=False,
        )
    )
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
