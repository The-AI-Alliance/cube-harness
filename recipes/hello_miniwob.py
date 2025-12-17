import sys
import time
from pathlib import Path

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.miniwob.benchmark import MiniWobBenchmark
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from agentlab2.tools.playwright import PlaywrightConfig


def main(debug: bool):
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path.home() / "agentlab_results" / "al2" / f"miniwob_{current_datetime}"

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = ReactAgentConfig(llm_config=llm_config)

    tool_config = PlaywrightConfig(use_screenshot=True, headless=True)
    benchmark = MiniWobBenchmark(tool_config=tool_config)

    exp = Experiment(
        name="miniwob",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    if debug:
        run_sequentially(exp, debug_limit=2)
    else:
        run_with_ray(exp, n_cpus=4)


if __name__ == "__main__":
    debug = sys.argv[-1] == "debug"
    main(debug)
