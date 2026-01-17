import sys
import time
from pathlib import Path

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.mind2web.benchmark import Mind2WebBenchmark
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from agentlab2.tools.playwright import PlaywrightConfig


def main(debug: bool) -> None:
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path.home() / "agentlab_results" / "al2" / f"mind2web_{current_datetime}"

    llm_config = LLMConfig(model_name="openai/gpt-5-nano", temperature=1.0)
    agent_config = ReactAgentConfig(llm_config=llm_config)

    tool_config = PlaywrightConfig(use_screenshot=False, use_html=True, headless=True)
    benchmark = Mind2WebBenchmark(
        tool_config=tool_config,
        split="train",
        max_tasks=1 if debug else None,
        shuffle=True,
    )

    exp = Experiment(
        name="mind2web",
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
