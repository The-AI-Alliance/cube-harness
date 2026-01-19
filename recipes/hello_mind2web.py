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
    output_dir = f"./traces/mind2web_{current_datetime}"
    # TODO: next steps
    # - print_stats should consider more details, such as binary/partial scorings
    # - is gpt-5-nano too bad? or why is it calling selector's that do not exist
    # - element matching is ignore backend_node_id and our matching is too simplistic comparing to theirs.

    llm_config = LLMConfig(model_name="openai/gpt-5-nano", temperature=1.0)
    agent_config = ReactAgentConfig(llm_config=llm_config, stateless=True)

    tool_config = PlaywrightConfig(use_screenshot=False, use_html=True, headless=True)
    max_tasks = 10 if debug else None
    benchmark = Mind2WebBenchmark(
        tool_config=tool_config,
        split="train",
        max_tasks=max_tasks,
        shuffle=True,
    )

    exp = Experiment(
        name="mind2web",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    if debug:
        run_sequentially(exp, debug_limit=max_tasks)
    else:
        run_with_ray(exp, n_cpus=4)


if __name__ == "__main__":
    debug = sys.argv[-1] == "debug"
    main(debug)
