import sys

from agentlab2 import make_experiment_output_dir
from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.miniwob.benchmark import MiniWobBenchmark
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from agentlab2.tools.playwright import PlaywrightConfig


def main(debug: bool):
    output_dir = make_experiment_output_dir("react", "miniwob")

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = ReactAgentConfig(llm_config=llm_config)

    tool_config = PlaywrightConfig(use_screenshot=True, headless=True)
    benchmark = MiniWobBenchmark(tool_config=tool_config, n_attempts=2, debug_task_limit=5 if debug else None)

    exp = Experiment(
        name="miniwob",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=10,
    )

    if debug:
        run_sequentially(exp)
    else:
        run_with_ray(
            exp,
            n_cpus=4,
            trace_output=f"{exp.output_dir}/traces",
            otlp_endpoint="http://localhost:4318/v1/traces",
        )


if __name__ == "__main__":
    debug = sys.argv[-1] == "debug"
    main(debug)
