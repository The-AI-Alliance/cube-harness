import sys

from miniwob_cube.benchmark import MiniWobBenchmark

from cube_harness import make_experiment_output_dir
from cube_harness.agents.react import ReactAgentConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.tools.playwright import PlaywrightConfig


def main(debug: bool) -> None:
    output_dir = make_experiment_output_dir("react", "miniwob")

    llm_config = LLMConfig(model_name="gpt-4.5-mini", temperature=1.0)
    agent_config = ReactAgentConfig(llm_config=llm_config)

    tool_config = PlaywrightConfig(use_screenshot=True, headless=True)
    benchmark = MiniWobBenchmark(default_tool_config=tool_config)

    exp = Experiment(
        name="miniwob",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=10,
    )

    if debug:
        run_sequentially(exp, debug_limit=2)
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
