# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "browsercomp-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# browsercomp-cube = { path = "../cubes/browsercomp", editable = true }
# ///

import argparse

from browsercomp_cube.benchmark import BrowseCompBenchmark
from browsercomp_cube.tool import SubmitAnswerToolConfig
from cube.tool import ToolboxConfig
from cube_web_tool import BraveWebSearchToolConfig, WebFetchToolConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.react import ReactAgentConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig


def main(debug: bool) -> None:
    output_dir = make_experiment_output_dir("react", "browsercomp")

    llm_config = LLMConfig(model_name="gpt-5.4-mini", temperature=1.0)
    agent_config = ReactAgentConfig(llm_config=llm_config)

    tool_config = ToolboxConfig(
        tool_configs=[
            BraveWebSearchToolConfig(),
            WebFetchToolConfig(query_llm_model="gpt-5.4-mini"),
            SubmitAnswerToolConfig(),
        ]
    )
    benchmark = BrowseCompBenchmark(default_tool_config=tool_config, scorer_model="gpt-5.4-mini")
    benchmark.install()

    exp = Experiment(
        name="browsercomp",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=50,
    )

    if debug:
        run_sequentially(exp, debug_limit=2)
    else:
        run_with_ray(
            exp,
            n_cpus=8,
            otlp_endpoint="http://localhost:4318/v1/traces",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.debug)
