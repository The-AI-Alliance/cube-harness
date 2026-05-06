# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "webarena-verified-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# webarena-verified-cube = { path = "../cubes/webarena-verified", editable = true }
# ///

import sys

from cube.infra_local import LocalInfraConfig
from cube.tool import ToolboxConfig
from cube_browser_playwright import PlaywrightSessionConfig
from webarena_verified_cube.benchmark import WebArenaVerifiedBenchmarkConfig
from webarena_verified_cube.resources import WEBARENA_ALL
from webarena_verified_cube.tool import HarPlaywrightConfig, SubmitResponseConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.react import ReactAgentConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig


def main(debug: bool) -> None:
    output_dir = make_experiment_output_dir("react", "webarena-verified")

    llm_config = LLMConfig(model_name="gpt-5-mini", temperature=1.0)
    agent_config = ReactAgentConfig(llm_config=llm_config)

    tool_config = ToolboxConfig(
        tool_configs=[
            HarPlaywrightConfig(browser=PlaywrightSessionConfig(headless=not debug)),
            SubmitResponseConfig(),
        ]
    )
    # Automatic mode: declare the DockerServiceConfig in resources= and let the runner
    # provision + launch on demand via Experiment.infra.
    # Swap LocalInfraConfig() for any other InfraConfig (e.g., AWSInfraConfig()) for cloud users.
    # Swap WEBARENA_ALL for any other entry in webarena_verified_cube.resources (e.g. WEBARENA_SHOPPING_ADMIN for a specific site).
    benchmark_config = WebArenaVerifiedBenchmarkConfig(
        tool_config=tool_config,
        resources=[WEBARENA_ALL],
    ).subset_from_glob("sites", "*shopping_admin*")

    exp = Experiment(
        name="webarena-verified",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=LocalInfraConfig(),
        max_steps=30,
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
