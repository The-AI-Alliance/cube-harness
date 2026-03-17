import sys

from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
from webarena_verified.types.task import WebArenaSite
from webarena_verified_cube.benchmark import WebArenaVerifiedBenchmark
from webarena_verified_cube.tool import WebArenaToolConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.react import ReactAgentConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig


def main(debug: bool):
    output_dir = make_experiment_output_dir("react", "webarena-verified")

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = ReactAgentConfig(llm_config=llm_config)

    tool_config = WebArenaToolConfig()
    wav_config = WebArenaVerifiedConfig(environments={
        WebArenaSite.SHOPPING: EnvironmentConfig(urls=["http://localhost:7770"]),
        WebArenaSite.SHOPPING_ADMIN: EnvironmentConfig(urls=["http://localhost:7780"]),
        WebArenaSite.GITLAB: EnvironmentConfig(urls=["http://localhost:8023"]),
        WebArenaSite.REDDIT: EnvironmentConfig(urls=["http://localhost:9999"]),
        WebArenaSite.WIKIPEDIA: EnvironmentConfig(urls=["http://localhost:8888"]),
        WebArenaSite.MAP: EnvironmentConfig(urls=["http://localhost:3000"]),
    })
    benchmark = WebArenaVerifiedBenchmark(tool_config=tool_config, wav_config=wav_config)

    exp = Experiment(
        name="webarena-verified",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
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