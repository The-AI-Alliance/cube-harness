import sys

from osworld_cube.benchmark import OSWorldBenchmark

from agentlab2 import make_experiment_output_dir
from agentlab2.agents.react import ReactAgentConfig
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig


def main(debug: bool) -> None:
    output_dir = make_experiment_output_dir("react", "osworld-cube")

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = ReactAgentConfig(llm_config=llm_config)

    benchmark = OSWorldBenchmark()

    exp = Experiment(
        name="osworld-cube",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=15,
    )

    if debug:
        run_sequentially(exp, debug_limit=2)
    else:
        run_with_ray(
            exp,
            n_cpus=4,
        )


if __name__ == "__main__":
    debug = sys.argv[-1] == "debug"
    main(debug)
