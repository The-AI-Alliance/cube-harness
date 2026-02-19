import logging

from agentlab2 import make_experiment_output_dir
from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.miniwob.benchmark import MiniWobBenchmark
from agentlab2.exp_runner import run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from agentlab2.tools.playwright import PlaywrightConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d %(funcName)s() - %(message)s",
)


def main() -> None:
    llm_config = LLMConfig(model_name="gpt-4.1-nano", temperature=1.0)
    agent_config = ReactAgentConfig(llm_config=llm_config)

    tool_config = PlaywrightConfig(use_screenshot=True, headless=True, chromium_sandbox=False)
    benchmark = MiniWobBenchmark(tool_config=tool_config)

    output_dir = make_experiment_output_dir(
        agent_name="react", benchmark_name="hello_world_study", tag="metrics"
    )
    exp = Experiment(
        name="hello_world_study",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    results = run_with_ray(
        exp,
        n_cpus=1,
        trace_output=f"{exp.output_dir}/traces",
        otlp_endpoint="http://localhost:4318/v1/traces",
    )

    logging.info(f"Completed {len(results.trajectories)} trajectories")


if __name__ == "__main__":
    main()
