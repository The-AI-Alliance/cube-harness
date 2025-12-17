import logging
import time
import uuid
from pathlib import Path

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.miniwob.benchmark import MiniWobBenchmark
from agentlab2.exp_runner import run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from agentlab2.metrics.tracer import AgentTracer
from agentlab2.tools.playwright import PlaywrightConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d %(funcName)s() - %(message)s",
)


def main() -> None:
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

    trace_output = f"{exp.output_dir}/metrics/{uuid.uuid4()}"
    tracer = AgentTracer(
        service_name=exp.name,
        output_dir=trace_output,
        otlp_endpoint="http://localhost:4318/v1/traces",
    )
    logging.info(f"Metrics output={tracer.output_dir}")

    try:
        with tracer.benchmark(exp.name):
            run_with_ray(exp, n_cpus=4)

        logging.info("Done")
    finally:
        tracer.shutdown()


if __name__ == "__main__":
    main()
