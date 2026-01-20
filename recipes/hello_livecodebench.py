"""Run LiveCodeBench evaluation with ReactAgent and DaytonaSWETool."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agentlab2.agents.react import ReactAgentConfig  # noqa: E402
from agentlab2.benchmarks.livecodebench import LiveCodeBenchBenchmark  # noqa: E402
from agentlab2.exp_runner import run_sequentially  # noqa: E402
from agentlab2.experiment import Experiment  # noqa: E402
from agentlab2.llm import LLMConfig  # noqa: E402
from agentlab2.tools.daytona import DaytonaSWEToolConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

system_prompt = """You are an expert competitive programmer solving coding problems.
You have access to a Linux sandbox with Python installed.

Strategy:
1. Read and understand the problem carefully
2. Write your solution to /workspace/solution.py
3. Test with the provided examples using: python solution.py < test_input.txt
4. Debug and fix any issues"""


def main() -> None:
    llm_config = LLMConfig(model_name="openai/gpt-5-nano")
    agent_config = ReactAgentConfig(llm_config=llm_config, system_prompt=system_prompt)
    tool_config = DaytonaSWEToolConfig(api_key=os.getenv("DAYTONA_API_KEY"))
    benchmark = LiveCodeBenchBenchmark(
        tool_config=tool_config,
        max_tasks=1,
        shuffle=True,
        shuffle_seed=42,
    )

    output_dir = Path(__file__).parent.parent / "outputs" / "livecodebench"
    experiment = Experiment(
        name="livecodebench",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    logger.info(f"Running LiveCodeBench experiment: {experiment.name}")
    logger.info(f"Output directory: {output_dir}")

    # Run sequentially for debugging
    run_sequentially(experiment, debug_limit=3)


if __name__ == "__main__":
    main()
