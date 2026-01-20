import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from agentlab2.agents.react import ReactAgentConfig  # noqa: E402
from agentlab2.benchmarks.livecodebench import LiveCodeBenchBenchmark  # noqa: E402
from agentlab2.exp_runner import run_sequentially, run_with_ray  # noqa: E402
from agentlab2.experiment import Experiment  # noqa: E402
from agentlab2.llm import LLMConfig  # noqa: E402
from agentlab2.tools.daytona import DaytonaSWEToolConfig  # noqa: E402

SYSTEM_PROMPT = """You are an expert competitive programmer solving coding problems.
You have access to a Linux sandbox with Python installed.

Strategy:
1. Read and understand the problem carefully
2. Write your solution to /workspace/solution.py
3. Test with the provided examples using: python solution.py < test_input.txt
4. Debug and fix any issues"""


def main(debug: bool) -> None:
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = "outputs/livecodebench" / f"livecodebench_{current_datetime}"

    llm_config = LLMConfig(model_name="openai/gpt-5-nano")
    agent_config = ReactAgentConfig(llm_config=llm_config, system_prompt=SYSTEM_PROMPT)
    tool_config = DaytonaSWEToolConfig(api_key=os.getenv("DAYTONA_API_KEY"))

    benchmark = LiveCodeBenchBenchmark(
        tool_config=tool_config,
        shuffle=True,
        shuffle_seed=42,
    )

    exp = Experiment(
        name="livecodebench",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    if debug:
        run_sequentially(exp, debug_limit=2)
    else:
        run_with_ray(exp, n_cpus=4)


if __name__ == "__main__":
    debug = sys.argv[-1] == "debug"
    main(debug)
