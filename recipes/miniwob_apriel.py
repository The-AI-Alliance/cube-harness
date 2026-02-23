import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.miniwob.benchmark import MiniWobBenchmark
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from agentlab2.tools.playwright import PlaywrightConfig

load_dotenv()  # Load environment variables from .env file

VLLM_BASE_URL = os.environ["VLLM_BASE_URL"]
VLLM_API_KEY = os.environ["VLLM_API_KEY"]


def main(debug: bool):
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path.home() / "agentlab_results" / "al2" / f"miniwob_{current_datetime}"

    llm_config = LLMConfig(
        model_name="openai/ServiceNow-AI/Apriel-1.6-15b-Thinker",
        api_base=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        temperature=1.0,
        num_retries=0,  # disable retries to see raw errors from Apriel
    )
    agent_config = ReactAgentConfig(llm_config=llm_config)

    tool_config = PlaywrightConfig(use_screenshot=True, headless=True)
    benchmark = MiniWobBenchmark(tool_config=tool_config)

    exp = Experiment(
        name="miniwob",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    if debug:
        run_sequentially(exp, debug_limit=2)
    else:
        run_with_ray(exp, n_cpus=16)


if __name__ == "__main__":
    debug = sys.argv[-1] == "debug"
    main(debug)
