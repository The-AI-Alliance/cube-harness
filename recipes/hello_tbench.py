"""Hello Terminal-Bench recipe.

This recipe runs the Terminal-Bench benchmark with a ReAct agent.
Terminal-Bench evaluates agents on real-world terminal tasks.

Usage:
    # Debug mode (2 tasks, sequential)
    uv run recipes/hello_tbench.py debug

    # Full run (parallel with Ray)
    uv run recipes/hello_tbench.py
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agentlab2.agents.react import ReactAgentConfig  # noqa: E402
from agentlab2.benchmarks.terminalbench import TerminalBenchBenchmark  # noqa: E402
from agentlab2.exp_runner import run_sequentially, run_with_ray  # noqa: E402
from agentlab2.experiment import Experiment  # noqa: E402
from agentlab2.llm import LLMConfig  # noqa: E402
from agentlab2.tools.daytona import DaytonaSWEToolConfig  # noqa: E402

SYSTEM_PROMPT = """You are an expert software engineer working in a Linux terminal environment.
You can execute bash commands, read files, and write files.

Your task is to complete the given objective by working in the /app directory.
Think step by step and use the available tools to accomplish the goal.

Important:
- Work in /app directory
- Read existing files to understand the context
- Test your solutions before declaring completion
- Be precise with file paths and command syntax"""


def main(debug: bool) -> None:
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/terminalbench") / f"tbench_{current_datetime}"

    llm_config = LLMConfig(model_name="openai/gpt-5-nano", tool_choice="required")
    agent_config = ReactAgentConfig(llm_config=llm_config, system_prompt=SYSTEM_PROMPT)
    tool_config = DaytonaSWEToolConfig(api_key=os.getenv("DAYTONA_API_KEY"))

    benchmark = TerminalBenchBenchmark(
        tool_config=tool_config,
        dataset_path="./data/terminal_bench",
        shuffle=True,
        shuffle_seed=42,
        # Start with easy tasks for testing
        task_ids=["hello-world"] if debug else None,
        max_tasks=2 if debug else None,
    )

    exp = Experiment(
        name="terminalbench",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    if debug:
        run_sequentially(exp, debug_limit=1)
    else:
        run_with_ray(exp, n_cpus=4)


if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[-1] == "debug"
    main(debug)
