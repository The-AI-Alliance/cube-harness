"""Hello Terminal-Bench recipe.

Terminal-Bench evaluates agents on real-world terminal tasks.

Usage:
    uv run recipes/hello_tbench.py debug        # 1 task, sequential
    uv run recipes/hello_tbench.py oracle       # 10 tasks with oracle agent
    uv run recipes/hello_tbench.py oracle_full  # All tasks with oracle agent (30 Ray workers)
    uv run recipes/hello_tbench.py easy         # 4 easy tasks with react agent
    uv run recipes/hello_tbench.py              # Full run with Ray
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agentlab2.agents.oracle import OracleAgentConfig  # noqa: E402
from agentlab2.agents.react import ReactAgentConfig  # noqa: E402
from agentlab2.benchmarks.terminalbench import TerminalBenchBenchmark  # noqa: E402
from agentlab2.exp_runner import run_sequentially, run_with_ray  # noqa: E402
from agentlab2.experiment import Experiment  # noqa: E402
from agentlab2.llm import LLMConfig  # noqa: E402
from agentlab2.tools.daytona import DaytonaSWEToolConfig  # noqa: E402

VALID_MODES = ("debug", "oracle", "oracle_full", "easy", "full")

SYSTEM_PROMPT = """You are an expert software engineer working in a Linux terminal.
Work in /app directory. Read existing files, test your solutions before declaring completion."""


def main(mode: str) -> None:
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/terminalbench") / f"tbench_{mode}_{current_datetime}"

    tool_config = DaytonaSWEToolConfig(api_key=os.getenv("DAYTONA_API_KEY"))

    if mode in ("oracle", "oracle_full"):
        agent_config = OracleAgentConfig()
    else:
        llm_config = LLMConfig(model_name="openai/gpt-5-nano", tool_choice="required")
        agent_config = ReactAgentConfig(llm_config=llm_config, system_prompt=SYSTEM_PROMPT)

    benchmark = TerminalBenchBenchmark(
        tool_config=tool_config,
        shuffle=mode != "oracle_full",
        shuffle_seed=42,
        max_tasks={"debug": 1, "oracle": 10, "easy": 4}.get(mode),
        difficulty_filter="easy" if mode == "easy" else None,
        oracle_mode=mode in ("oracle", "oracle_full"),
    )

    exp = Experiment(
        name="terminalbench",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    if mode in ("debug", "oracle", "easy"):
        run_sequentially(exp, debug_limit=1 if mode == "debug" else None)
    elif mode == "oracle_full":
        run_with_ray(exp, n_cpus=30)
    else:
        run_with_ray(exp, n_cpus=4)


if __name__ == "__main__":
    mode = sys.argv[-1] if len(sys.argv) > 1 and sys.argv[-1] in VALID_MODES else "full"
    main(mode)
