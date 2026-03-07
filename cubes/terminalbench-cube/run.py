"""Run terminalbench-cube with AgentLab2.

Usage:
    uv run cubes/terminalbench-cube/run.py debug              # 1 task, sequential
    uv run cubes/terminalbench-cube/run.py oracle              # 10 tasks with oracle
    uv run cubes/terminalbench-cube/run.py easy                # easy tasks with react agent
"""

import argparse
import logging
import os
import time
from pathlib import Path

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.exp_runner import run_sequentially
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from cube.backends.daytona import DaytonaContainerBackend
from terminalbench_cube import TerminalBenchBenchmark, TerminalBenchToolConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

SYSTEM_PROMPT = """You are an expert software engineer working in a Linux terminal.
Work in /app directory. Read existing files, test your solutions before declaring completion."""


def main(mode: str, model: str = "openai/gpt-5-nano") -> None:
    model_short = model.split("/")[-1]
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path.home() / "agentlab_results" / "al2" / f"tbench_cube_{mode}_{model_short}_{current_datetime}"

    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        raise ValueError("DAYTONA_API_KEY environment variable is required")

    container_backend = DaytonaContainerBackend(api_key=api_key)
    tool_config = TerminalBenchToolConfig()

    llm_config = LLMConfig(model_name=model, tool_choice="required")
    agent_config = ReactAgentConfig(
        llm_config=llm_config,
        system_prompt=SYSTEM_PROMPT,
        max_actions=100,
        max_obs_chars=200000,
        max_history_tokens=240000,
    )

    benchmark = TerminalBenchBenchmark(
        container_backend=container_backend,
        default_tool_config=tool_config,
        shuffle=True,
        shuffle_seed=42,
        max_tasks={"debug": 1, "easy": 4}.get(mode),
        difficulty_filter="easy" if mode == "easy" else None,
    )

    benchmark.install()
    benchmark.setup()

    exp = Experiment(
        name="terminalbench-cube",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    run_sequentially(exp, debug_limit=1 if mode == "debug" else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Terminal-Bench cube experiments")
    parser.add_argument("mode", nargs="?", default="debug", choices=["debug", "easy", "full"])
    parser.add_argument("--model", default="openai/gpt-5-nano")
    args = parser.parse_args()
    main(args.mode, model=args.model)
