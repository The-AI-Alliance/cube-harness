"""Run swebench-live-cube with AgentLab2.

Usage:
    uv run recipes/hello_swebench_live.py debug              # 2 tasks, sequential
    uv run recipes/hello_swebench_live.py 10 --model gpt-4.1 # 10 tasks with Ray
    uv run recipes/hello_swebench_live.py full --model gpt-4.1
"""

import argparse
import logging
import time
from pathlib import Path

from cube.backends.daytona import DaytonaContainerBackend
from swebench_live_cube.benchmark import SWEBenchLiveBenchmark

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

SWE_SYSTEM_PROMPT = """\
You are an autonomous coding agent. You have access to a Linux sandbox with the repository already cloned at /testbed.
Your task is to resolve the GitHub issue described below. Use the provided tools to explore the codebase, \
understand the problem, and implement a fix.
Start by exploring the repository structure and reading relevant files before making changes.
When you are confident the fix is correct, call final_step to submit."""


def main(mode: str, model: str = "gpt-4.1-mini", split: str = "lite") -> None:
    model_short = model.split("/")[-1]
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path.home() / "agentlab_results" / "al2" / f"swebench_live_{mode}_{model_short}_{current_datetime}"

    backend = DaytonaContainerBackend()

    max_tasks = {"debug": 2, "10": 10}.get(mode)

    benchmark = SWEBenchLiveBenchmark(
        container_backend=backend,
        max_tasks=max_tasks,
        split=split,
    )

    agent_config = ReactAgentConfig(
        llm_config=LLMConfig(model_name=model),
        system_prompt=SWE_SYSTEM_PROMPT,
    )

    exp = Experiment(
        name="swebench-live",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=30,
    )

    if mode == "debug":
        run_sequentially(exp)
    else:
        n_cpus = min(max_tasks or 100, 10)
        run_with_ray(exp, n_cpus=n_cpus, episode_timeout=1800.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SWE-bench Live experiments")
    parser.add_argument("mode", nargs="?", default="debug", choices=["debug", "10", "full"])
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--split", default="lite", choices=["lite", "full"])
    args = parser.parse_args()
    main(args.mode, model=args.model, split=args.split)
