"""Run swebench-verified-cube with AgentLab2.

Usage:
    uv run recipes/hello_swebench_verified.py debug              # 2 django tasks, sequential
    uv run recipes/hello_swebench_verified.py 10 --model gpt-4.1 # 10 tasks with Ray
    uv run recipes/hello_swebench_verified.py full --model gpt-4.1
"""

import argparse
import logging
import time
from pathlib import Path

from cube.backends.daytona import DaytonaContainerBackend
from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmark

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


def main(mode: str, model: str = "gpt-4.1-mini") -> None:
    model_short = model.split("/")[-1]
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path.home() / "agentlab_results" / "al2" / f"swebench_verified_{mode}_{model_short}_{current_datetime}"

    backend = DaytonaContainerBackend()

    max_tasks = {"debug": 2, "10": 10}.get(mode)
    repo_filter = "django/django" if mode == "debug" else None

    benchmark = SWEBenchVerifiedBenchmark(
        container_backend=backend,
        max_tasks=max_tasks,
        repo_filter=repo_filter,
    )

    agent_config = ReactAgentConfig(
        llm_config=LLMConfig(model_name=model),
        system_prompt=SWE_SYSTEM_PROMPT,
    )

    exp = Experiment(
        name="swebench-verified",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=30,
    )

    if mode == "debug":
        run_sequentially(exp)
    else:
        n_cpus = min(max_tasks or 500, 10)
        run_with_ray(exp, n_cpus=n_cpus, episode_timeout=1800.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SWE-bench Verified experiments")
    parser.add_argument("mode", nargs="?", default="debug", choices=["debug", "10", "full"])
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()
    main(args.mode, model=args.model)
