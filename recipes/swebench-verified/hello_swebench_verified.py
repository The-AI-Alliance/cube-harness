"""Run swebench-verified-cube with AgentLab2.

Usage:
    uv run --project recipes/swebench-verified recipes/swebench-verified/hello_swebench_verified.py debug              # 2 django tasks, sequential
    uv run --project recipes/swebench-verified recipes/swebench-verified/hello_swebench_verified.py 10 --model gpt-4.1 # 10 tasks with Ray
    uv run --project recipes/swebench-verified recipes/swebench-verified/hello_swebench_verified.py full --model gpt-4.1

The recipe in "full" mode runs all 500 tasks. Use --repo to filter by repository, e.g.:
    uv run --project recipes/swebench-verified recipes/swebench-verified/hello_swebench_verified.py full --model gpt-4.1 --repo django/django
"""

import argparse
import logging
import time
from pathlib import Path

from cube.backends.daytona import DaytonaContainerBackend
from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmark

from cube_harness.agents.react import ReactAgentConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

SWE_SYSTEM_PROMPT = """\
You are an autonomous coding agent. You have access to a Linux sandbox with the repository already cloned at /testbed.
Your task is to resolve the GitHub issue described below. Use the provided tools to explore the codebase, \
understand the problem, and implement a fix.
Start by exploring the repository structure and reading relevant files before making changes.
When you are confident the fix is correct, call final_step to submit."""


def main(mode: str, model: str = "gpt-4.1-mini", repo: str | None = None) -> None:
    model_short = model.split("/")[-1]
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path.home() / "cube_harness_results" / f"swebench_verified_{mode}_{model_short}_{current_datetime}"

    backend = DaytonaContainerBackend()

    benchmark = SWEBenchVerifiedBenchmark(container_backend=backend)

    # In debug mode, restrict to 2 django tasks so tests run fast locally
    if mode == "debug":
        benchmark = benchmark.subset_from_glob("repo", "django/django")
        tasks = list(benchmark.task_metadata.keys())[:2]
        benchmark = benchmark.subset_from_list(tasks)
    elif repo is not None:
        benchmark = benchmark.subset_from_glob("repo", repo)

    max_tasks = {"10": 10}.get(mode)
    if max_tasks is not None:
        tasks = list(benchmark.task_metadata.keys())[:max_tasks]
        benchmark = benchmark.subset_from_list(tasks)

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
    parser.add_argument("--repo", default=None, help="Filter by repository, e.g. 'django/django'")
    args = parser.parse_args()
    main(args.mode, model=args.model, repo=args.repo)
