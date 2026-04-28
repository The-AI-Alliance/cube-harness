# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "swebench-verified-cube",
#     "cube-infra-daytona",
#     "python-dotenv",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# swebench-verified-cube = { path = "../cubes/swebench-verified-cube", editable = true }
# cube-infra-daytona = { path = "../../cube-standard/cube-resources/cube-infra-daytona", editable = true }
# ///

"""Run swebench-verified-cube with AgentLab2.

Usage:
    uv run recipes/hello_swebench_verified.py debug              # 2 default tasks, sequential
    uv run recipes/hello_swebench_verified.py debug --tasks psf__requests-1142,pallets__flask-5014
    uv run recipes/hello_swebench_verified.py 10 --model gpt-5.4 # 10 tasks with Ray
    uv run recipes/hello_swebench_verified.py full --model gpt-5.4

The recipe in "full" mode runs all 500 tasks. Use --repo to filter by repository, e.g.:
    uv run recipes/hello_swebench_verified.py full --model gpt-5.4 --repo django/django

Prerequisites:
    - DAYTONA_API_KEY in env (or ~/.env-cube)
    - Sibling checkout of cube-standard at ../../cube-standard for the
      cube-infra-daytona editable dep above
"""

import argparse
import logging
import time
from pathlib import Path

from cube_infra_daytona import DaytonaInfraConfig
from dotenv import load_dotenv
from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmark

from cube_harness.agents.react import ReactAgentConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig

# Credentials: prefer ~/.env-cube (per PR #314), fall back to ~/.env when keys live there.
load_dotenv(Path.home() / ".env-cube")
load_dotenv(Path.home() / ".env", override=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

SWE_SYSTEM_PROMPT = """\
You are an autonomous coding agent. You have access to a Linux sandbox with the repository already cloned at /testbed.
Your task is to resolve the GitHub issue described below. Use the provided tools to explore the codebase, \
understand the problem, and implement a fix.
Start by exploring the repository structure and reading relevant files before making changes.

IMPORTANT — the issue requires you to ADD or CHANGE behavior in the source code. \
The existing test suite will pass before your fix — that is expected. \
Do NOT call final_step just because existing tests pass. \
Only call final_step after you have actually modified the source code to resolve the issue.

Before calling final_step, verify your fix by running the relevant tests:
- Django projects: cd /testbed && ./tests/runtests.py --verbosity 2 <test_module>
  (e.g. ./tests/runtests.py validators for validators tests). Do NOT use "python -m unittest" directly.
- SymPy projects: cd /testbed && ./bin/test <path/to/test_file.py>
- Other Python projects: cd /testbed && python -m pytest <test_path> -x -q

IMPORTANT: Every response must include a tool call — use `final_step` when done."""


# Default debug tasks: clean signal, 1 fail_to_pass test each, simple pytest setup (no Django DB).
# psf__requests-1142: don't send Content-Length on GET; 1 f2p, 5 p2p.
# pallets__flask-5014: raise ValueError on empty Blueprint name; 1 f2p, 59 p2p.
DEBUG_TASKS = ["psf__requests-1142", "pallets__flask-5014"]


def main(
    mode: str,
    model: str = "azure/gpt-5.4",
    repo: str | None = None,
    task_ids: list[str] | None = None,
) -> None:
    model_short = model.split("/")[-1]
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path.home() / "cube_harness_results" / f"swebench_verified_{mode}_{model_short}_{current_datetime}"

    infra = DaytonaInfraConfig()

    benchmark = SWEBenchVerifiedBenchmark(infra=infra)

    if mode == "debug":
        tasks = task_ids or DEBUG_TASKS
        benchmark = benchmark.subset_from_list(tasks)
    elif task_ids is not None:
        benchmark = benchmark.subset_from_list(task_ids)
    elif repo is not None:
        benchmark = benchmark.subset_from_glob("repo", repo)

    max_tasks = {"10": 10}.get(mode)
    if max_tasks is not None:
        tasks = list(benchmark.task_metadata.keys())[:max_tasks]
        benchmark = benchmark.subset_from_list(tasks)

    # max_actions is per-agent and defaults to 10 — must be raised together with
    # max_steps or the harness force-stops the agent at turn 10 with reward 0.
    # SWE-bench tasks typically need 20+ steps to explore + diff + verify.
    agent_config = ReactAgentConfig(
        llm_config=LLMConfig(model_name=model),
        system_prompt=SWE_SYSTEM_PROMPT,
        max_actions=30,
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
        n_cpus = min(max_tasks or 500, 5)  # Daytona free tier: ~10GiB total, ~2GiB/sandbox
        run_with_ray(exp, n_cpus=n_cpus, step_timeout_s=1800.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SWE-bench Verified experiments")
    parser.add_argument("mode", nargs="?", default="debug", choices=["debug", "10", "full"])
    parser.add_argument("--model", default="azure/gpt-5.4")
    parser.add_argument("--repo", default=None, help="Filter by repository, e.g. 'django/django'")
    parser.add_argument(
        "--tasks", default=None, help="Comma-separated task IDs to run, e.g. 'psf__requests-1142,pallets__flask-5014'"
    )
    args = parser.parse_args()
    task_ids = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    main(args.mode, model=args.model, repo=args.repo, task_ids=task_ids)
