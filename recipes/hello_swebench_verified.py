# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "swebench-verified-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# swebench-verified-cube = { path = "../cubes/swebench-verified-cube", editable = true }
# ///

"""Run swebench-verified-cube with the ReactAgent.

Usage:
    uv run recipes/hello_swebench_verified.py                          # 2 debug tasks, sequential
    uv run recipes/hello_swebench_verified.py --tasks psf__requests-1142,pallets__flask-5014
    uv run recipes/hello_swebench_verified.py --model azure/gpt-5.4 --n-parallel 5
    uv run recipes/hello_swebench_verified.py --repo django/django --model azure/gpt-5.4

Infrastructure:
    The benchmark defaults to LocalInfraConfig (local Docker/Podman).
    To use Daytona, install cube-infra-daytona separately and pass infra=DaytonaInfraConfig()
    to SWEBenchVerifiedBenchmark.

Prerequisites:
    - Docker or Podman running locally (default)
    - Model API key in environment (e.g. AZURE_API_KEY, OPENAI_API_KEY)
"""

import argparse
import logging

from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmark

from cube_harness import make_experiment_output_dir
from cube_harness.agents.react import ReactAgentConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

# Default debug tasks: clean signal, 1 fail_to_pass test each, simple pytest setup.
# psf__requests-1142: don't send Content-Length on GET; 1 f2p, 5 p2p.
# pallets__flask-5014: raise ValueError on empty Blueprint name; 1 f2p, 59 p2p.
DEBUG_TASKS = ["psf__requests-1142", "pallets__flask-5014"]

SWE_SYSTEM_PROMPT = """\
You are an autonomous coding agent. You have access to a Linux sandbox with the repository already cloned at /testbed.
Your task is to resolve the GitHub issue described below. Use the provided tools to explore the codebase, \
understand the problem, and implement a fix.
Start by exploring the repository structure and reading relevant files before making changes.

IMPORTANT — the issue requires you to ADD or CHANGE behavior in the source code. \
The existing test suite will pass before your fix — that is expected. \
Do NOT call final_step just because existing tests pass. \
Only call final_step after you have actually modified the source code to resolve the issue.

Before calling final_step, verify your fix by running the relevant tests.
IMPORTANT: All test dependencies are in the conda 'testbed' environment — always prefix with
`conda run -n testbed` or activate first: `. /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed`
- Django projects: cd /testbed && conda run -n testbed python -m pytest tests/<module> -x -q
  (older Django: ./tests/runtests.py --verbosity 2 <test_module>). Do NOT use "python -m unittest" directly.
- SymPy projects: cd /testbed && conda run -n testbed bin/test <path/to/test_file.py>
- Other Python projects: cd /testbed && conda run -n testbed python -m pytest <test_path> -x -q
Never use bare `python -m pytest` — the base Python lacks test dependencies.

IMPORTANT: Do NOT modify test files (files under tests/ or with test_ prefix). \
The evaluation framework applies its own test patch during evaluation. \
Only modify source code files to fix the bug.

IMPORTANT: Every response must include a tool call — use `final_step` when done."""


def main(
    debug: bool,
    model: str = "azure/gpt-5.4",
    repo: str | None = None,
    task_ids: list[str] | None = None,
    n_parallel: int = 5,
) -> None:
    model_short = model.split("/")[-1]
    output_dir = make_experiment_output_dir("react", "swebench-verified", llm_name=model_short)

    benchmark = SWEBenchVerifiedBenchmark()

    if debug:
        tasks = task_ids or DEBUG_TASKS
        benchmark = benchmark.subset_from_list(tasks)
    elif task_ids is not None:
        benchmark = benchmark.subset_from_list(task_ids)
    elif repo is not None:
        benchmark = benchmark.subset_from_glob("repo", repo)

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

    if debug:
        run_sequentially(exp)
    else:
        run_with_ray(exp, n_cpus=n_parallel, step_timeout_s=1800.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SWE-bench Verified experiments")
    parser.add_argument("--debug", action="store_true", help="Run 2 debug tasks sequentially")
    parser.add_argument("--model", default="azure/gpt-5.4")
    parser.add_argument("--repo", default=None, help="Filter by repository glob, e.g. 'django/django'")
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs")
    parser.add_argument("--n-parallel", type=int, default=5, help="Ray workers for parallel run (default: 5)")
    args = parser.parse_args()
    task_ids = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    main(args.debug, model=args.model, repo=args.repo, task_ids=task_ids, n_parallel=args.n_parallel)
