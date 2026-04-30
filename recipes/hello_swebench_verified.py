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

"""Run swebench-verified-cube with the Genny agent.

Usage:
    .venv/bin/python recipes/hello_swebench_verified.py                         # 2 debug tasks, sequential
    .venv/bin/python recipes/hello_swebench_verified.py gpt-5.4                 # gpt-5.4, no hints
    .venv/bin/python recipes/hello_swebench_verified.py gpt-5.4 hints           # with task hints
    .venv/bin/python recipes/hello_swebench_verified.py debug                   # sequential, 2 tasks
    .venv/bin/python recipes/hello_swebench_verified.py retry /path/to/exp      # retry crashed episodes

Task filtering:
    .venv/bin/python recipes/hello_swebench_verified.py --tasks psf__requests-1142,pallets__flask-5014
    .venv/bin/python recipes/hello_swebench_verified.py --repo django/django

Infrastructure:
    Defaults to LocalInfraConfig (local Docker/Podman).
    To use Daytona, install cube-infra-daytona and pass infra=DaytonaInfraConfig()
    to SWEBenchVerifiedBenchmark.
"""

import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env so credentials are available even when the shell didn't source ~/.zshrc.
# Ray workers inherit the parent process env, so this must run before ray.init().
_project_env = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_project_env if _project_env.exists() else Path.home() / ".env", override=True)

# Podman machine sets DOCKER_HOST=http+unix://... which the Docker CLI and Python SDK reject.
# Normalize to unix:// so both tools can connect to the same socket.
_docker_host = os.environ.get("DOCKER_HOST", "")
if _docker_host.startswith("http+unix://"):
    os.environ["DOCKER_HOST"] = re.sub(r"^http\+unix://", "unix://", _docker_host)

from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmark  # noqa: E402

from cube_harness import make_experiment_output_dir  # noqa: E402
from cube_harness.agents.genny import GennyConfig  # noqa: E402
from cube_harness.exp_runner import run_sequentially, run_with_ray  # noqa: E402
from cube_harness.experiment import Experiment  # noqa: E402
from cube_harness.llm import LLMConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

MODEL_CONFIGS: dict[str, LLMConfig] = {
    "gpt-5.4-mini": LLMConfig(model_name="azure/gpt-5.4-mini"),
    "gpt-5.4": LLMConfig(model_name="azure/gpt-5.4"),
}

# Default tasks for debug runs: clean signal, simple pytest setup.
# psf__requests-1142: don't send Content-Length on GET; 1 f2p, 5 p2p.
# pallets__flask-5014: raise ValueError on empty Blueprint name; 1 f2p, 59 p2p.
DEBUG_TASKS = ["psf__requests-1142", "pallets__flask-5014"]

# Per-task hints passed to GennyConfig.task_hints.
# Key: instance_id, Value: hint injected as a ## Task Hint block after the goal.
TASK_HINTS: dict[str, str] = {}

SWE_SYSTEM_PROMPT = """\
You are an autonomous coding agent. You have access to a Linux sandbox with the repository \
already cloned at /testbed.
Your task is to resolve the GitHub issue described below. Use the provided tools to explore \
the codebase, understand the problem, and implement a fix.
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


def run_for_model(
    model_key: str,
    llm_config: LLMConfig,
    debug: bool,
    use_hints: bool,
    task_ids: list[str] | None,
    repo: str | None,
    n_parallel: int,
    retry_dir: Path | None,
) -> None:
    benchmark = SWEBenchVerifiedBenchmark()
    if debug:
        benchmark = benchmark.subset_from_list(task_ids or DEBUG_TASKS)
    elif task_ids is not None:
        benchmark = benchmark.subset_from_list(task_ids)
    elif repo is not None:
        benchmark = benchmark.subset_from_glob("repo", repo)

    suffix = "hints" if use_hints else "nohints"
    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=SWE_SYSTEM_PROMPT,
        max_actions=30,
        render_last_n_obs=2,
        task_hints=TASK_HINTS if use_hints else {},
    )

    if retry_dir is not None:
        output_dir = retry_dir
        retry_failed = True
        resume = True
    else:
        output_dir = make_experiment_output_dir("genny", f"swebench-verified-{suffix}-{model_key}")
        retry_failed = False
        resume = False

    exp = Experiment(
        name=f"swebench-verified-{suffix}-{model_key}",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=30,
        retry_failed=retry_failed,
        resume=resume,
    )

    if debug:
        run_sequentially(exp)
    else:
        run_with_ray(exp, n_cpus=n_parallel, episode_timeout=3600.0)


def main(
    debug: bool,
    models: list[str],
    use_hints: bool,
    task_ids: list[str] | None,
    repo: str | None,
    n_parallel: int,
    retry_dir: Path | None,
) -> None:
    for model_key in models:
        llm_config = MODEL_CONFIGS[model_key]
        hint_label = "WITH hints" if use_hints else "NO hints"
        label = f"RETRY {retry_dir}" if retry_dir else hint_label
        print(f"\n=== {model_key} | {label} ===")
        run_for_model(model_key, llm_config, debug, use_hints, task_ids, repo, n_parallel, retry_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SWE-bench Verified with Genny")
    parser.add_argument("model", nargs="?", default="gpt-5.4-mini", choices=list(MODEL_CONFIGS), help="Model key")
    parser.add_argument("--debug", action="store_true", help="Run 2 debug tasks sequentially")
    parser.add_argument("--hints", action="store_true", help="Inject per-task hints via GennyConfig.task_hints")
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs")
    parser.add_argument("--repo", default=None, help="Filter by repo glob, e.g. 'django/django'")
    parser.add_argument("--n-parallel", type=int, default=5, help="Ray workers (default: 5)")
    parser.add_argument("--retry", metavar="DIR", default=None, help="Retry crashed episodes from output dir")
    args = parser.parse_args()

    main(
        debug=args.debug,
        models=[args.model],
        use_hints=args.hints,
        task_ids=[t.strip() for t in args.tasks.split(",")] if args.tasks else None,
        repo=args.repo,
        n_parallel=args.n_parallel,
        retry_dir=Path(args.retry) if args.retry else None,
    )
