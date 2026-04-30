"""Unified SWE-style agent recipe — swebench-verified, swebench-live, terminalbench.

Usage:
    .venv/bin/python recipes/swe_agent_recipe.py                              # swebench-verified, gpt-5.4-mini, debug
    .venv/bin/python recipes/swe_agent_recipe.py swebench-verified gpt-5.4   # full run
    .venv/bin/python recipes/swe_agent_recipe.py swebench-live gpt-5.4       # swe-bench live
    .venv/bin/python recipes/swe_agent_recipe.py terminalbench gpt-5.4       # terminal-bench

Options:
    --debug              2 tasks, sequential (swebench) or 1 task (terminalbench)
    --hints              Inject task hints (swebench-verified only)
    --tasks t1,t2        Run specific task IDs
    --subset NAME        Named subset: lite/verified/full (swebench-live), easy (terminalbench)
    --n-parallel N       Ray workers (default: 5)
    --retry DIR          Resume / retry from an existing output directory

Each cube must be installed in the active venv:
    uv pip install -e cubes/swebench-verified-cube
    uv pip install -e cubes/swebench-live-cube
    uv pip install -e cubes/terminalbench-cube
"""

import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# meta_agent/ is not a Python package — add it to sys.path so we can import swebench_hints.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "meta_agent"))

# Normalise DOCKER_HOST before load_dotenv — python-dotenv doesn't expand $(...) shell
# expressions, so override=True would clobber the already-expanded value from the shell.
_docker_host = os.environ.get("DOCKER_HOST", "")
if _docker_host.startswith("http+unix://"):
    _docker_host = re.sub(r"^http\+unix://", "unix://", _docker_host)
if _docker_host:
    os.environ["DOCKER_HOST"] = _docker_host

_project_env = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_project_env if _project_env.exists() else Path.home() / ".env", override=True)

# Re-apply after load_dotenv(override=True) to avoid clobbering with unexpanded expression.
if _docker_host:
    os.environ["DOCKER_HOST"] = _docker_host

from cube_harness import make_experiment_output_dir  # noqa: E402
from cube_harness.agents.genny import GennyConfig  # noqa: E402
from cube_harness.exp_runner import run_sequentially, run_with_ray  # noqa: E402
from cube_harness.experiment import Experiment  # noqa: E402
from cube_harness.llm import LLMConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

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

# Default tasks for debug runs (swebench-verified).
DEBUG_TASKS = ["psf__requests-1142", "pallets__flask-5014"]

MODEL_CONFIGS: dict[str, LLMConfig] = {
    "gpt-5.4-mini": LLMConfig(model_name="azure/gpt-5.4-mini"),
    "gpt-5.4": LLMConfig(model_name="azure/gpt-5.4"),
}


# ---------------------------------------------------------------------------
# Benchmark factory
# ---------------------------------------------------------------------------


def _make_benchmark(
    benchmark_name: str,
    debug: bool,
    task_ids: list[str] | None,
    subset: str | None,
) -> object:
    """Instantiate and filter the requested benchmark. Imports are lazy so only
    the installed cube is required."""
    if benchmark_name == "swebench-verified":
        from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmark

        bench = SWEBenchVerifiedBenchmark()
        if debug:
            bench = bench.subset_from_list(task_ids or DEBUG_TASKS)
        elif task_ids:
            bench = bench.subset_from_list(task_ids)
        elif subset:
            bench = bench.named_subset(subset)
        return bench

    if benchmark_name == "swebench-live":
        from swebench_live_cube.benchmark import SWEBenchLiveBenchmark

        bench = SWEBenchLiveBenchmark()
        if subset:
            bench = bench.named_subset(subset)
        if debug:
            tasks = list(bench.task_metadata.keys())[:2]
            bench = bench.subset_from_list(task_ids or tasks)
        elif task_ids:
            bench = bench.subset_from_list(task_ids)
        return bench

    if benchmark_name == "terminalbench":
        from terminalbench_cube import TerminalBenchBenchmark

        TerminalBenchBenchmark.install()
        bench = TerminalBenchBenchmark()
        if subset:
            bench = bench.named_subset(subset)
        if debug:
            tasks = list(bench.task_metadata.keys())[:1]
            bench = bench.subset_from_list(task_ids or tasks)
        elif task_ids:
            bench = bench.subset_from_list(task_ids)
        return bench

    raise ValueError(f"Unknown benchmark: {benchmark_name!r}. Choose: swebench-verified, swebench-live, terminalbench")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(
    benchmark_name: str,
    model_key: str,
    *,
    debug: bool,
    use_hints: bool,
    task_ids: list[str] | None,
    subset: str | None,
    n_parallel: int,
    retry_dir: Path | None,
) -> None:
    llm_config = MODEL_CONFIGS[model_key]

    task_hints: dict[str, str] = {}
    if use_hints and benchmark_name == "swebench-verified":
        from swebench_hints import SWEBENCH_TASK_HINTS

        task_hints = SWEBENCH_TASK_HINTS

    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=SWE_SYSTEM_PROMPT,
        max_actions=30,
        render_last_n_obs=2,
        task_hints=task_hints,
    )

    tag = "hints" if use_hints else None
    if retry_dir is not None:
        output_dir = retry_dir
        resume = True
    else:
        output_dir = make_experiment_output_dir("genny", benchmark_name, llm_name=model_key, tag=tag)
        resume = False

    benchmark = _make_benchmark(benchmark_name, debug, task_ids, subset)

    exp = Experiment(
        name=f"genny-{benchmark_name}",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=30,
        resume=resume,
    )

    label = f"RETRY {retry_dir}" if retry_dir else (f"hints={use_hints}" if benchmark_name == "swebench-verified" else "")
    print(f"\n=== {benchmark_name} | {model_key} | {label or 'no hints'} ===")

    if debug:
        run_sequentially(exp)
    else:
        run_with_ray(exp, n_cpus=n_parallel)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SWE-style benchmarks with the Genny agent")
    parser.add_argument(
        "benchmark",
        nargs="?",
        default="swebench-verified",
        choices=["swebench-verified", "swebench-live", "terminalbench"],
    )
    parser.add_argument("model", nargs="?", default="gpt-5.4-mini", choices=list(MODEL_CONFIGS))
    parser.add_argument("--debug", action="store_true", help="Run 1-2 tasks sequentially")
    parser.add_argument("--hints", action="store_true", help="Inject task hints (swebench-verified only)")
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs")
    parser.add_argument("--subset", default=None, help="Named subset (e.g. lite, easy)")
    parser.add_argument("--n-parallel", type=int, default=5, help="Ray workers (default: 5)")
    parser.add_argument("--retry", metavar="DIR", default=None, help="Resume/retry from output dir")
    args = parser.parse_args()

    run(
        benchmark_name=args.benchmark,
        model_key=args.model,
        debug=args.debug,
        use_hints=args.hints,
        task_ids=[t.strip() for t in args.tasks.split(",")] if args.tasks else None,
        subset=args.subset,
        n_parallel=args.n_parallel,
        retry_dir=Path(args.retry) if args.retry else None,
    )
