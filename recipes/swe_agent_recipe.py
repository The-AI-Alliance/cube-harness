"""Unified SWE-style agent recipe — swebench-verified, swebench-live, terminalbench.

Usage:
    .venv/bin/python recipes/swe_agent_recipe.py                              # swebench-verified, gpt-5.4-mini, debug
    .venv/bin/python recipes/swe_agent_recipe.py swebench-verified gpt-5.4   # full run
    .venv/bin/python recipes/swe_agent_recipe.py swebench-live gpt-5.4       # swe-bench live
    .venv/bin/python recipes/swe_agent_recipe.py terminalbench gpt-5.4       # terminal-bench

Options:
    --debug              Cube's canonical debug tasks, sequential
    --hints              Inject task hints (swebench-verified only)
    --tasks t1,t2        Run specific task IDs
    --subset NAME        Named subset: lite/verified/full (swebench-live), easy (terminalbench)
    --n-parallel N       Ray workers (default: 5)
    --retry DIR          Resume / retry from an existing output directory
    --toolkit            Use ToolkitInfraConfig — submit each task as an eai job (no local Docker needed)
    --eai-profile PROF   Toolkit profile (default: yul101)
    --eai-path PATH      Path to the eai binary (default: eai; use /bin/eai inside a toolkit job)
    --preemptable        Submit task jobs as preemptable (cheaper; may be interrupted)

Toolkit workflow (from an interactive job):
    uv run recipes/swe_agent_recipe.py --toolkit --eai-path /bin/eai --n-parallel 20 --debug

Each cube must be installed in the active venv:
    uv pip install -e cubes/swebench-verified-cube
    uv pip install -e cubes/swebench-live-cube
    uv pip install -e cubes/terminalbench-cube
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# meta_agent/ is not a Python package — add it to sys.path so we can import hints.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "meta_agent"))

_project_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_project_env if _project_env.exists() else Path.home() / ".env", override=True)

from hints import load_hints  # noqa: E402

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

Before calling final_step, verify your fix by running a targeted test — ideally a single \
test class or test function that exercises the reported behavior. Run `grep -r "def test_" \
tests/ | grep <keyword>` to find the relevant test if you don't know it.
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

IMPORTANT: For navigating source files use `view` — it shows line numbers and tells you \
how many lines are above/below so you can scroll efficiently. Use `bash grep -n` to find \
the right line number, then `view(path, line_start=<N>)` to read context around it. \
Only use `read_file` when you need raw text for `str_replace` (no line numbers needed).

IMPORTANT: For targeted code changes prefer str_replace over bash+sed. \
str_replace fails clearly if the text is not found or is ambiguous, and runs a syntax check \
on .py files after the edit — sed silently succeeds even when it edits the wrong location. \
Once str_replace reports "Replaced 1 occurrence", the fix is in place — do NOT follow it \
with write_file, which would overwrite your edit with whatever you pass as content.

IMPORTANT: Every response must include a tool call — use `final_step` when done. \
Each step is labeled [Step X/N] so you know your remaining budget. \
If you are well past the halfway point and still have not located the root cause, \
call `final_step` — do not spend the remaining steps re-exploring. \
Evaluation runs on whatever file state exists when you stop, so leaving a partial \
but correct change is better than burning steps and reverting to the original."""

MODEL_CONFIGS: dict[str, LLMConfig] = {
    "gpt-5.4-mini": LLMConfig(model_name="azure/gpt-5.4-mini"),
    "gpt-5.4": LLMConfig(model_name="azure/gpt-5.4"),
}


# ---------------------------------------------------------------------------
# Benchmark factory
# ---------------------------------------------------------------------------


def _make_infra(
    toolkit: bool,
    eai_profile: str,
    eai_path: str,
    preemptable: bool,
) -> object:
    """Return the appropriate InfraConfig. Lazy import so cube_infra_toolkit is optional."""
    if toolkit:
        from cube_infra_toolkit import ToolkitInfraConfig

        return ToolkitInfraConfig(
            profile=eai_profile,
            eai_path=eai_path,
            preemptable=preemptable,
            launch_timeout_seconds=300,
        )
    from cube.infra_local import LocalInfraConfig

    return LocalInfraConfig()


def _make_benchmark_config(
    benchmark_name: str,
    debug: bool,
    task_ids: list[str] | None,
    subset: str | None,
    infra: object,
) -> object:
    """Return a BenchmarkConfig (not a live Benchmark). Imports are lazy so only
    the installed cube is required. The Experiment runner calls .make(infra)."""
    if debug:
        if benchmark_name == "swebench-verified":
            from swebench_verified_cube.debug import get_debug_benchmark
            return get_debug_benchmark(infra=infra)
        elif benchmark_name == "swebench-live":
            from swebench_live_cube.debug import get_debug_benchmark
        elif benchmark_name == "terminalbench":
            from terminalbench_cube.debug import get_debug_benchmark
        else:
            raise ValueError(
                f"Unknown benchmark: {benchmark_name!r}. Choose: swebench-verified, swebench-live, terminalbench"
            )
        bench = get_debug_benchmark()
        bench.infra = infra
        return bench

    if benchmark_name == "swebench-verified":
        from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmarkConfig

        bench = SWEBenchVerifiedBenchmarkConfig()
    elif benchmark_name == "swebench-live":
        from swebench_live_cube.benchmark import SWEBenchLiveBenchmarkConfig

        bench = SWEBenchLiveBenchmarkConfig()
    elif benchmark_name == "terminalbench":
        from terminalbench_cube.benchmark import TerminalBenchBenchmarkConfig

        bench = TerminalBenchBenchmarkConfig()
    else:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name!r}. Choose: swebench-verified, swebench-live, terminalbench"
        )

    if subset:
        bench = bench.named_subset(subset)
    if task_ids:
        bench = bench.subset_from_list(task_ids)
    return bench


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
    toolkit: bool,
    eai_profile: str,
    eai_path: str,
    preemptable: bool,
    max_actions: int = 50,
) -> None:
    llm_config = MODEL_CONFIGS[model_key]

    task_hints = load_hints(benchmark_name) if use_hints else {}

    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=SWE_SYSTEM_PROMPT,
        max_actions=max_actions,
        render_last_n_obs=2,
        task_hints=task_hints,
    )

    if retry_dir is not None:
        output_dir = retry_dir
        resume = True
    else:
        output_dir = None
        resume = False

    infra = _make_infra(toolkit, eai_profile, eai_path, preemptable)
    benchmark_config = _make_benchmark_config(benchmark_name, debug, task_ids, subset, infra)

    infra_label = f"toolkit:{eai_profile}" if toolkit else "local"
    exp = Experiment(
        name=f"genny-{benchmark_name}-{infra_label}",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        max_steps=max_actions,
        resume=resume,
    )

    label = (
        f"RETRY {retry_dir}" if retry_dir else (f"hints={use_hints}" if benchmark_name == "swebench-verified" else "")
    )
    print(f"\n=== {benchmark_name} | {model_key} | {infra_label} | {label or 'no hints'} ===")

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
    parser.add_argument("--debug", action="store_true", help="Run cube debug tasks sequentially")
    parser.add_argument("--hints", action="store_true", help="Inject task hints")
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs")
    parser.add_argument("--subset", default=None, help="Named subset (e.g. lite, easy)")
    parser.add_argument("--n-parallel", type=int, default=5, help="Ray workers (default: 5)")
    parser.add_argument("--retry", metavar="DIR", default=None, help="Resume/retry from output dir")
    parser.add_argument("--toolkit", action="store_true", help="Use ToolkitInfraConfig (submit each task as an eai job)")
    parser.add_argument("--eai-profile", default="yul101", help="Toolkit profile (default: yul101)")
    parser.add_argument("--eai-path", default="eai", help="Path to eai binary (default: eai; use /bin/eai inside a toolkit job)")
    parser.add_argument("--preemptable", action="store_true", help="Submit task jobs as preemptable")
    parser.add_argument("--max-actions", type=int, default=50, help="Step budget per episode (default: 50)")
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
        toolkit=args.toolkit,
        eai_profile=args.eai_profile,
        eai_path=args.eai_path,
        preemptable=args.preemptable,
        max_actions=args.max_actions,
    )
