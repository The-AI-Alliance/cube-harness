"""Genny2 Terminal-Bench recipe — real-world terminal tasks with pytest-based validation.

Usage:
    # Debug run (oracle tasks, sequential, no LLM):
    .venv/bin/python recipes/genny2_terminalbench_recipe.py --debug

    # Easy tasks, local Docker:
    .venv/bin/python recipes/genny2_terminalbench_recipe.py --difficulty easy

    # Full run on Toolkit:
    .venv/bin/python recipes/genny2_terminalbench_recipe.py \\
        --toolkit --eai-profile yul101 --eai-path ~/bin/eai --n-parallel 20

    # Specific model:
    .venv/bin/python recipes/genny2_terminalbench_recipe.py sonnet --debug
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

_project_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_project_env if _project_env.exists() else Path.home() / ".env", override=True)

from cube_harness.agents.genny2_swe_config import INSTANCE_TEMPLATES, MODEL_CONFIGS, make_agent_config  # noqa: E402
from cube_harness.exp_runner import run_sequentially, run_with_ray  # noqa: E402
from cube_harness.experiment import Experiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")


def _make_infra(toolkit: bool, eai_profile: str, eai_path: str, preemptable: bool) -> object:
    if toolkit:
        from cube_infra_toolkit import ToolkitInfraConfig

        return ToolkitInfraConfig(
            profile=eai_profile,
            eai_path=eai_path,
            preemptable=preemptable,
            launch_timeout_seconds=3000,
        )
    from cube.infra_local import LocalInfraConfig

    return LocalInfraConfig()


def _make_benchmark_config(
    debug: bool,
    difficulty: str | None,
    category: str | None,
    task_ids: list[str] | None,
    oracle_mode: bool,
) -> object:
    from terminalbench_cube.benchmark import TerminalBenchBenchmarkConfig
    from terminalbench_cube.tool import TerminalBenchToolConfig

    if debug:
        from terminalbench_cube.debug import get_debug_benchmark

        return get_debug_benchmark()

    TerminalBenchBenchmarkConfig.install()
    config = TerminalBenchBenchmarkConfig(
        tool_config=TerminalBenchToolConfig(),
        oracle_mode=oracle_mode,
    )
    if difficulty:
        config = config.subset_from_glob("difficulty", difficulty)
    if category:
        config = config.subset_from_glob("category", category)
    if task_ids:
        config = config.subset_from_list(task_ids)
    return config


def run(
    model_key: str,
    *,
    template: str,
    debug: bool,
    difficulty: str | None,
    category: str | None,
    task_ids: list[str] | None,
    oracle_mode: bool,
    n_parallel: int,
    retry_dir: Path | None,
    toolkit: bool,
    eai_profile: str,
    eai_path: str,
    preemptable: bool,
    max_actions: int = 100,
    cost_limit: float = 1.0,
) -> None:
    agent_config = make_agent_config(model_key, template, max_actions, cost_limit)

    infra = _make_infra(toolkit, eai_profile, eai_path, preemptable)
    benchmark_config = _make_benchmark_config(debug, difficulty, category, task_ids, oracle_mode)

    output_dir = retry_dir if retry_dir is not None else None
    resume = retry_dir is not None

    infra_label = f"toolkit:{eai_profile}" if toolkit else "local"
    exp = Experiment(
        name=f"genny2-tbench-{model_key}-{template}-{infra_label}",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=infra,
        max_steps=max_actions,
        resume=resume,
    )

    print(f"\n=== genny2 | {model_key} | terminalbench | {template} | {infra_label} ===")

    if debug:
        run_sequentially(exp)
    else:
        run_with_ray(exp, n_cpus=n_parallel)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genny2 Terminal-Bench recipe")
    parser.add_argument("model", nargs="?", default="gpt-5.4-mini", choices=list(MODEL_CONFIGS))
    parser.add_argument(
        "--template",
        default="thought-workflow",
        choices=list(INSTANCE_TEMPLATES),
        help="Instance template variant (default: thought-workflow)",
    )
    parser.add_argument("--debug", action="store_true", help="Run debug oracle tasks sequentially")
    parser.add_argument("--difficulty", default=None, help="Filter by difficulty (easy, medium, hard)")
    parser.add_argument("--category", default=None, help="Filter by category glob")
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs")
    parser.add_argument("--oracle-mode", action="store_true", help="Upload gold solution at reset")
    parser.add_argument("--n-parallel", type=int, default=5)
    parser.add_argument("--retry", metavar="DIR", default=None, help="Resume from output dir")
    parser.add_argument("--toolkit", action="store_true")
    parser.add_argument("--eai-profile", default="yul101")
    parser.add_argument("--eai-path", default="eai")
    parser.add_argument("--preemptable", action="store_true")
    parser.add_argument("--max-actions", type=int, default=100)
    parser.add_argument("--cost-limit", type=float, default=1.0)
    args = parser.parse_args()

    run(
        model_key=args.model,
        template=args.template,
        debug=args.debug,
        difficulty=args.difficulty,
        category=args.category,
        task_ids=[t.strip() for t in args.tasks.split(",")] if args.tasks else None,
        oracle_mode=args.oracle_mode,
        n_parallel=args.n_parallel,
        retry_dir=Path(args.retry) if args.retry else None,
        toolkit=args.toolkit,
        eai_profile=args.eai_profile,
        eai_path=args.eai_path,
        preemptable=args.preemptable,
        max_actions=args.max_actions,
        cost_limit=args.cost_limit,
    )
