"""Genny2 SWE recipe — swebench-verified and swebench-live.

Single entry point for Genny2 against both SWE-bench Verified and SWE-bench Live.
Agent construction is delegated to ``cube_harness.agents.genny2_swe_config``.

Usage:
    # Debug oracle tasks (no LLM, sequential):
    .venv/bin/python recipes/swe_agent_recipe.py --debug
    .venv/bin/python recipes/swe_agent_recipe.py --benchmark live --debug

    # Full swebench-verified on Toolkit:
    .venv/bin/python recipes/swe_agent_recipe.py --toolkit --eai-path ~/bin/eai --n-parallel 20

    # HAL-50 subset on swebench-verified:
    .venv/bin/python recipes/swe_agent_recipe.py --subset hal_mini --toolkit ...

    # SWE-bench Live, golden 30, Daytona:
    .venv/bin/python recipes/swe_agent_recipe.py --benchmark live --subset live-golden-30 --daytona
"""

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

_project_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_project_env if _project_env.exists() else Path.home() / ".env", override=True)

from cube_harness.agents.genny2_swe_config import (  # noqa: E402
    DEFAULT_TEMPLATE,
    INSTANCE_TEMPLATES,
    MODEL_CONFIGS,
    make_agent_config,
)
from cube_harness.exp_runner import run_sequentially, run_with_ray  # noqa: E402
from cube_harness.experiment import Experiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

# ---------------------------------------------------------------------------
# Hardcoded task subsets
#
# These belong long-term in each cube's BenchmarkConfig as named subsets; they
# live here for now because the cubes ship the full benchmark and these subsets
# are recipe-level concerns (which Princeton split, which gold-confirmed sample).
# ---------------------------------------------------------------------------

_HAL_MINI_TASK_IDS: frozenset[str] = frozenset(
    [
        # Django (25)
        "django__django-11790",
        "django__django-11815",
        "django__django-11848",
        "django__django-11880",
        "django__django-11885",
        "django__django-11951",
        "django__django-11964",
        "django__django-11999",
        "django__django-12039",
        "django__django-12050",
        "django__django-12143",
        "django__django-12155",
        "django__django-12193",
        "django__django-12209",
        "django__django-12262",
        "django__django-12273",
        "django__django-12276",
        "django__django-12304",
        "django__django-12308",
        "django__django-12325",
        "django__django-12406",
        "django__django-12708",
        "django__django-12713",
        "django__django-12774",
        "django__django-9296",
        # Sphinx (25)
        "sphinx-doc__sphinx-10323",
        "sphinx-doc__sphinx-10435",
        "sphinx-doc__sphinx-10466",
        "sphinx-doc__sphinx-10673",
        "sphinx-doc__sphinx-11510",
        "sphinx-doc__sphinx-7590",
        "sphinx-doc__sphinx-8056",
        "sphinx-doc__sphinx-8265",
        "sphinx-doc__sphinx-8269",
        "sphinx-doc__sphinx-8475",
        "sphinx-doc__sphinx-8548",
        "sphinx-doc__sphinx-8551",
        "sphinx-doc__sphinx-8638",
        "sphinx-doc__sphinx-8721",
        "sphinx-doc__sphinx-9229",
        "sphinx-doc__sphinx-9230",
        "sphinx-doc__sphinx-9281",
        "sphinx-doc__sphinx-9320",
        "sphinx-doc__sphinx-9367",
        "sphinx-doc__sphinx-9461",
        "sphinx-doc__sphinx-9698",
    ]
)


# ---------------------------------------------------------------------------
# Infra / benchmark helpers
# ---------------------------------------------------------------------------


def _make_infra(
    *,
    toolkit: bool,
    daytona: bool,
    eai_profile: str,
    eai_path: str,
    preemptable: bool,
    launch_timeout: int,
) -> object:
    if daytona:
        from cube_infra_daytona import DaytonaInfraConfig

        return DaytonaInfraConfig(launch_timeout_seconds=launch_timeout)
    if toolkit:
        from cube_infra_toolkit import ToolkitInfraConfig

        return ToolkitInfraConfig(
            profile=eai_profile,
            eai_path=eai_path,
            preemptable=preemptable,
            launch_timeout_seconds=launch_timeout,
        )
    from cube.infra_local import LocalInfraConfig

    return LocalInfraConfig()


def _make_verified_benchmark(
    *,
    debug: bool,
    task_ids: list[str] | None,
    subset: str | None,
) -> object:
    from cube.tools.terminal import TerminalToolConfig
    from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmarkConfig

    if debug:
        from swebench_verified_cube.debug import get_debug_benchmark

        return get_debug_benchmark()

    cfg = SWEBenchVerifiedBenchmarkConfig(tool_config=TerminalToolConfig(working_dir="/testbed"))
    if subset == "hal_mini":
        cfg = cfg.subset_from_list(list(_HAL_MINI_TASK_IDS))
    elif subset:
        cfg = cfg.named_subset(subset)
    if task_ids:
        cfg = cfg.subset_from_list(task_ids)
    return cfg


def _make_live_benchmark(
    *,
    debug: bool,
    task_ids: list[str] | None,
    subset: str | None,
    n_tasks: int | None,
    solvable_from: Path | None,
) -> object:
    from swebench_live_cube.benchmark import SWEBenchLiveBenchmarkConfig

    if debug:
        from swebench_live_cube.debug import get_debug_benchmark

        return get_debug_benchmark()

    cfg = SWEBenchLiveBenchmarkConfig()
    if solvable_from is not None:
        task_ids = json.loads(solvable_from.read_text())
    if subset == "live-golden-30":
        from swebench_live_cube.gold_patch.recipe import _LIVE_GOLDEN_30

        cfg = cfg.subset_from_list(list(_LIVE_GOLDEN_30))
    elif subset:
        cfg = cfg.named_subset(subset)
    if task_ids:
        cfg = cfg.subset_from_list(task_ids)
    elif n_tasks:
        cfg = cfg.subset_from_list(list(cfg.task_metadata.keys())[:n_tasks])
    return cfg


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    agent_config = make_agent_config(args.model, args.template, args.max_actions, args.cost_limit)
    infra = _make_infra(
        toolkit=args.toolkit,
        daytona=args.daytona,
        eai_profile=args.eai_profile,
        eai_path=args.eai_path,
        preemptable=args.preemptable,
        launch_timeout=args.launch_timeout,
    )

    task_ids = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    if args.benchmark == "verified":
        benchmark_config = _make_verified_benchmark(
            debug=args.debug,
            task_ids=task_ids,
            subset=args.subset,
        )
    else:
        benchmark_config = _make_live_benchmark(
            debug=args.debug,
            task_ids=task_ids,
            subset=args.subset,
            n_tasks=args.n_tasks,
            solvable_from=Path(args.solvable_from) if args.solvable_from else None,
        )

    infra_label = "daytona" if args.daytona else (f"toolkit:{args.eai_profile}" if args.toolkit else "local")
    retry_dir = Path(args.retry) if args.retry else None
    exp = Experiment(
        name=f"genny2-swe-{args.benchmark}-{args.model}-{infra_label}",
        output_dir=retry_dir,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=infra,
        max_steps=args.max_actions,
        resume=retry_dir is not None,
    )

    print(
        f"\n=== genny2 | swe-{args.benchmark} | {args.model} | {args.template} | {infra_label} | subset={args.subset or 'all'} ==="
    )

    if args.debug:
        run_sequentially(exp)
    else:
        run_with_ray(exp, n_cpus=args.n_parallel)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genny2 SWE recipe — verified or live")
    parser.add_argument("model", nargs="?", default="gpt-5.4-mini", choices=list(MODEL_CONFIGS))
    parser.add_argument(
        "--benchmark",
        default="verified",
        choices=["verified", "live"],
        help="Which SWE-bench variant to run (default: verified)",
    )
    parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        choices=list(INSTANCE_TEMPLATES),
        help=f"Instance template variant (default: {DEFAULT_TEMPLATE})",
    )
    parser.add_argument("--debug", action="store_true", help="Run cube's debug oracle tasks sequentially (no LLM)")
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs")
    parser.add_argument(
        "--subset",
        default=None,
        help="Named subset. Verified: hal_mini, or any named subset on the cube. "
        "Live: live-golden-30, solvable-lite, lite, verified, full, test.",
    )
    parser.add_argument("--n-tasks", type=int, default=None, help="Take first N tasks from subset (live only)")
    parser.add_argument("--n-parallel", type=int, default=5)
    parser.add_argument("--retry", metavar="DIR", default=None, help="Resume from output dir")
    parser.add_argument(
        "--solvable-from",
        metavar="PATH",
        default=None,
        help="JSON file of solvable task IDs (e.g. output of gold_patch.recipe --dump-solvable). Live only.",
    )
    parser.add_argument("--toolkit", action="store_true")
    parser.add_argument("--daytona", action="store_true", help="Use Daytona infra (requires DAYTONA_API_KEY)")
    parser.add_argument("--eai-profile", default="yul101")
    parser.add_argument("--eai-path", default="eai")
    parser.add_argument("--preemptable", action="store_true")
    parser.add_argument("--max-actions", type=int, default=150)
    parser.add_argument("--cost-limit", type=float, default=1.0)
    parser.add_argument("--launch-timeout", type=int, default=900)
    args = parser.parse_args()
    run(args)
