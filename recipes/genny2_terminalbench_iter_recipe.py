"""Genny2 TerminalBench iteration recipe — fixed 40-task medium subset.

Fixed task list for reproducible iteration across agent/config changes.
Uses the generic SWE agent config (not tbench-specific) as a neutral baseline.

Usage:
    # Toolkit run (standard):
    .venv/bin/python recipes/genny2_terminalbench_iter_recipe.py \
        --toolkit --eai-profile yul101 --sidecar-data snow.allac.cube_sidecar

    # Debug run (scripted oracle agent, no LLM):
    .venv/bin/python recipes/genny2_terminalbench_iter_recipe.py --debug \
        --toolkit --eai-profile yul101 --sidecar-data snow.allac.cube_sidecar

    # Single task for quick debugging:
    .venv/bin/python recipes/genny2_terminalbench_iter_recipe.py \
        --tasks chess-best-move \
        --toolkit --eai-profile yul101 --sidecar-data snow.allac.cube_sidecar

    # Different model:
    .venv/bin/python recipes/genny2_terminalbench_iter_recipe.py gpt-5.4 \
        --toolkit --eai-profile yul101 --sidecar-data snow.allac.cube_sidecar
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

_project_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_project_env if _project_env.exists() else Path.home() / ".env", override=True)

import terminalbench_cube.debug as _tbench_debug  # noqa: E402
from cube.testing import run_debug_suite  # noqa: E402

from cube_harness.agents.genny2_swe_config import (  # noqa: E402
    DEFAULT_TEMPLATE,
    INSTANCE_TEMPLATES,
    MODEL_CONFIGS,
    make_agent_config,
)
from cube_harness.exp_runner import run_with_ray  # noqa: E402
from cube_harness.experiment import Experiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

# Fixed 40-task medium subset — proportional across all 14 categories.
# Keep this list stable across iterations so scores are directly comparable.
ITER_TASKS: list[str] = [
    # data-processing (4)
    "financial-document-processor",
    "log-summary-date-ranges",
    "multi-source-data-merger",
    "regex-log",
    # data-science (3)
    "modernize-scientific-stack",
    "query-optimize",
    "rstan-to-pystan",
    # debugging (4)
    "build-cython-ext",
    "custom-memory-heap-crash",
    "merge-diff-arc-agi-task",
    "sqlite-db-truncate",
    # file-operations (4)
    "db-wal-recovery",
    "extract-elf",
    "gcode-to-text",
    "large-scale-text-editing",
    # games (1)
    "chess-best-move",
    # machine-learning (1)
    "distribution-search",
    # mathematics (1)
    "largest-eigenval",
    # model-training (1)
    "count-dataset-tokens",
    # optimization (1)
    "portfolio-optimization",
    # personal-assistant (1)
    "constraints-scheduling",
    # scientific-computing (3)
    "adaptive-rejection-sampler",
    "raman-fitting",
    "tune-mjcf",
    # security (5)
    "break-filter-js-from-html",
    "crack-7z-hash",
    "filter-js-from-html",
    "openssl-selfsigned-cert",
    "vulnerable-secret",
    # software-engineering (6)
    "build-pov-ray",
    "code-from-image",
    "kv-store-grpc",
    "polyglot-c-py",
    "pypi-server",
    "schemelike-metacircular-eval",
    # system-administration (5)
    "compile-compcert",
    "git-multibranch",
    "nginx-request-logging",
    "qemu-startup",
    "sqlite-with-gcov",
]


def _make_infra(
    toolkit: bool,
    eai_profile: str,
    eai_path: str,
    preemptable: bool,
    sidecar_data: str | None,
    assets_data: str | None = None,
) -> object:
    if toolkit:
        from cube_infra_toolkit import ToolkitInfraConfig

        return ToolkitInfraConfig(
            profile=eai_profile,
            eai_path=eai_path,
            preemptable=preemptable,
            launch_timeout_seconds=3000,
            sidecar_data=sidecar_data,
            assets_data=assets_data,
        )
    from cube.infra_local import LocalInfraConfig

    return LocalInfraConfig()


def _make_benchmark_config(task_ids: list[str], max_output_bytes: int) -> object:
    from cube.tools.terminal import TerminalToolConfig
    from terminalbench_cube.benchmark import TerminalBenchBenchmarkConfig

    TerminalBenchBenchmarkConfig.install()
    return TerminalBenchBenchmarkConfig(
        tool_config=TerminalToolConfig(
            working_dir="/app",
            max_timeout=900,
            enable_file_actions=True,
            max_output_bytes=max_output_bytes,
        ),
    ).subset_from_list(task_ids)


def run(
    model_key: str,
    *,
    template: str,
    debug: bool,
    task_ids: list[str],
    n_parallel: int,
    retry_dir: Path | None,
    toolkit: bool,
    eai_profile: str,
    eai_path: str,
    preemptable: bool,
    sidecar_data: str | None = None,
    assets_data: str | None = None,
    max_actions: int = 100,
    cost_limit: float = 0.5,
    max_output_bytes: int = 8_000,
) -> None:
    infra = _make_infra(toolkit, eai_profile, eai_path, preemptable, sidecar_data, assets_data)

    if debug:
        results = run_debug_suite("terminalbench-cube", _tbench_debug, workers=1, infra=infra)
        failed = [r for r in results if r["error"] or not r["done"] or r["reward"] < 1.0]
        sys.exit(1 if failed else 0)

    agent_config = make_agent_config(model_key, template, max_actions, cost_limit)
    benchmark_config = _make_benchmark_config(task_ids, max_output_bytes)

    output_dir = retry_dir if retry_dir is not None else None
    resume = retry_dir is not None

    infra_label = f"toolkit:{eai_profile}" if toolkit else "local"
    exp = Experiment(
        name=f"genny2-tbench-iter-{model_key}-{template}-{infra_label}",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=infra,
        max_steps=max_actions,
        resume=resume,
    )

    print(f"\n=== genny2-iter | {model_key} | {len(task_ids)} tasks | {template} | {infra_label} ===")

    run_with_ray(exp, n_cpus=n_parallel)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genny2 TerminalBench iteration recipe")
    parser.add_argument("model", nargs="?", default="gpt-5.4-mini", choices=list(MODEL_CONFIGS))
    parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        choices=list(INSTANCE_TEMPLATES),
        help=f"Instance template (default: {DEFAULT_TEMPLATE})",
    )
    parser.add_argument("--debug", action="store_true", help="Run scripted debug agent on oracle tasks (no LLM)")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task IDs (default: ITER_TASKS subset)",
    )
    parser.add_argument("--n-parallel", type=int, default=10)
    parser.add_argument("--retry", metavar="DIR", default=None, help="Resume from output dir")
    parser.add_argument("--toolkit", action="store_true")
    parser.add_argument("--eai-profile", default="yul101")
    parser.add_argument("--eai-path", default="eai")
    parser.add_argument("--preemptable", action="store_true")
    parser.add_argument(
        "--sidecar-data",
        default=None,
        help="EAI data name for the exec-relay sidecar binary (e.g. snow.allac.cube_sidecar)",
    )
    parser.add_argument(
        "--assets-data",
        default=None,
        help="EAI data name for cube-side helper binaries mounted at /opt/cube-assets/ (e.g. snow.allac.cube_uv)",
    )
    parser.add_argument("--max-actions", type=int, default=100)
    parser.add_argument("--cost-limit", type=float, default=0.5)
    parser.add_argument(
        "--max-output-bytes", type=int, default=8_000, help="Max bash output bytes per step (default: 8000)"
    )
    args = parser.parse_args()

    task_ids = [t.strip() for t in args.tasks.split(",")] if args.tasks else ITER_TASKS

    run(
        model_key=args.model,
        template=args.template,
        debug=args.debug,
        task_ids=task_ids,
        n_parallel=args.n_parallel,
        retry_dir=Path(args.retry) if args.retry else None,
        toolkit=args.toolkit,
        eai_profile=args.eai_profile,
        eai_path=args.eai_path,
        preemptable=args.preemptable,
        sidecar_data=args.sidecar_data,
        assets_data=args.assets_data,
        max_actions=args.max_actions,
        cost_limit=args.cost_limit,
        max_output_bytes=args.max_output_bytes,
    )
