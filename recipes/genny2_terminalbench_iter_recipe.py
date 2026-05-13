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
from typing import Annotated

import typer
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


def main(
    model: Annotated[str, typer.Argument(help="Model key from MODEL_CONFIGS")] = "gpt-5.4-mini",
    template: Annotated[str, typer.Option(help="Instance template variant")] = DEFAULT_TEMPLATE,
    debug: Annotated[bool, typer.Option(help="Run scripted debug agent on oracle tasks (no LLM)")] = False,
    tasks: Annotated[str | None, typer.Option(help="Comma-separated task IDs (default: ITER_TASKS subset)")] = None,
    n_parallel: Annotated[int, typer.Option(help="Number of Ray workers")] = 10,
    retry: Annotated[Path | None, typer.Option(help="Resume from output dir")] = None,
    toolkit: Annotated[bool, typer.Option(help="Use EAI Toolkit instead of local Docker")] = False,
    eai_profile: Annotated[str, typer.Option(help="EAI profile")] = "yul101",
    eai_path: Annotated[str, typer.Option(help="Path to eai CLI")] = "eai",
    preemptable: Annotated[bool, typer.Option(help="Request preemptable resources")] = False,
    sidecar_data: Annotated[
        str | None, typer.Option(help="EAI data name for exec-relay sidecar (e.g. snow.allac.cube_sidecar)")
    ] = None,
    assets_data: Annotated[
        str | None, typer.Option(help="EAI data name for cube assets mount (e.g. snow.allac.cube_uv)")
    ] = None,
    max_actions: Annotated[int, typer.Option(help="Max actions per episode")] = 100,
    cost_limit: Annotated[float, typer.Option(help="Cost limit per episode (USD)")] = 0.5,
    max_output_bytes: Annotated[int, typer.Option(help="Max bash output bytes per step")] = 8_000,
) -> None:
    """Iterate on the fixed 40-task medium subset with the generic SWE agent config."""
    if model not in MODEL_CONFIGS:
        raise typer.BadParameter(f"unknown model {model!r}; pick from {sorted(MODEL_CONFIGS)}")
    if template not in INSTANCE_TEMPLATES:
        raise typer.BadParameter(f"unknown template {template!r}; pick from {sorted(INSTANCE_TEMPLATES)}")

    task_ids = [t.strip() for t in tasks.split(",")] if tasks else ITER_TASKS
    infra = _make_infra(toolkit, eai_profile, eai_path, preemptable, sidecar_data, assets_data)

    if debug:
        results = run_debug_suite("terminalbench-cube", _tbench_debug, workers=1, infra=infra)
        failed = [r for r in results if r["error"] or not r["done"] or r["reward"] < 1.0]
        sys.exit(1 if failed else 0)

    agent_config = make_agent_config(model, template, max_actions, cost_limit)
    benchmark_config = _make_benchmark_config(task_ids, max_output_bytes)

    infra_label = f"toolkit:{eai_profile}" if toolkit else "local"
    exp = Experiment(
        name=f"genny2-tbench-iter-{model}-{template}-{infra_label}",
        output_dir=retry,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=infra,
        max_steps=max_actions,
        resume=retry is not None,
    )

    print(f"\n=== genny2-iter | {model} | {len(task_ids)} tasks | {template} | {infra_label} ===")

    run_with_ray(exp, n_cpus=n_parallel)


if __name__ == "__main__":
    typer.run(main)
