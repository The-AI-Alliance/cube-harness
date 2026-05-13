"""Genny2 SWE recipe — swebench-verified, swebench-live, and terminalbench.

Single entry point for Genny2 against any of the SWE-like cube benchmarks. Agent
construction is delegated to ``cube_harness.agents.genny2_swe_config``.

Usage:
    # Debug oracle tasks (no LLM, sequential):
    .venv/bin/python recipes/swe_agent_recipe.py --debug
    .venv/bin/python recipes/swe_agent_recipe.py --benchmark live --debug
    .venv/bin/python recipes/swe_agent_recipe.py --benchmark tbench --debug

    # Full swebench-verified on Toolkit:
    .venv/bin/python recipes/swe_agent_recipe.py --toolkit --eai-path ~/bin/eai --n-parallel 20

    # HAL-50 subset on swebench-verified:
    .venv/bin/python recipes/swe_agent_recipe.py --subset hal_mini --toolkit ...

    # SWE-bench Live, golden 30, Daytona:
    .venv/bin/python recipes/swe_agent_recipe.py --benchmark live --subset live-golden-30 --daytona

    # TerminalBench, fixed 40-task iteration subset, on Toolkit with sidecar + uv mount:
    .venv/bin/python recipes/swe_agent_recipe.py --benchmark tbench --subset tbench-iter-40 \\
        --toolkit --eai-profile yul101 \\
        --sidecar-data snow.allac.cube_sidecar --assets-data snow.allac.cube_uv
"""

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
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

# Fixed 40-task TerminalBench medium subset — proportional across the 14 categories.
# Keep stable across iterations so scores stay directly comparable across runs.
_TBENCH_ITER_40: frozenset[str] = frozenset(
    [
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
    sidecar_data: str | None = None,
    assets_data: str | None = None,
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
            sidecar_data=sidecar_data,
            assets_data=assets_data,
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
        task_ids = _load_solvable_task_ids(solvable_from)
    if subset == "solvable-lite":
        # Gold-patch-confirmed subset of lite; resource bundled in the cube package.
        from importlib.resources import files

        snapshot = files("swebench_live_cube").joinpath("lite_solvable_2026-05-12.json")
        task_ids = _load_solvable_task_ids(Path(str(snapshot)))
    elif subset == "live-golden-30":
        from swebench_live_cube.gold_patch.recipe import _LIVE_GOLDEN_30

        cfg = cfg.subset_from_list(list(_LIVE_GOLDEN_30))
    elif subset:
        cfg = cfg.named_subset(subset)
    if task_ids:
        cfg = cfg.subset_from_list(task_ids)
    elif n_tasks:
        cfg = cfg.subset_from_list(list(cfg.task_metadata.keys())[:n_tasks])
    return cfg


def _make_tbench_benchmark(
    *,
    debug: bool,
    task_ids: list[str] | None,
    subset: str | None,
    oracle_mode: bool,
    max_output_bytes: int,
) -> object:
    from cube.tools.terminal import TerminalToolConfig
    from terminalbench_cube.benchmark import TerminalBenchBenchmarkConfig

    if debug:
        from terminalbench_cube.debug import get_debug_benchmark

        return get_debug_benchmark()

    TerminalBenchBenchmarkConfig.install()
    cfg = TerminalBenchBenchmarkConfig(
        tool_config=TerminalToolConfig(
            working_dir="/app",
            max_timeout=900,
            enable_file_actions=True,
            max_output_bytes=max_output_bytes,
        ),
        oracle_mode=oracle_mode,
    )
    if subset == "tbench-iter-40":
        cfg = cfg.subset_from_list(list(_TBENCH_ITER_40))
    elif subset:
        cfg = cfg.named_subset(subset)
    if task_ids:
        cfg = cfg.subset_from_list(task_ids)
    return cfg


def _load_solvable_task_ids(path: Path) -> list[str]:
    """Read either a bare list (legacy) or the metadata-wrapped schema.

    Modern schema (preferred): ``{"date", "source_set", "n_tasks", "task_ids": [...]}``.
    Legacy: a bare ``["task_id_1", ...]`` JSON list. Both are accepted so older
    ``--solvable-from path.json`` dumps from ``gold_patch.recipe --dump-solvable``
    keep working.
    """
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data["task_ids"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    model: Annotated[str, typer.Argument(help="Model key from MODEL_CONFIGS")] = "gpt-5.4-mini",
    benchmark: Annotated[str, typer.Option(help="SWE benchmark: verified | live | tbench")] = "verified",
    template: Annotated[str, typer.Option(help="Instance template variant")] = DEFAULT_TEMPLATE,
    debug: Annotated[bool, typer.Option(help="Run cube's debug oracle tasks sequentially (no LLM)")] = False,
    tasks: Annotated[str | None, typer.Option(help="Comma-separated task IDs")] = None,
    subset: Annotated[
        str | None,
        typer.Option(
            help="Named subset. Verified: hal_mini, or any named subset on the cube. "
            "Live: live-golden-30, solvable-lite, lite, verified, full, test. "
            "Tbench: tbench-iter-40, or any difficulty/category subset on the cube."
        ),
    ] = None,
    n_tasks: Annotated[int | None, typer.Option(help="Take first N tasks from subset (live only)")] = None,
    n_parallel: Annotated[int, typer.Option(help="Number of Ray workers")] = 5,
    retry: Annotated[Path | None, typer.Option(help="Resume from output dir")] = None,
    solvable_from: Annotated[
        Path | None,
        typer.Option(
            help="JSON file of solvable task IDs (e.g. output of gold_patch.recipe --dump-solvable). Live only."
        ),
    ] = None,
    oracle_mode: Annotated[bool, typer.Option(help="Upload gold solution at reset (tbench/verified)")] = False,
    toolkit: Annotated[bool, typer.Option(help="Use EAI Toolkit infra")] = False,
    daytona: Annotated[bool, typer.Option(help="Use Daytona infra (requires DAYTONA_API_KEY)")] = False,
    eai_profile: Annotated[str, typer.Option(help="EAI profile")] = "yul101",
    eai_path: Annotated[str, typer.Option(help="Path to eai CLI")] = "eai",
    preemptable: Annotated[bool, typer.Option(help="Request preemptable resources")] = False,
    sidecar_data: Annotated[
        str | None, typer.Option(help="EAI data name for exec-relay sidecar (tbench/verified on Toolkit)")
    ] = None,
    assets_data: Annotated[
        str | None, typer.Option(help="EAI data name for cube assets mount, e.g. snow.allac.cube_uv (tbench)")
    ] = None,
    max_actions: Annotated[int, typer.Option(help="Max actions per episode")] = 150,
    cost_limit: Annotated[float, typer.Option(help="Cost limit per episode (USD)")] = 1.0,
    max_output_bytes: Annotated[int, typer.Option(help="Max bash output bytes per step (tbench)")] = 8_000,
    launch_timeout: Annotated[int, typer.Option(help="Container launch timeout (seconds)")] = 900,
) -> None:
    """Run Genny2 against an SWE-like cube benchmark."""
    if model not in MODEL_CONFIGS:
        raise typer.BadParameter(f"unknown model {model!r}; pick from {sorted(MODEL_CONFIGS)}")
    if template not in INSTANCE_TEMPLATES:
        raise typer.BadParameter(f"unknown template {template!r}; pick from {sorted(INSTANCE_TEMPLATES)}")
    if benchmark not in {"verified", "live", "tbench"}:
        raise typer.BadParameter(f"unknown benchmark {benchmark!r}; pick from verified | live | tbench")

    agent_config = make_agent_config(model, template, max_actions, cost_limit)
    infra = _make_infra(
        toolkit=toolkit,
        daytona=daytona,
        eai_profile=eai_profile,
        eai_path=eai_path,
        preemptable=preemptable,
        launch_timeout=launch_timeout,
        sidecar_data=sidecar_data,
        assets_data=assets_data,
    )

    task_ids = [t.strip() for t in tasks.split(",")] if tasks else None
    if benchmark == "verified":
        benchmark_config = _make_verified_benchmark(debug=debug, task_ids=task_ids, subset=subset)
    elif benchmark == "live":
        benchmark_config = _make_live_benchmark(
            debug=debug,
            task_ids=task_ids,
            subset=subset,
            n_tasks=n_tasks,
            solvable_from=solvable_from,
        )
    else:
        benchmark_config = _make_tbench_benchmark(
            debug=debug,
            task_ids=task_ids,
            subset=subset,
            oracle_mode=oracle_mode,
            max_output_bytes=max_output_bytes,
        )

    infra_label = "daytona" if daytona else (f"toolkit:{eai_profile}" if toolkit else "local")
    exp = Experiment(
        name=f"genny2-swe-{benchmark}-{model}-{infra_label}",
        output_dir=retry,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=infra,
        max_steps=max_actions,
        resume=retry is not None,
    )

    print(f"\n=== genny2 | swe-{benchmark} | {model} | {template} | {infra_label} | subset={subset or 'all'} ===")

    if debug:
        run_sequentially(exp)
    else:
        run_with_ray(exp, n_cpus=n_parallel)


if __name__ == "__main__":
    typer.run(main)
