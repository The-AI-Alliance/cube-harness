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
from typing import Annotated

import typer
from dotenv import load_dotenv

_project_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_project_env if _project_env.exists() else Path.home() / ".env", override=True)

from cube_harness.agents.genny2_swe_config import (  # noqa: E402
    INSTANCE_TEMPLATES,
    MODEL_CONFIGS,
    make_agent_config,
)
from cube_harness.exp_runner import run_sequentially, run_with_ray  # noqa: E402
from cube_harness.experiment import Experiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

DEFAULT_TBENCH_TEMPLATE = "workflow-tbench"


def _make_infra(toolkit: bool, eai_profile: str, eai_path: str, preemptable: bool, sidecar_data: str | None) -> object:
    if toolkit:
        from cube_infra_toolkit import ToolkitInfraConfig

        return ToolkitInfraConfig(
            profile=eai_profile,
            eai_path=eai_path,
            preemptable=preemptable,
            launch_timeout_seconds=3000,
            sidecar_data=sidecar_data,
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
    from cube.tools.terminal import TerminalToolConfig
    from terminalbench_cube.benchmark import TerminalBenchBenchmarkConfig

    if debug:
        from terminalbench_cube.debug import get_debug_benchmark

        config = get_debug_benchmark()
        if task_ids:
            config = config.subset_from_list(task_ids)
        return config

    TerminalBenchBenchmarkConfig.install()
    config = TerminalBenchBenchmarkConfig(
        tool_config=TerminalToolConfig(working_dir="/app", max_timeout=900, enable_file_actions=True),
        oracle_mode=oracle_mode,
    )
    if difficulty:
        config = config.subset_from_glob("difficulty", difficulty)
    if category:
        config = config.subset_from_glob("category", category)
    if task_ids:
        config = config.subset_from_list(task_ids)
    return config


def main(
    model: Annotated[str, typer.Argument(help="Model key from MODEL_CONFIGS")] = "gpt-5.4-mini",
    template: Annotated[str, typer.Option(help="Instance template variant")] = DEFAULT_TBENCH_TEMPLATE,
    debug: Annotated[bool, typer.Option(help="Run debug oracle tasks sequentially (no LLM)")] = False,
    difficulty: Annotated[str | None, typer.Option(help="Filter by difficulty (easy/medium/hard)")] = None,
    category: Annotated[str | None, typer.Option(help="Filter by category glob")] = None,
    tasks: Annotated[str | None, typer.Option(help="Comma-separated task IDs")] = None,
    oracle_mode: Annotated[bool, typer.Option(help="Upload gold solution at reset")] = False,
    n_parallel: Annotated[int, typer.Option(help="Number of Ray workers")] = 5,
    retry: Annotated[Path | None, typer.Option(help="Resume from output dir")] = None,
    toolkit: Annotated[bool, typer.Option(help="Use EAI Toolkit instead of local Docker")] = False,
    eai_profile: Annotated[str, typer.Option(help="EAI profile")] = "yul101",
    eai_path: Annotated[str, typer.Option(help="Path to eai CLI")] = "eai",
    preemptable: Annotated[bool, typer.Option(help="Request preemptable resources")] = False,
    sidecar_data: Annotated[
        str | None, typer.Option(help="EAI data name for exec-relay sidecar (e.g. snow.allac.cube_sidecar)")
    ] = None,
    max_actions: Annotated[int, typer.Option(help="Max actions per episode")] = 100,
    cost_limit: Annotated[float, typer.Option(help="Cost limit per episode (USD)")] = 1.0,
) -> None:
    """Run Genny2 against Terminal-Bench 2 tasks."""
    if model not in MODEL_CONFIGS:
        raise typer.BadParameter(f"unknown model {model!r}; pick from {sorted(MODEL_CONFIGS)}")
    if template not in INSTANCE_TEMPLATES:
        raise typer.BadParameter(f"unknown template {template!r}; pick from {sorted(INSTANCE_TEMPLATES)}")

    task_ids = [t.strip() for t in tasks.split(",")] if tasks else None
    agent_config = make_agent_config(model, template, max_actions, cost_limit)

    infra = _make_infra(toolkit, eai_profile, eai_path, preemptable, sidecar_data)
    benchmark_config = _make_benchmark_config(debug, difficulty, category, task_ids, oracle_mode)

    infra_label = f"toolkit:{eai_profile}" if toolkit else "local"
    exp = Experiment(
        name=f"genny2-tbench-{model}-{template}-{infra_label}",
        output_dir=retry,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=infra,
        max_steps=max_actions,
        resume=retry is not None,
    )

    print(f"\n=== genny2 | {model} | terminalbench | {template} | {infra_label} ===")

    if debug:
        run_sequentially(exp)
    else:
        run_with_ray(exp, n_cpus=n_parallel)


if __name__ == "__main__":
    typer.run(main)
