"""CLI entry point for the trajectory judge (`ch-judge`).

Two subcommands:
- `ch-judge run <experiment_dir> [options]` — batch judge episodes.
- `ch-judge init-context <experiment_dir>` — (re)generate `judge_context.md`
  via the benchmark-context sub-agent.

`ch-judge <experiment_dir>` (no subcommand) is also accepted — it dispatches to
`run` so existing scripts keep working.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from cube_harness.analyze.judge.benchmark_context_agent import generate_context_file
from cube_harness.analyze.judge.core import (
    DEFAULT_SAMPLE_FRACTION,
    JudgeBatchConfig,
    judge_experiment,
)
from cube_harness.analyze.judge.driver import AgentDriver, ClaudeCodeSDKDriver, TerminalClaudeDriver
from cube_harness.analyze.judge.recipe import JudgeRecipe
from cube_harness.analyze.judge.use_cases import RECIPE_CATALOG
from cube_harness.eval_log import BaseJudgeOutput, JudgeMetadata

logger = logging.getLogger(__name__)


def _print_summary_table(results: dict[str, tuple[BaseJudgeOutput, JudgeMetadata]]) -> None:
    if not results:
        print("(no episodes judged)")
        return
    print(f"\n{'trajectory_id':50s}  {'outcome':25s}  {'blame':24s}  {'conf':4s}  {'h_conf':6s}")
    print(f"{'-' * 50}  {'-' * 25}  {'-' * 24}  {'-' * 4}  {'-' * 6}")
    for tid, (o, _) in results.items():
        print(
            f"{tid[:50]:50s}  {o.outcome.value:25s}  {o.primary_blame.value:24s}  "
            f"{o.primary_blame_confidence:4d}  {o.hypothesis_confidence:6d}"
        )

    outcomes = Counter(o.outcome.value for o, _ in results.values())
    blames = Counter(o.primary_blame.value for o, _ in results.values())
    total_cost = sum(m.cost_usd for _, m in results.values())
    n = len(results)
    print(f"\nJudged {n} episodes  |  total cost: ${total_cost:.2f}  |  avg: ${total_cost / n:.3f}/ep")
    print(f"Outcomes:       {dict(outcomes)}")
    print(f"Primary blame:  {dict(blames)}")


def _resolve_recipe(name: str) -> JudgeRecipe:
    """Look up a recipe by name from the catalog."""
    if name not in RECIPE_CATALOG:
        known = ", ".join(sorted(RECIPE_CATALOG))
        raise typer.BadParameter(f"unknown recipe {name!r} — known: {known}")
    return RECIPE_CATALOG[name]


def _make_driver(name: str) -> AgentDriver:
    """Build a concrete driver from a CLI flag value."""
    if name == "claude-code-sdk":
        return ClaudeCodeSDKDriver()
    if name == "claude-terminal":
        return TerminalClaudeDriver()
    raise typer.BadParameter(f"unknown driver {name!r} — choose one of: claude-code-sdk, claude-terminal")


def _resolve_sample(
    *,
    all_eps: bool,
    sample: float | None,
    n: int | None,
    ids: list[str] | None,
) -> float | None:
    """Apply the same sample-fraction defaults the old argparse path used."""
    if all_eps:
        return 1.0
    if sample is not None:
        return sample
    if n is not None or ids is not None:
        return None
    return DEFAULT_SAMPLE_FRACTION


def _run(
    *,
    path: Path,
    recipe: str,
    driver: str,
    audit: bool,
    n_seeds: int,
    ids: str | None,
    sample: float | None,
    n: int | None,
    all_eps: bool,
    failures_only: bool,
    overwrite: bool,
    seed: int | None,
    summary_only: bool,
    n_parallel: int,
    verbose: bool,
    trace_mode: str,
    model: str | None,
    synthesize: bool,
    synthesis_model: str,
    journal_dir: Path | None,
) -> None:
    """Shared implementation between the root command and the `run` subcommand."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    chosen_recipe = _resolve_recipe(recipe)
    chosen_driver = _make_driver(driver)

    if model is not None:
        warnings.warn(
            "`--model` is deprecated — set `model` on the recipe instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    id_list = [s.strip() for s in ids.split(",")] if ids else None
    effective_sample = _resolve_sample(all_eps=all_eps, sample=sample, n=n, ids=id_list)

    if trace_mode not in ("actions", "full", "off"):
        raise typer.BadParameter(f"--trace must be one of: actions, full, off (got {trace_mode!r})")

    # Honour the deprecated --model flag by overriding the chosen recipe's model.
    if model is not None and model != chosen_recipe.model:
        chosen_recipe = chosen_recipe.model_copy(update={"model": model})

    config = JudgeBatchConfig(
        recipe=chosen_recipe,
        driver=chosen_driver,
        audit=audit,
        n_seeds=n_seeds,
        ids=id_list,
        sample=effective_sample,
        n=n,
        failures_only=failures_only,
        overwrite=overwrite,
        seed=seed,
        verbose=verbose,
        n_parallel=n_parallel,
        trace_mode=trace_mode,  # type: ignore[arg-type]
        synthesize=synthesize,
        synthesis_model=synthesis_model,
        journal_dir=journal_dir,
    )
    results = judge_experiment(path, config)

    if summary_only:
        _print_summary_table(results)
    else:
        for tid, (o, _) in results.items():
            print(f"{tid}: {o.outcome.value} / {o.primary_blame.value} (conf={o.primary_blame_confidence})")


app = typer.Typer(
    add_completion=False,
    help="Post-hoc trajectory judge for cube-harness experiments.",
    no_args_is_help=True,
)


@app.command("run")
def run_cmd(
    path: Annotated[Path, typer.Argument(help="Experiment directory or single episode directory.")],
    recipe: Annotated[str, typer.Option(help="Recipe name (see use_cases/ subdirectories).")] = "general_blame",
    driver: Annotated[
        str,
        typer.Option(help="Coding-agent driver: 'claude-code-sdk' (API key) or 'claude-terminal' (subscription)."),
    ] = "claude-code-sdk",
    audit: Annotated[bool, typer.Option(help="Run the audit pass after each judgment (writes audit.json).")] = False,
    n_seeds: Annotated[int, typer.Option("--n-seeds", help="Judge each episode this many times.")] = 1,
    ids: Annotated[str | None, typer.Option(help="Comma-separated trajectory IDs (or task IDs) to judge.")] = None,
    sample: Annotated[float | None, typer.Option(help="Random fraction of eligible episodes.")] = None,
    n: Annotated[int | None, typer.Option(help="Random N eligible episodes.")] = None,
    all_eps: Annotated[bool, typer.Option("--all", help="Judge every eligible episode.")] = False,
    failures_only: Annotated[bool, typer.Option("--failures-only", help="Restrict to is_correct=False.")] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Re-judge episodes that already have judge_output.")
    ] = False,
    seed: Annotated[int | None, typer.Option(help="Seed for sampling reproducibility.")] = None,
    summary_only: Annotated[bool, typer.Option("--summary", help="Print aggregate distribution at end.")] = False,
    n_parallel: Annotated[int, typer.Option("--n-parallel", help="Concurrent judge sub-processes.")] = 1,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Stream tool calls + text to stderr.")] = False,
    trace_mode: Annotated[
        str, typer.Option("--trace", help="Judge action trace level: actions, full, off.")
    ] = "actions",
    model: Annotated[
        str | None, typer.Option(help="DEPRECATED — set on the recipe. Triggers a deprecation warning.")
    ] = None,
    synthesize: Annotated[
        bool, typer.Option("--synthesize/--no-synthesize", help="Run the meta-analysis sub-agent after the batch.")
    ] = True,
    synthesis_model: Annotated[
        str, typer.Option("--synthesis-model", help="Model for the meta-analysis sub-agent. Opus by default.")
    ] = "claude-opus-4-7",
    journal_dir: Annotated[
        Path | None,
        typer.Option(
            "--journal-dir",
            help="Mirror meta_analysis.{json,md} into <journal-dir>/<experiment>/ — defaults to ~/cube_meta_agent_journal.",
        ),
    ] = Path("~/cube_meta_agent_journal").expanduser(),
    no_journal: Annotated[
        bool, typer.Option("--no-journal", help="Skip the journal mirror even when --journal-dir is set.")
    ] = False,
) -> None:
    """Batch judge episodes in an experiment directory."""
    _run(
        path=path,
        recipe=recipe,
        driver=driver,
        audit=audit,
        n_seeds=n_seeds,
        ids=ids,
        sample=sample,
        n=n,
        all_eps=all_eps,
        failures_only=failures_only,
        overwrite=overwrite,
        seed=seed,
        summary_only=summary_only,
        n_parallel=n_parallel,
        verbose=verbose,
        trace_mode=trace_mode,
        model=model,
        synthesize=synthesize,
        synthesis_model=synthesis_model,
        journal_dir=None if no_journal else journal_dir,
    )


@app.command("init-context")
def init_context_cmd(
    experiment_dir: Annotated[Path, typer.Argument(help="Experiment directory.")],
    driver: Annotated[str, typer.Option(help="Driver to invoke the context agent through.")] = "claude-code-sdk",
    model: Annotated[str, typer.Option(help="Model name for the context agent.")] = "claude-opus-4-7",
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Invoke the benchmark-context sub-agent to (re)generate judge_context.md."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    chosen_driver = _make_driver(driver)
    path = asyncio.run(
        generate_context_file(experiment_dir, driver=chosen_driver, model=model, verbose=verbose),
    )
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point.

    Back-compat: when argv is `[<path>, ...flags...]` (i.e. no subcommand
    keyword), we dispatch to the `run` subcommand by prepending its name. This
    keeps `ch-judge <experiment_dir> --recipe X` working.
    """
    if argv is None:
        argv = sys.argv[1:]
    # Distinguish a subcommand from a positional path: if the first arg matches a
    # known subcommand name, leave argv alone; otherwise prepend `run`.
    subcommand_names = {"run", "init-context", "--help", "-h"}
    if argv and argv[0] not in subcommand_names:
        argv = ["run", *argv]
    try:
        app(args=argv, standalone_mode=False)
    except typer.Exit as e:
        return int(e.exit_code or 0)
    except SystemExit as e:
        return int(e.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
