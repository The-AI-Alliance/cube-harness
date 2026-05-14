"""Judge self-review recipe — turn every judge signal on against one experiment.

Existing experiment dir → judge it with audit + multi-seed cross-judge agreement
+ post-batch meta-analysis. The point is to surface ALL the signals the judge
emits about itself, so a human can inspect:

  - per-episode `audit.json` — the judge's self-critique (verdict, missed
    evidence, alternative blames),
  - `cross_judge_agreement.csv` — does the judge agree with itself across N
    seeds on the same trajectory?
  - per-episode `judge_trace.json` — what tools did the judge actually use?
  - `judge_context.md` — was the context sub-agent's output any good?
  - `meta_analysis.{json,md}` — the standard synthesis (about the agent under
    test); useful here as a sanity check that the synthesis sees what the
    audits see.

Outputs land in `<experiment_dir>/` and (mirrored) in
`~/cube_meta_agent_journal/<experiment>/`. The human-in-the-loop reviews them
to decide what to change in the judge prompts, the context sub-agent, the
recipe enums, etc.

Cost guideline (Sonnet judge + Opus synthesis, 5-episode reference experiment):
  - audit pass: ~25-30% on top of the per-episode judge cost
  - n_seeds=3: 3x the per-episode judge cost
  - synthesis: one Opus call, ~$0.10
  - At ~$0.10/episode/judge, a 5-episode self-review ≈ $1.50–$2.

Usage:
    uv run meta_agent/recipes/judge_self_review.py <experiment_dir>
    uv run meta_agent/recipes/judge_self_review.py <experiment_dir> --n-seeds 5
    uv run meta_agent/recipes/judge_self_review.py <experiment_dir> --recipe profiling
    uv run meta_agent/recipes/judge_self_review.py <experiment_dir> --ids "tA,tB"
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from cube_harness.analyze.judge import (
    RECIPE_CATALOG,
    ClaudeCodeSDKDriver,
    JudgeBatchConfig,
    judge_experiment,
)

# Load .env for ANTHROPIC_API_KEY (and any LITELLM_PROXY_URL configured).
_project_env = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_project_env if _project_env.exists() else Path.home() / ".env", override=True)


def main(
    experiment_dir: Annotated[Path, typer.Argument(help="Reference experiment directory to self-review.")],
    recipe: Annotated[
        str, typer.Option(help="Recipe name. Default `general_blame` exercises the full taxonomy.")
    ] = "general_blame",
    n_seeds: Annotated[
        int, typer.Option("--n-seeds", help="Judge each episode this many times for cross-judge agreement.")
    ] = 3,
    n_parallel: Annotated[
        int, typer.Option("--n-parallel", help="Concurrent judge sub-processes. Sonnet handles ~8 comfortably.")
    ] = 4,
    ids: Annotated[
        str | None,
        typer.Option(help="Comma-separated trajectory IDs to scope the review. Default: all eligible episodes."),
    ] = None,
    judge_model: Annotated[str | None, typer.Option("--judge-model", help="Override the recipe's judge model.")] = None,
    synthesis_model: Annotated[
        str, typer.Option("--synthesis-model", help="Model for the post-batch meta-analysis.")
    ] = "claude-opus-4-7",
    journal_dir: Annotated[
        Path,
        typer.Option("--journal-dir", help="Mirror meta_analysis.{json,md} into <journal-dir>/<experiment>/."),
    ] = Path("~/cube_meta_agent_journal").expanduser(),
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Run the judge against `experiment_dir` with every signal turned on."""
    chosen = RECIPE_CATALOG[recipe]
    if judge_model is not None and judge_model != chosen.model:
        chosen = chosen.model_copy(update={"model": judge_model})

    config = JudgeBatchConfig(
        recipe=chosen,
        driver=ClaudeCodeSDKDriver(),
        # Self-review knobs:
        audit=True,
        n_seeds=n_seeds,
        # Selection / execution:
        ids=[s.strip() for s in ids.split(",")] if ids else None,
        overwrite=True,  # always re-judge so the audit + seeds are fresh
        n_parallel=n_parallel,
        verbose=verbose,
        synthesis_model=synthesis_model,
        journal_dir=journal_dir,
    )

    typer.echo(
        f"judge_self_review — recipe={chosen.name} judge_model={chosen.model} "
        f"synthesis_model={synthesis_model} n_seeds={n_seeds} audit=on overwrite=on"
    )
    typer.echo(f"Experiment: {experiment_dir}")
    typer.echo(f"Journal mirror: {journal_dir}")
    typer.echo()

    results = judge_experiment(experiment_dir, config)

    typer.echo()
    typer.echo(f"Judged {len(results)} episode(s). Inspect:")
    typer.echo(f"  - {experiment_dir}/episodes/<id>/audit.json")
    typer.echo(f"  - {experiment_dir}/episodes/<id>/judge_trace.json")
    typer.echo(f"  - {experiment_dir}/judge_context.md")
    if n_seeds > 1:
        typer.echo(f"  - {experiment_dir}/cross_judge_agreement.csv")
    typer.echo(f"  - {experiment_dir}/meta_analysis.md  (and {journal_dir}/{experiment_dir.name}/)")


if __name__ == "__main__":
    typer.run(main)
