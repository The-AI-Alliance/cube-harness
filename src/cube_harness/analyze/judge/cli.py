"""CLI entry point for the trajectory judge (ch-judge)."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from cube_harness.analyze.judge.core import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_FRACTION,
    judge_experiment,
)
from cube_harness.eval_log import JudgeMetadata, JudgeOutput

logger = logging.getLogger(__name__)


def _print_summary_table(results: dict[str, tuple[JudgeOutput, JudgeMetadata]]) -> None:
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ch-judge",
        description="Post-hoc trajectory judge for cube-harness experiments.",
    )
    p.add_argument("path", type=Path, help="Experiment directory or single episode directory.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Judge model (default: {DEFAULT_MODEL}).")

    sel = p.add_argument_group("episode selection (mutually exclusive)")
    g = sel.add_mutually_exclusive_group()
    g.add_argument("--ids", default=None, help="Comma-separated trajectory IDs (or task IDs) to judge exactly.")
    g.add_argument(
        "--sample",
        type=float,
        default=None,
        metavar="FRACTION",
        help=f"Random fraction of eligible episodes (default: {DEFAULT_SAMPLE_FRACTION} when no other selector given).",
    )
    g.add_argument("--n", type=int, default=None, help="Random N eligible episodes.")
    g.add_argument("--all", action="store_true", help="Judge every eligible episode.")

    p.add_argument("--failures-only", action="store_true", help="Restrict to episodes with is_correct=False.")
    p.add_argument("--overwrite", action="store_true", help="Re-judge episodes that already have judge_output.")
    p.add_argument("--seed", type=int, default=None, help="Seed for sampling reproducibility.")
    p.add_argument("--summary", action="store_true", help="Print aggregate blame/outcome distribution.")
    p.add_argument("--n-parallel", type=int, default=1, help="Number of episodes to judge concurrently (default: 1).")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Stream the judge's tool calls and assistant text to stderr while it runs.",
    )
    p.add_argument(
        "--trace",
        choices=["actions", "full", "off"],
        default="actions",
        dest="trace_mode",
        help=(
            "Judge action trace level stored in judge_metadata.judge_actions. "
            "'actions' (default): compact list of (tool, summarised_input). "
            "'full': also includes raw_input dict. "
            "'off': nothing stored."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ids = [s.strip() for s in args.ids.split(",")] if args.ids else None
    sample: float | None
    if args.all:
        sample = 1.0
    elif args.sample is not None:
        sample = args.sample
    elif args.n is not None or ids is not None:
        sample = None
    else:
        sample = DEFAULT_SAMPLE_FRACTION

    results = judge_experiment(
        args.path,
        model=args.model,
        ids=ids,
        sample=sample,
        n=args.n,
        failures_only=args.failures_only,
        overwrite=args.overwrite,
        seed=args.seed,
        verbose=args.verbose,
        n_parallel=args.n_parallel,
        trace_mode=args.trace_mode,
    )

    if args.summary:
        _print_summary_table(results)
    else:
        for tid, (o, m) in results.items():
            print(f"{tid}: {o.outcome.value} / {o.primary_blame.value} (conf={o.primary_blame_confidence})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
