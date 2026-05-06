"""Aggregate judge blame counts across experiments grouped by modality.

Generates the data table behind the "Blame distribution by modality" stacked-bar
figure (Web / CUA / SWE × 10-category taxonomy). Auto-detects modality from the
benchmark cube name in each experiment dir, walks every `experiment_judge_summary.json`
under the given root(s), aggregates blame counts per modality, and prints both a
CSV (one row per modality, one column per blame category) and a tidy long-form
table for plotting.

Usage:
    python scripts/blame_by_modality.py ~/cube_harness_results
    python scripts/blame_by_modality.py ~/cube_harness_results --csv blame.csv

Modality classification:
    swebench / terminalbench → SWE
    osworld / computer-use → CUA
    workarena / webarena / miniwob / browsergym → Web
    other → "other" (still emitted so nothing is silently dropped)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Order matches the figure legend (Agent → Tool → Bench → none).
BLAME_CATEGORIES: list[str] = [
    "model_capability",
    "agent_scaffolding",
    "submission_format",
    "tool_failure",
    "action_space_limited",
    "insufficient_observation",
    "env_failure",
    "task_unclear",
    "eval_brittle",
    "none",
]

MODALITY_ORDER: list[str] = ["Web", "CUA", "SWE", "other"]


def classify_modality(experiment_name: str) -> str:
    """Map an experiment dir name to one of {Web, CUA, SWE, other}.

    Matches on cube/benchmark substrings; first match wins.
    """
    name = experiment_name.lower()
    if "swebench" in name or "terminalbench" in name:
        return "SWE"
    if "osworld" in name or "computer-use" in name or "computeruse" in name:
        return "CUA"
    if "workarena" in name or "webarena" in name or "miniwob" in name or "browsergym" in name:
        return "Web"
    return "other"


def collect(roots: list[Path]) -> dict[str, Counter]:
    """Walk roots for `experiment_judge_summary.json` and aggregate per modality."""
    per_modality: dict[str, Counter] = defaultdict(Counter)
    for root in roots:
        for summary_path in root.rglob("experiment_judge_summary.json"):
            exp_dir = summary_path.parent
            modality = classify_modality(exp_dir.name)
            try:
                summary = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Skipping %s: %s", summary_path, e)
                continue
            blame_counts = summary.get("primary_blame", {})
            for cat, n in blame_counts.items():
                per_modality[modality][cat] += n
            logger.info(
                "Counted %s (%s, n_judged=%d) → %s",
                exp_dir.name,
                modality,
                summary.get("n_judged", 0),
                dict(blame_counts),
            )
    return per_modality


def write_wide_csv(per_modality: dict[str, Counter], path: Path | None) -> None:
    """One row per modality, one column per blame category (order: BLAME_CATEGORIES)."""
    fields = ["modality", "n_total", *BLAME_CATEGORIES]
    out = sys.stdout if path is None else path.open("w", newline="")
    try:
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()
        for mod in MODALITY_ORDER:
            counts = per_modality.get(mod)
            if not counts:
                continue
            row = {"modality": mod, "n_total": sum(counts.values())}
            row.update({c: counts.get(c, 0) for c in BLAME_CATEGORIES})
            w.writerow(row)
    finally:
        if path is not None:
            out.close()
            print(f"Wide CSV → {path}", file=sys.stderr)


def print_long_form(per_modality: dict[str, Counter]) -> None:
    """Tidy long-form table to stdout: modality, blame, count, pct."""
    print(f"\n{'modality':10s}  {'blame':25s}  {'count':>5s}  {'pct':>5s}")
    print("-" * 55)
    for mod in MODALITY_ORDER:
        counts = per_modality.get(mod)
        if not counts:
            continue
        total = sum(counts.values())
        for cat in BLAME_CATEGORIES:
            n = counts.get(cat, 0)
            if n == 0:
                continue
            print(f"{mod:10s}  {cat:25s}  {n:5d}  {100 * n / total:4.1f}%")
        print(f"{mod:10s}  {'TOTAL':25s}  {total:5d}")
        print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Aggregate judge blame counts by modality (Web/CUA/SWE) across experiments."
    )
    p.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="One or more directories containing cube-harness experiment folders.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write the wide-format CSV to this path (default: stdout).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )

    per_modality = collect(args.roots)
    if not per_modality:
        print("No experiment_judge_summary.json files found.", file=sys.stderr)
        return 1

    write_wide_csv(per_modality, args.csv)
    print_long_form(per_modality)
    return 0


if __name__ == "__main__":
    sys.exit(main())
