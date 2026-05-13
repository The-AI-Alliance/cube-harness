"""Regenerate the judge summary + CSV from whatever judge_output records are on disk.

Useful while a long `ch-judge` batch is mid-flight: this tool scans every
`episode_record.json` under the experiment, picks up the ones that already have
a `judge_output`, and rewrites `experiment_judge_summary.json` +
`experiment_judge_report.csv` to reflect the partial state.

CLI: `ch-judge-report <experiment_dir>` (alias for `python -m cube_harness.analyze.judge_report`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cube_harness.analyze.judge import (
    EXPERIMENT_JUDGE_REPORT_FILENAME,
    EXPERIMENT_JUDGE_SUMMARY_FILENAME,
    EpisodeRef,
    _load_episode_record,
    _write_summary,
    discover_episodes,
)

logger = logging.getLogger(__name__)


def regenerate_report(experiment_dir: Path) -> int:
    """Scan `episode_record.json` files for judge_output and rewrite summary + CSV.

    Returns the number of judged episodes found.
    """
    experiment_dir = experiment_dir.resolve()
    refs = discover_episodes(experiment_dir)

    selected: list[EpisodeRef] = []
    results: dict[str, tuple[object, object]] = {}
    for ref in refs:
        if ref.record is None:
            ref.record = _load_episode_record(ref.record_path)
        if ref.record is None or ref.record.judge_output is None:
            continue
        selected.append(ref)
        results[ref.trajectory_id] = (ref.record.judge_output, ref.record.judge_metadata)

    if not results:
        logger.warning("No judged episodes found under %s", experiment_dir)
        return 0

    # Pick the most-recent judge model seen for the summary header.
    model = max(
        (m for _, m in results.values() if m is not None),
        key=lambda m: getattr(m, "timestamp", 0),
        default=None,
    )
    model_name = getattr(model, "model", "unknown") if model is not None else "unknown"

    _write_summary(experiment_dir, selected, results, model=model_name)
    return len(results)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ch-judge-report",
        description="Regenerate experiment_judge_summary.json + experiment_judge_report.csv from disk.",
    )
    p.add_argument("path", type=Path, help="Experiment directory.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )

    n = regenerate_report(args.path)
    if n == 0:
        print("No judged episodes found.", file=sys.stderr)
        return 1
    summary_path = args.path / EXPERIMENT_JUDGE_SUMMARY_FILENAME
    csv_path = args.path / EXPERIMENT_JUDGE_REPORT_FILENAME
    print(f"Regenerated report from {n} judged episodes:")
    print(f"  {summary_path}")
    print(f"  {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
