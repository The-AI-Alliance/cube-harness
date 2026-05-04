#!/usr/bin/env python3
"""Experiment results reporter.

Scans cube_harness_results/ and prints a markdown summary table.

Usage:
    scripts/report.py                          # all experiments, newest first
    scripts/report.py --match haiku            # filter by substring in dir name
    scripts/report.py --match "haiku.*tw"      # regex match
    scripts/report.py --since 2026-05-03       # on or after date
    scripts/report.py --last 10                # most recent N experiments
    scripts/report.py --results-dir /path/to  # custom results root
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_RESULTS_DIR = Path.home() / "cube_harness_results"


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _parse_experiment(exp_dir: Path) -> dict | None:
    record = _load_json(exp_dir / "experiment_record.json")
    if not record:
        return None

    name = record.get("experiment_name", exp_dir.name)
    benchmark = record.get("benchmark_name", "?")
    # Shorten known benchmark names for display
    benchmark = (
        benchmark.replace("swebench-verified-cube", "sweb-v")
        .replace("workarena", "workarena")
        .replace("miniwob", "miniwob")
    )
    model = record.get("agent", {}).get("llm_model", "?")
    template = record.get("agent", {}).get("config", {}).get("goal_template", "")
    t = template.lower()
    if not template or template.strip() == "{{task}}":
        template_label = "minimal"
    elif "suggested approach" in t and "take time" in t:
        template_label = "thought-workflow"
    elif "suggested approach" in t:
        template_label = "workflow"
    elif "take time" in t:
        template_label = "thought"
    else:
        template_label = template[:30].replace("\n", " ")

    ts = record.get("evaluation_timestamp", 0)
    date = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else "?"

    episodes_dir = exp_dir / "episodes"
    if not episodes_dir.exists():
        return None

    # Canonical status buckets (raw values from status.json)
    # in-flight: RUNNING, QUEUED
    # terminal-valid: COMPLETED, MAX_STEPS_REACHED  (have a reward)
    # terminal-error: FAILED, STALE, CANCELLED      (no valid reward)
    counts: dict[str, int] = {
        "COMPLETED": 0,
        "MAX_STEPS_REACHED": 0,
        "RUNNING": 0,
        "QUEUED": 0,
        "FAILED": 0,
        "STALE": 0,
        "CANCELLED": 0,
    }
    completed_rewards: list[float] = []
    context_window_errors = 0
    total_cost = 0.0
    seen_episode_ids: set[str] = set()

    for ep_dir in episodes_dir.iterdir():
        ep_name = ep_dir.name
        # Skip archived entries (e.g. "ep7.archived_1234")
        if ".archived_" in ep_name:
            continue
        if not ep_dir.is_dir():
            continue

        status_data = _load_json(ep_dir / "status.json")
        if not status_data:
            continue

        ep_id = f"{status_data.get('task_id', ep_name)}_{status_data.get('episode_id', '')}"
        if ep_id in seen_episode_ids:
            continue
        seen_episode_ids.add(ep_id)

        raw_status = status_data.get("status", "UNKNOWN")
        reward = status_data.get("reward")
        error_type = status_data.get("error_type", "")

        if error_type and "ContextWindow" in str(error_type):
            context_window_errors += 1

        if raw_status in counts:
            counts[raw_status] += 1
        else:
            counts["FAILED"] += 1  # unknown terminal status → treat as error

        if raw_status in ("COMPLETED", "MAX_STEPS_REACHED") and reward is not None:
            completed_rewards.append(float(reward))

        rec = _load_json(ep_dir / "episode_record.json")
        total_cost += rec.get("usage", {}).get("total_cost_usd", 0.0)

    n_done = counts["COMPLETED"] + counts["MAX_STEPS_REACHED"]
    n_inflight = counts["RUNNING"] + counts["QUEUED"]
    n_error = counts["FAILED"] + counts["STALE"] + counts["CANCELLED"]
    n_total = n_done + n_inflight + n_error

    if completed_rewards:
        acc = sum(completed_rewards) / len(completed_rewards)
        se = math.sqrt(acc * (1 - acc) / len(completed_rewards)) if len(completed_rewards) > 1 else 0.0
        acc_str = f"{acc:.1%} ±{se:.1%}"
    else:
        acc_str = "—"

    status_str = f"{n_done}/{n_total}"
    if n_inflight:
        inflight_parts = []
        if counts["RUNNING"]:
            inflight_parts.append(f"{counts['RUNNING']}▶")
        if counts["QUEUED"]:
            inflight_parts.append(f"{counts['QUEUED']}🕐")
        status_str += f" +{'+'.join(inflight_parts)}"
    if n_error:
        error_parts = []
        if counts["FAILED"]:
            error_parts.append(f"{counts['FAILED']}✗")
        if counts["STALE"]:
            error_parts.append(f"{counts['STALE']}👻")
        if counts["CANCELLED"]:
            error_parts.append(f"{counts['CANCELLED']}🚫")
        status_str += f" {' '.join(error_parts)}"
    if context_window_errors:
        status_str += f" {context_window_errors}ctx"

    cost_str = f"${total_cost:.2f}" if total_cost else "—"

    return {
        "date": date,
        "ts": ts,
        "name": name,
        "benchmark": benchmark,
        "model": model.split("/")[-1],  # strip provider prefix
        "template": template_label,
        "accuracy": acc_str,
        "episodes": status_str,
        "cost": cost_str,
        "dir": exp_dir.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report cube-harness experiment results")
    parser.add_argument("--match", default=None, help="Regex filter on experiment directory name")
    parser.add_argument("--since", default=None, help="Only experiments on/after YYYY-MM-DD")
    parser.add_argument("--last", type=int, default=None, help="Show only the N most recent")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Results root (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument("--no-header", action="store_true", help="Omit markdown table header")
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    if not results_root.exists():
        print(f"Results dir not found: {results_root}", file=sys.stderr)
        sys.exit(1)

    since_ts = None
    if args.since:
        since_ts = datetime.strptime(args.since, "%Y-%m-%d").timestamp()

    pattern = re.compile(args.match, re.IGNORECASE) if args.match else None

    rows = []
    for exp_dir in sorted(results_root.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue
        if pattern and not pattern.search(exp_dir.name):
            continue

        row = _parse_experiment(exp_dir)
        if row is None:
            continue
        if since_ts and row["ts"] < since_ts:
            continue
        rows.append(row)

    rows.sort(key=lambda r: r["ts"], reverse=True)
    if args.last:
        rows = rows[: args.last]

    if not rows:
        print("No matching experiments found.", file=sys.stderr)
        sys.exit(0)

    col_widths = {
        "date": max(len("date"), max(len(r["date"]) for r in rows)),
        "benchmark": max(len("bench"), max(len(r["benchmark"]) for r in rows)),
        "model": max(len("model"), max(len(r["model"]) for r in rows)),
        "template": max(len("template"), max(len(r["template"]) for r in rows)),
        "accuracy": max(len("accuracy"), max(len(r["accuracy"]) for r in rows)),
        "episodes": max(len("done/total"), max(len(r["episodes"]) for r in rows)),
        "cost": max(len("cost"), max(len(r["cost"]) for r in rows)),
        "name": max(len("experiment"), max(len(r["name"]) for r in rows)),
    }

    def fmt(row: dict) -> str:
        return (
            f"| {row['date']:<{col_widths['date']}} "
            f"| {row['benchmark']:<{col_widths['benchmark']}} "
            f"| {row['model']:<{col_widths['model']}} "
            f"| {row['template']:<{col_widths['template']}} "
            f"| {row['accuracy']:<{col_widths['accuracy']}} "
            f"| {row['episodes']:<{col_widths['episodes']}} "
            f"| {row['cost']:<{col_widths['cost']}} "
            f"| {row['name']:<{col_widths['name']}} |"
        )

    if not args.no_header:
        print(
            "**Legend** — done/total: `+N▶` running · `+N🕐` queued · `N✗` failed · `N👻` stale (dead worker) · `N🚫` cancelled · `Nctx` context-window errors | accuracy ± SE over completed episodes\n"
        )
        header = (
            f"| {'date':<{col_widths['date']}} "
            f"| {'bench':<{col_widths['benchmark']}} "
            f"| {'model':<{col_widths['model']}} "
            f"| {'template':<{col_widths['template']}} "
            f"| {'accuracy':<{col_widths['accuracy']}} "
            f"| {'done/total':<{col_widths['episodes']}} "
            f"| {'cost':<{col_widths['cost']}} "
            f"| {'experiment':<{col_widths['name']}} |"
        )
        sep = (
            f"| {'-' * col_widths['date']} "
            f"| {'-' * col_widths['benchmark']} "
            f"| {'-' * col_widths['model']} "
            f"| {'-' * col_widths['template']} "
            f"| {'-' * col_widths['accuracy']} "
            f"| {'-' * col_widths['episodes']} "
            f"| {'-' * col_widths['cost']} "
            f"| {'-' * col_widths['name']} |"
        )
        print(header)
        print(sep)

    for row in rows:
        print(fmt(row))


if __name__ == "__main__":
    main()
