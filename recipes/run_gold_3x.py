"""Orchestrate 3 gold-patch baseline runs and produce stable/flaky solvable subsets.

Runs the gold baseline N times (default 3) sequentially on a given subset,
then intersects the results to produce:
  - solvable_<subset>_stable.json  — tasks that resolved in ALL N runs
  - solvable_<subset>_flaky.json   — tasks that resolved in SOME but not all runs

Can also incorporate a pre-existing run via --existing-run-dir, useful when
run 1 is already in progress.

Usage:
    # 3 fresh runs on lite:
    .venv/bin/python recipes/run_gold_3x.py --subset lite \\
        --toolkit --eai-profile yul101 --eai-path ~/bin/eai

    # Incorporate an existing run 1, do 2 more:
    .venv/bin/python recipes/run_gold_3x.py --subset lite \\
        --toolkit --eai-profile yul101 --eai-path ~/bin/eai \\
        --existing-run-dir /path/to/run1_dir --n-runs 2

    # Post-hoc: intersect 3 already-completed run dirs:
    .venv/bin/python recipes/run_gold_3x.py \\
        --from-runs dir1 dir2 dir3 --out solvable_lite_stable.json
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cubes/swebench-live-cube/src"))

from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")
_project_env = Path(__file__).resolve().parents[1] / ".env"
if _project_env.exists():
    load_dotenv(_project_env, override=True)

from recipes.gold_patch_baseline_recipe import _run_once, _write_solvable_results  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="3× gold baseline — stable/flaky solvable subset")
    parser.add_argument("--subset", default="lite", choices=["live30", "lite", "verified", "full", "test"])
    parser.add_argument("--n-runs", type=int, default=3, help="Number of fresh gold runs to execute (default: 3)")
    parser.add_argument(
        "--existing-run-dir",
        metavar="DIR",
        default=None,
        help="Pre-existing gold run dir to include (counts as run 1, reduces fresh runs needed)",
    )
    parser.add_argument(
        "--from-runs",
        metavar="DIR",
        nargs="+",
        default=None,
        help="Skip running; intersect these completed run dirs directly",
    )
    parser.add_argument("--out", metavar="PATH", default=None, help="Output path for stable JSON (default: auto)")
    parser.add_argument("--n-parallel", type=int, default=50)
    parser.add_argument("--launch-timeout", type=int, default=900)
    parser.add_argument("--toolkit", action="store_true")
    parser.add_argument("--eai-profile", default="yul101")
    parser.add_argument("--eai-path", default="eai")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(f"solvable_{args.subset}_stable.json")
    recipes_dir = Path(__file__).resolve().parent
    if not out_path.is_absolute():
        out_path = recipes_dir.parent / out_path

    if args.from_runs:
        run_dirs = [Path(d) for d in args.from_runs]
        print(f"Post-hoc intersection of {len(run_dirs)} run dirs → {out_path}")
        _write_solvable_results(run_dirs, out_path, total_tasks=0)
        return

    run_dirs: list[Path] = []
    if args.existing_run_dir:
        existing = Path(args.existing_run_dir)
        print(f"Including existing run dir: {existing}")
        run_dirs.append(existing)

    runs_needed = args.n_runs - len(run_dirs)
    total_runs = args.n_runs
    common = dict(
        debug=False,
        task_ids=None,
        subset=args.subset,
        n_tasks=None,
        n_parallel=args.n_parallel,
        toolkit=args.toolkit,
        eai_profile=args.eai_profile,
        eai_path=args.eai_path,
        launch_timeout=args.launch_timeout,
    )

    last_total = 0
    for i in range(runs_needed):
        run_num = len(run_dirs) + 1
        label = f"run {run_num}/{total_runs}"
        print(f"\n{'=' * 60}\n  Gold pass {label}\n{'=' * 60}")
        run_dir, result = _run_once(**common, run_label=label)
        run_dirs.append(run_dir)
        last_total = result.tasks_num

    _write_solvable_results(run_dirs, out_path, total_tasks=last_total)


if __name__ == "__main__":
    main()
