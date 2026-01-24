#!/usr/bin/env python
"""
Export Terminal-Bench tasks to a local HuggingFace dataset.

Clones the official terminal-bench repo temporarily, runs their export script,
and saves the dataset locally (and optionally pushes to HuggingFace Hub).

Usage:
    # Local only
    uv run scripts/export_terminal_bench.py --outdir ./data/terminal_bench

    # Push to HuggingFace
    uv run scripts/export_terminal_bench.py --outdir ./data/terminal_bench --push --repo-id myuser/terminal-bench
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Terminal-Bench tasks to HF dataset")
    parser.add_argument(
        "--outdir",
        type=str,
        default="./data/terminal_bench",
        help="Output directory for the dataset (default: ./data/terminal_bench)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace repo to push to (e.g., myuser/terminal-bench)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the HuggingFace repo as private",
    )
    args = parser.parse_args()

    if args.push and not args.repo_id:
        print("[ERROR] --repo-id is required when using --push", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir).resolve()

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "terminal-bench"

        # Clone the repo (shallow clone for speed)
        print("[INFO] Cloning laude-institute/terminal-bench...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/laude-institute/terminal-bench.git", str(repo_dir)],
            check=True,
        )

        # Check that original-tasks exists
        tasks_root = repo_dir / "original-tasks"
        if not tasks_root.exists():
            print(f"[ERROR] Tasks directory not found: {tasks_root}", file=sys.stderr)
            sys.exit(1)

        task_count = len(list(tasks_root.iterdir()))
        print(f"[INFO] Found {task_count} tasks in {tasks_root}")

        # Run the export script
        export_script = repo_dir / "scripts_python" / "export_tasks_to_hf_dataset.py"
        if not export_script.exists():
            print(f"[ERROR] Export script not found: {export_script}", file=sys.stderr)
            sys.exit(1)

        print("[INFO] Running export script...")
        cmd = [
            sys.executable,
            str(export_script),
            "--tasks-root",
            str(tasks_root),
            "--outdir",
            str(outdir),
        ]
        if args.push:
            cmd.extend(["--push", "--repo-id", args.repo_id])
        if args.private:
            cmd.append("--private")

        subprocess.run(cmd, check=True)

    print(f"\n[DONE] Dataset saved to: {outdir}")
    print("\nTo load the dataset locally:")
    print(f"  from datasets import load_from_disk")
    print(f"  ds = load_from_disk('{outdir}')")

    if args.push:
        print(f"\nOr load from HuggingFace Hub:")
        print(f"  from datasets import load_dataset")
        print(f"  ds = load_dataset('{args.repo_id}', split='test')")


if __name__ == "__main__":
    main()
