#!/usr/bin/env python
"""
Export Terminal-Bench 2 tasks to a local HuggingFace dataset.

Thin CLI wrapper around TerminalBenchBenchmark.install().

Usage:
    uv run scripts/export_terminal_bench.py
    uv run scripts/export_terminal_bench.py --outdir ~/.agentlab/data/terminal_bench_v2
"""

import argparse

from agentlab2.benchmarks.terminalbench.benchmark import DEFAULT_DATASET_PATH, TerminalBenchBenchmark
from agentlab2.tools.daytona import DaytonaSWEToolConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Terminal-Bench 2 tasks to HF dataset")
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f"Output directory for the dataset (default: {DEFAULT_DATASET_PATH})",
    )
    args = parser.parse_args()

    # Create a minimal benchmark instance just for install()
    # tool_config is required by Benchmark but unused during install
    benchmark = TerminalBenchBenchmark(
        tool_config=DaytonaSWEToolConfig(api_key="unused"),
        dataset_path=args.outdir,
    )
    benchmark.install()


if __name__ == "__main__":
    main()
