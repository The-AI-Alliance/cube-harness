#!/usr/bin/env python
"""
Export Terminal-Bench 2 tasks to a local HuggingFace dataset.

Clones the official terminal-bench-2 repo temporarily, processes tasks,
and saves the dataset locally.

Usage:
    uv run scripts/export_terminal_bench.py --outdir ./data/terminal_bench_v2
"""

import argparse
import io
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path

from datasets import Dataset


def create_task_archive(task_dir: Path) -> bytes:
    """Create a tar.gz archive of the task directory."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for item in task_dir.rglob("*"):
            if item.is_file():
                arcname = str(item.relative_to(task_dir))
                tar.add(item, arcname=arcname)
    return buffer.getvalue()


def load_task(task_dir: Path) -> dict | None:
    """Load a single task from a directory."""
    task_toml = task_dir / "task.toml"
    instruction_md = task_dir / "instruction.md"

    if not task_toml.exists() or not instruction_md.exists():
        return None

    # Parse task.toml
    with open(task_toml, "rb") as f:
        config = tomllib.load(f)

    # Read instruction
    instruction = instruction_md.read_text(encoding="utf-8").strip()

    # Extract metadata
    metadata = config.get("metadata", {})
    env_config = config.get("environment", {})
    agent_config = config.get("agent", {})
    verifier_config = config.get("verifier", {})

    # Create archive of task files
    archive = create_task_archive(task_dir)

    return {
        "task_id": task_dir.name,
        "base_description": instruction,
        "archive": archive,
        "difficulty": metadata.get("difficulty", "unknown"),
        "category": metadata.get("category", ""),
        "tags": metadata.get("tags", []),
        "docker_image": env_config.get("docker_image", "python:3.13"),
        "cpus": env_config.get("cpus", 1),
        "memory": env_config.get("memory", "4G"),
        "storage": env_config.get("storage", "10G"),
        "max_agent_timeout_sec": int(agent_config.get("timeout_sec", 900)),
        "max_test_timeout_sec": int(verifier_config.get("timeout_sec", 900)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Terminal-Bench 2 tasks to HF dataset")
    parser.add_argument(
        "--outdir",
        type=str,
        default="./data/terminal_bench_v2",
        help="Output directory for the dataset",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "terminal-bench-2"

        # Clone the repo
        print("[INFO] Cloning laude-institute/terminal-bench-2...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/laude-institute/terminal-bench-2.git", str(repo_dir)],
            check=True,
        )

        # Find all task directories (directories with task.toml)
        tasks = []
        for item in sorted(repo_dir.iterdir()):
            if item.is_dir() and (item / "task.toml").exists():
                task = load_task(item)
                if task:
                    tasks.append(task)
                    print(f"  Loaded task: {task['task_id']} ({task['difficulty']}, image: {task['docker_image']})")

        print(f"\n[INFO] Loaded {len(tasks)} tasks")

        # Create dataset
        ds = Dataset.from_list(tasks)
        outdir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(outdir))

    print(f"\n[DONE] Dataset saved to: {outdir}")
    print("\nTo load the dataset:")
    print("  from datasets import load_from_disk")
    print(f"  ds = load_from_disk('{outdir}')")


if __name__ == "__main__":
    main()
