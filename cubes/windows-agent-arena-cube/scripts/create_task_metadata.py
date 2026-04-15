#!/usr/bin/env python3
"""Generate src/waa_cube/task_metadata.json from the WindowsAgentArena repo.

This is a developer tool. Run it after cloning (or updating) the WAA repo
to regenerate the shipped package resource. The output file is committed to
the repository — end users never need to run this script.

Usage:
    python scripts/create_task_metadata.py [--force]

Options:
    --force      Overwrite task_metadata.json even if it already exists.

Requires WAA_EVAL_EXAMPLES_DIR to be set (or pass --eval-dir).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Make the package importable from the cube root without venv activation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cube.task import TaskMetadata

logger = logging.getLogger(__name__)

WAA_EVAL_EXAMPLES_ENV = "WAA_EVAL_EXAMPLES_DIR"
_DEFAULT_OUTPUT = Path(__file__).parent.parent / "src" / "waa_cube" / "task_metadata.json"
_TEST_SETS = ["test_all", "test_small", "test_custom"]


def generate_task_metadata(
    eval_examples_dir: Path,
    output_path: Path = _DEFAULT_OUTPUT,
    *,
    force: bool = False,
) -> int:
    """Parse the WAA evaluation_examples_windows/ dir and write task_metadata.json.

    Returns the number of tasks written (0 if skipped).
    """
    if output_path.exists() and not force:
        logger.info("task_metadata.json already exists at %s — skipping. Pass --force to regenerate.", output_path)
        return 0

    if not eval_examples_dir.exists():
        raise RuntimeError(f"evaluation_examples_windows not found at {eval_examples_dir}")

    # Collect which test sets each task belongs to
    task_sets: dict[str, list[str]] = {}
    task_domains: dict[str, str] = {}

    for set_name in _TEST_SETS:
        set_file = eval_examples_dir / f"{set_name}.json"
        if not set_file.exists():
            logger.warning("Test set file not found: %s", set_file)
            continue
        with open(set_file) as f:
            tasks_by_domain: dict[str, list[str]] = json.load(f)
        for domain_name, task_ids in tasks_by_domain.items():
            for task_id in task_ids:
                task_sets.setdefault(task_id, []).append(set_name)
                task_domains.setdefault(task_id, domain_name)

    # Load each task JSON and build metadata
    all_task_ids = set()
    for set_name in _TEST_SETS:
        set_file = eval_examples_dir / f"{set_name}.json"
        if not set_file.exists():
            continue
        with open(set_file) as f:
            for task_ids in json.load(f).values():
                all_task_ids.update(task_ids)

    metadata: list[dict] = []
    for task_id in sorted(all_task_ids):
        domain = task_domains.get(task_id, "unknown")
        task_file = eval_examples_dir / "examples" / domain / f"{task_id}.json"
        if not task_file.exists():
            logger.warning("Task file not found: %s", task_file)
            continue
        try:
            with open(task_file) as f:
                td = json.load(f)
        except Exception as exc:
            logger.error("Failed to load task %s: %s", task_id, exc)
            continue

        tm = TaskMetadata(
            id=td.get("id", task_id),
            abstract_description=td.get("instruction", ""),
            extra_info={
                "domain": domain,
                "snapshot": td.get("snapshot", "init_state"),
                "config": td.get("config", []),
                "evaluator": td.get("evaluator", {}),
                "related_apps": td.get("related_apps", []),
                "test_sets": task_sets.get(task_id, []),
            },
        )
        metadata.append(tm.model_dump())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Saved %d tasks to %s", len(metadata), output_path)
    return len(metadata)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if file already exists")
    parser.add_argument("--eval-dir", type=str, default=None, help="Path to evaluation_examples_windows/")
    args = parser.parse_args()

    eval_dir = args.eval_dir or os.environ.get(WAA_EVAL_EXAMPLES_ENV)
    if not eval_dir:
        print(f"Error: set {WAA_EVAL_EXAMPLES_ENV} or pass --eval-dir", file=sys.stderr)
        sys.exit(1)

    n = generate_task_metadata(Path(eval_dir), force=args.force)
    if n:
        print(f"Generated task_metadata.json with {n} tasks")
    else:
        print("Skipped (already exists). Use --force to regenerate.")
