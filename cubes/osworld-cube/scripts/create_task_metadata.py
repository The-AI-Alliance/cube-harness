#!/usr/bin/env python3
"""Generate src/osworld_cube/task_metadata.json from the OSWorld repo.

This is a developer tool. Run it after cloning (or updating) the OSWorld repo
to regenerate the shipped package resource. The output file is committed to
the repository - end users never need to run this script.

The script clones the repo automatically if it is not already present.

Usage:
    python scripts/create_task_metadata.py [--force] [--no-clone]

Options:
    --force      Overwrite task_metadata.json even if it already exists.
    --no-clone   Raise instead of cloning when the repo is missing.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

# Make the package importable when executed from the cube root without a venv
# activation step. This is a no-op when the package is already installed.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from osworld_cube.benchmark import OSWORLD_COMMIT, OSWorldTestSet
from osworld_cube.task import OSWorldTaskMetadata

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT = Path(__file__).parent.parent / "src" / "osworld_cube" / "task_metadata.json"


def _clone_repo(target: Path) -> None:
    """Clone the OSWorld repository and check out the pinned commit."""
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/xlang-ai/OSWorld", str(target)],
        check=True,
    )
    subprocess.run(["git", "checkout", OSWORLD_COMMIT], cwd=str(target), check=True)
    logger.info("Cloned OSWorld repo to %s @ %s", target, OSWORLD_COMMIT)


def generate_task_metadata(
    repo_dir: Path,
    output_path: Path = _DEFAULT_OUTPUT,
    *,
    force: bool = False,
    clone_if_missing: bool = True,
) -> int:
    """Parse the OSWorld repo and write task_metadata.json.

    Args:
        repo_dir:          Path to the cloned OSWorld repo.
        output_path:       Destination path for the JSON file.
                           Defaults to src/osworld_cube/task_metadata.json.
        force:             Overwrite even if output_path already exists.
        clone_if_missing:  Clone the repo if repo_dir does not exist.
                           When False, raises RuntimeError instead.

    Returns:
        Number of tasks written (0 if skipped due to idempotency).

    Raises:
        RuntimeError: If repo_dir does not exist and clone_if_missing=False.
    """
    if output_path.exists() and not force:
        logger.info(
            "task_metadata.json already exists at %s — skipping. Pass force=True to regenerate.",
            output_path,
        )
        return 0

    if not repo_dir.exists():
        if not clone_if_missing:
            raise RuntimeError(
                f"OSWorld repo not found at {repo_dir}. Pass clone_if_missing=True or clone it manually first."
            )
        _clone_repo(repo_dir)

    eval_examples_dir = repo_dir / "evaluation_examples"

    # Collect which test sets each task_id belongs to
    task_sets: dict[str, list[str]] = {}
    task_raw: dict[str, dict] = {}

    for test_set in OSWorldTestSet:
        test_set_file = eval_examples_dir / test_set.value
        if not test_set_file.exists():
            logger.warning("Test set file not found: %s", test_set_file)
            continue
        with open(test_set_file) as f:
            tasks_by_domain: dict[str, list[str]] = json.load(f)

        set_name = test_set.value.replace(".json", "")
        for domain_name, task_ids in tasks_by_domain.items():
            for task_id in task_ids:
                task_sets.setdefault(task_id, []).append(set_name)
                if task_id not in task_raw:
                    task_file = eval_examples_dir / "examples" / domain_name / f"{task_id}.json"
                    if not task_file.exists():
                        logger.warning("Task file not found: %s", task_file)
                        continue
                    try:
                        with open(task_file) as f:
                            td = json.load(f)
                        task_raw[task_id] = {"domain": domain_name, "data": td}
                    except Exception as e:
                        logger.error("Failed to load task %s: %s", task_id, e)

    metadata: list[OSWorldTaskMetadata] = []
    for task_id, info in task_raw.items():
        td = info["data"]
        domain_name = info["domain"]
        tm = OSWorldTaskMetadata(
            id=td.get("id", task_id),
            abstract_description="",
            instruction=td.get("instruction", ""),
            domain=domain_name,
            test_sets=task_sets.get(task_id, []),
            snapshot=td.get("snapshot", "init_state"),
            os_type=td.get("os_type", "ubuntu"),
            related_apps=td.get("related_apps", []),
        )
        metadata.append(tm)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([tm.model_dump() for tm in metadata], indent=2))
    logger.info("Saved %d tasks to %s", len(metadata), output_path)
    return len(metadata)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if file already exists")
    parser.add_argument("--no-clone", action="store_true", help="Raise instead of cloning when the repo is missing")
    args = parser.parse_args()

    from osworld_cube import OSWORLD_REPO_DIR

    generate_task_metadata(
        repo_dir=OSWORLD_REPO_DIR,
        force=args.force,
        clone_if_missing=not args.no_clone,
    )
