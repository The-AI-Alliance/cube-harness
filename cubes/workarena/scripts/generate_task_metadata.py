#!/usr/bin/env python3
"""Generate src/workarena_cube/task_metadata.json from the browsergym-workarena library.

This is a developer tool. Run it when the WorkArena task list changes to
regenerate the shipped package resource. The output file is committed to the
repository — end users never need to run this script.

Only lightweight public fields are written (level, in_human_curriculum, task_class_path).
WorkArena has no heavy execution data — all task logic is available from the
browsergym-workarena library at runtime via the task class path.

Usage:
    python scripts/generate_task_metadata.py [--output PATH] [--force]

Options:
    --output    Destination file (default: task_metadata.json inside the workarena_cube package).
    --force     Overwrite task_metadata.json even if it already exists.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from browsergym.workarena import get_all_tasks_agents

import workarena_cube
from workarena_cube.task import WorkArenaTaskMetadata

logger = logging.getLogger(__name__)

assert workarena_cube.__file__ is not None
_DEFAULT_OUTPUT = Path(workarena_cube.__file__).parent / "task_metadata.json"


def _build_task_metadata() -> dict[str, WorkArenaTaskMetadata]:
    """Enumerate all WorkArena task types and build typed metadata.

    Calls get_all_tasks_agents for L1, L2 agent superset, and L3 agent superset,
    then marks which tasks are in the human evaluation curriculum.
    Uses n_seed_l1=1 and meta_seed=0 — seeds are irrelevant here; only the
    task classes are needed.
    """
    metadata: dict[str, WorkArenaTaskMetadata] = {}

    # L1 — no curriculum concept
    for task_class, _ in get_all_tasks_agents(filter="l1", meta_seed=0, n_seed_l1=1):
        task_id = task_class.get_task_id()
        if task_id not in metadata:
            metadata[task_id] = WorkArenaTaskMetadata(
                id=task_id,
                level="l1",
                in_human_curriculum=False,
                task_class_path=f"{task_class.__module__}.{task_class.__qualname__}",
            )

    # L2 and L3 — agent curriculum is the superset; mark human curriculum tasks
    for level in ("l2", "l3"):
        human_ids: set[str] = {
            task_class.get_task_id()
            for task_class, _ in get_all_tasks_agents(filter=level, meta_seed=0, n_seed_l1=1, is_agent_curriculum=False)
        }
        for task_class, _ in get_all_tasks_agents(filter=level, meta_seed=0, n_seed_l1=1, is_agent_curriculum=True):
            task_id = task_class.get_task_id()
            if task_id not in metadata:
                metadata[task_id] = WorkArenaTaskMetadata(
                    id=task_id,
                    level=level,  # type: ignore[arg-type]
                    in_human_curriculum=task_id in human_ids,
                    task_class_path=f"{task_class.__module__}.{task_class.__qualname__}",
                )

    return metadata


def generate_task_metadata(
    output_path: Path = _DEFAULT_OUTPUT,
    *,
    force: bool = False,
) -> int:
    """Enumerate WorkArena task types and write the shipped task_metadata.json.

    Args:
        output_path: Destination path. Defaults to src/workarena_cube/task_metadata.json.
        force:       Overwrite even if output_path already exists.

    Returns:
        Number of tasks written (0 if skipped due to idempotency).
    """
    if output_path.exists() and not force:
        logger.info(
            "task_metadata.json already exists at %s — skipping. Pass --force to regenerate.",
            output_path,
        )
        return 0

    logger.info("Enumerating WorkArena task types...")
    metadata = _build_task_metadata()

    n_l1 = sum(1 for tm in metadata.values() if tm.level == "l1")
    n_l2 = sum(1 for tm in metadata.values() if tm.level == "l2")
    n_l3 = sum(1 for tm in metadata.values() if tm.level == "l3")
    logger.info("  %d tasks loaded (l1=%d, l2=%d, l3=%d)", len(metadata), n_l1, n_l2, n_l3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([tm.model_dump() for tm in metadata.values()], indent=2))
    logger.info("Saved %d tasks to %s", len(metadata), output_path)
    return len(metadata)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Destination file (default: task_metadata.json inside the workarena_cube package)",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if file already exists")
    args = parser.parse_args()

    generate_task_metadata(args.output, force=args.force)
