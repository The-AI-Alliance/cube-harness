#!/usr/bin/env python3
"""Generate src/webarena_verified_cube/task_metadata.json from the webarena-verified library.

This is a developer tool. Run it when the webarena-verified task list changes to
regenerate the shipped package resource. The output file is committed to the
repository — end users never need to run this script.

Only lightweight public fields are written (sites, expected_action, intent_template_id).
WebArena has no heavy execution data — all task information is available from the
webarena-verified library at runtime.

Usage:
    python scripts/generate_task_metadata.py [--output PATH] [--force]

Options:
    --output    Destination file (default: task_metadata.json inside the webarena_verified_cube package).
    --force     Overwrite task_metadata.json even if it already exists.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from webarena_verified.api.webarena_verified import WebArenaVerified

import webarena_verified_cube
from webarena_verified_cube.task import WebArenaVerifiedTaskMetadata

logger = logging.getLogger(__name__)

assert webarena_verified_cube.__file__ is not None
_DEFAULT_OUTPUT = Path(webarena_verified_cube.__file__).parent / "task_metadata.json"


def generate_task_metadata(
    output_path: Path = _DEFAULT_OUTPUT,
    *,
    force: bool = False,
) -> int:
    """Load tasks from the webarena-verified library and write the shipped task_metadata.json.

    Args:
        output_path: Destination path. Defaults to src/webarena_verified_cube/task_metadata.json.
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

    logger.info("Loading tasks from webarena-verified library...")
    wav = WebArenaVerified()
    tasks = wav.get_tasks()
    logger.info("  %d tasks loaded", len(tasks))

    metadata: dict[str, WebArenaVerifiedTaskMetadata] = {
        str(t.task_id): WebArenaVerifiedTaskMetadata(
            id=str(t.task_id),
            abstract_description=t.intent,
            recommended_max_steps=30,
            sites=[s.value for s in t.sites],
            expected_action=t.expected_action,
            intent_template_id=t.intent_template_id,
        )
        for t in tasks
    }

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
        help="Destination file (default: task_metadata.json inside the webarena_verified_cube package)",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if file already exists")
    args = parser.parse_args()

    generate_task_metadata(args.output, force=args.force)
