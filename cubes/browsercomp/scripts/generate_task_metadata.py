#!/usr/bin/env python3
"""Generate src/browsercomp_cube/task_metadata.json from the OpenAI public CSV.

Developer tool. Run when the BrowseComp dataset is updated to regenerate the
shipped package resource. The output file is committed to the repository — end
users never need to run this script.

Only lightweight public fields are written (id, abstract_description,
recommended_max_steps, topic). The encrypted ``problem`` and ``answer`` columns
remain encrypted and are decrypted at task make() time from the per-task
execution cache populated by ``BrowseCompBenchmarkConfig.install()``.

Usage:
    python scripts/generate_task_metadata.py [--output PATH] [--force]

Options:
    --output    Destination file (default: task_metadata.json inside the browsercomp_cube package).
    --force     Overwrite task_metadata.json even if it already exists.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
from pathlib import Path

import browsercomp_cube
from browsercomp_cube.benchmark import BrowseCompBenchmarkConfig
from browsercomp_cube.task import BrowseCompTaskMetadata

logger = logging.getLogger(__name__)

assert browsercomp_cube.__file__ is not None
_DEFAULT_OUTPUT = Path(browsercomp_cube.__file__).parent / "task_metadata.json"


def _build_task_metadata(rows: list[dict[str, str]]) -> list[BrowseCompTaskMetadata]:
    """Build lightweight TaskMetadata from CSV rows. Only ``problem_topic`` is read."""
    return [
        BrowseCompTaskMetadata(
            id=f"browsecomp-{idx:04d}",
            recommended_max_steps=50,
            topic=row.get("problem_topic", ""),
        )
        for idx, row in enumerate(rows)
    ]


def generate_task_metadata(output_path: Path = _DEFAULT_OUTPUT, *, force: bool = False) -> int:
    """Download the source CSV and write the shipped task_metadata.json."""
    if output_path.exists() and not force:
        logger.info(
            "task_metadata.json already exists at %s — skipping. Pass --force to regenerate.",
            output_path,
        )
        return 0

    csv_path = BrowseCompBenchmarkConfig._download_dataset()
    text = csv_path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    logger.info("Loaded %d rows from %s", len(rows), csv_path)

    metadata = _build_task_metadata(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([tm.model_dump(exclude_defaults=True) for tm in metadata], indent=2))
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
        help="Destination file (default: task_metadata.json inside the browsercomp_cube package)",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if file already exists")
    args = parser.parse_args()

    generate_task_metadata(args.output, force=args.force)
