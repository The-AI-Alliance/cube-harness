"""Helpers for per-episode logging."""

import logging
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterator, TextIO

LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d %(funcName)s() - %(message)s"


class _TeeWriter:
    """Write the same output to multiple text streams."""

    def __init__(self, *writers: TextIO) -> None:
        self.writers = writers

    def write(self, data: str) -> int:
        for writer in self.writers:
            writer.write(data)
        return len(data)

    def flush(self) -> None:
        for writer in self.writers:
            writer.flush()


def trajectory_log_id(task_id: str, episode_id: int) -> str:
    """Build a stable trajectory identifier used by storage and logs."""
    return f"{task_id}_ep{episode_id}"


def get_log_path(output_dir: str | Path, trajectory_id: str) -> Path:
    """Return log file path for a trajectory (same directory as other experiment files)."""
    return Path(output_dir) / f"{trajectory_id}.log"


@contextmanager
def redirect_output_to_log(
    log_file: Path,
    *,
    append: bool,
    tee: bool,
    log_format: str = LOG_FORMAT,
) -> Iterator[None]:
    """Capture stdout/stderr into a log file and reconfigure logging stream."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    mode = "a" if append else "w"

    with log_file.open(mode, buffering=1) as log_stream:
        output_writer: TextIO = log_stream
        error_writer: TextIO = log_stream
        if tee:
            output_writer = _TeeWriter(original_stdout, log_stream)
            error_writer = _TeeWriter(original_stderr, log_stream)

        with redirect_stdout(output_writer), redirect_stderr(error_writer):  # type: ignore[type-var]
            logging.basicConfig(level=logging.INFO, format=log_format, stream=error_writer, force=True)
            try:
                yield
            finally:
                logging.basicConfig(level=logging.INFO, format=log_format, stream=original_stderr, force=True)
