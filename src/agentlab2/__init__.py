import os
import re
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("agentlab2")
except PackageNotFoundError:
    __version__ = "unknown"

# Default experiment output base directory. Override with AL2_EXP_DIR.
_EXP_DIR_RAW = os.environ.get("AL2_EXP_DIR", "~/al2_exp_dir")
EXP_DIR: Path = Path(_EXP_DIR_RAW).expanduser().resolve()

__all__ = ["__version__", "EXP_DIR", "make_experiment_output_dir"]


def _sanitize(name: str) -> str:
    """Sanitize a string for use in a filesystem path (no slashes, minimal safe chars)."""
    return re.sub(r"[^\w\-.]", "_", name).strip("_") or "run"


def make_experiment_output_dir(
    agent_name: str,
    benchmark_name: str,
    tag: str | None = None,
) -> Path:
    """
    Create and return an experiment output directory under EXP_DIR.

    Path is: EXP_DIR / {date}_{agent}_{benchmark}_{tag}
    with date as YYYYMMDD and tag defaulting to HHMMSS if not provided.

    Args:
        agent_name: Agent identifier (e.g. "react").
        benchmark_name: Benchmark identifier (e.g. "miniwob").
        tag: Optional user tag. If None, a timestamp is used so each run is unique.

    Returns:
        Path to the created directory (parents created automatically).
    """
    date_str = time.strftime("%Y%m%d")
    tag_str = _sanitize(tag) if (tag and str(tag).strip()) else time.strftime("%H%M%S")
    agent_s = _sanitize(agent_name)
    benchmark_s = _sanitize(benchmark_name)
    subdir = f"{date_str}_{agent_s}_{benchmark_s}_{tag_str}"
    output_dir = EXP_DIR / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
