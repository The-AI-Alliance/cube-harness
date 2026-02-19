import os
import time
from pathlib import Path

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentlab2")
except PackageNotFoundError:
    __version__ = "unknown"

# Standard experiment output root; override with AL2_EXP_DIR (default: ~/al2_exp_dir)
_EXP_DIR_RAW = os.environ.get("AL2_EXP_DIR", "~/al2_exp_dir")
EXP_DIR = Path(_EXP_DIR_RAW).expanduser().resolve()


def make_experiment_output_dir(
    agent_name: str,
    benchmark_name: str,
    tag: str | None = None,
) -> Path:
    """Create and return a new experiment output directory under EXP_DIR.

    Directory name is: {date}_{agent_name}_{benchmark_name}[_{tag}].
    The directory is created if it does not exist.

    Args:
        agent_name: Agent identifier (e.g. 'react').
        benchmark_name: Benchmark identifier (e.g. 'miniwob', 'workarena').
        tag: Optional extra tag (e.g. 'resumption_demo', 'l1').

    Returns:
        Path to the created directory.
    """
    now = time.strftime("%Y%m%d_%H%M%S")
    parts = [now, agent_name, benchmark_name]
    if tag:
        parts.append(tag)
    dir_name = "_".join(parts)
    path = EXP_DIR / dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["EXP_DIR", "__version__", "make_experiment_output_dir"]
