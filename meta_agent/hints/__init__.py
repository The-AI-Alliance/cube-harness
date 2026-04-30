"""Task hint and clarification loaders for cube benchmarks.

- hints/<benchmark>.json          → load_hints(benchmark)
- clarifications/<benchmark>.json → load_clarifications(benchmark)

Each file is a flat JSON object mapping task_id → text.
"""

import json
from pathlib import Path

_META_AGENT_DIR = Path(__file__).parent.parent


def load_hints(benchmark_name: str) -> dict[str, str]:
    """Load task hints for the given benchmark.

    Args:
        benchmark_name: Benchmark identifier matching the JSON filename
                        (e.g. 'swebench-verified', 'workarena').

    Returns:
        Dict mapping task_id to hint text, or empty dict if no file exists.
    """
    path = _META_AGENT_DIR / "hints" / f"{benchmark_name}.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def load_clarifications(benchmark_name: str) -> dict[str, str]:
    """Load task clarifications (task_clarification in GennyConfig) for the given benchmark.

    Args:
        benchmark_name: Benchmark identifier matching the JSON filename
                        (e.g. 'workarena').

    Returns:
        Dict mapping task_id to clarification text, or empty dict if no file exists.
    """
    path = _META_AGENT_DIR / "clarifications" / f"{benchmark_name}.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)
