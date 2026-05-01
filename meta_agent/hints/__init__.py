"""Task hint loader for cube benchmarks.

JSON hint files live alongside this module: hints/<benchmark_name>.json.
Each file is a flat JSON object mapping task_id → hint text.
"""

import json
from pathlib import Path

_HINTS_DIR = Path(__file__).parent


def load_hints(benchmark_name: str) -> dict[str, str]:
    """Load task hints for the given benchmark.

    Args:
        benchmark_name: Benchmark identifier matching the JSON filename
                        (e.g. 'swebench-verified').

    Returns:
        Dict mapping task_id to hint text, or empty dict if no file exists.
    """
    path = _HINTS_DIR / f"{benchmark_name}.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)
