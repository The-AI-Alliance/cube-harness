"""Gold-patch baseline recipe — thin shim.

The implementation lives in the swebench-live-cube package:
  swebench_live_cube.gold_patch_agent   — GoldPatchAgent, GoldPatchAgentConfig
  swebench_live_cube.gold_patch_recipe  — run_gold_baseline, extract_solvable, intersect_solvable

Run directly from the cube (preferred):
    .venv/bin/python -m swebench_live_cube.gold_patch_recipe --help

Or via this shim (same behaviour):
    .venv/bin/python recipes/gold_patch_baseline_recipe.py --help
"""

from pathlib import Path

from dotenv import load_dotenv

# Re-export for any code that imports from here (e.g. run_gold_3x.py)
from swebench_live_cube.gold_patch_agent import GoldPatchAgent, GoldPatchAgentConfig  # noqa: F401
from swebench_live_cube.gold_patch_recipe import (  # noqa: F401
    _run_once,
    _write_solvable,
    extract_solvable,
    intersect_solvable,
    run_gold_baseline,
    run_once,
)

load_dotenv(Path.home() / ".env")
_project_env = Path(__file__).resolve().parents[1] / ".env"
if _project_env.exists():
    load_dotenv(_project_env, override=True)

if __name__ == "__main__":
    import runpy

    runpy.run_module("swebench_live_cube.gold_patch_recipe", run_name="__main__")
