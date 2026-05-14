"""Trajectory judge — post-hoc LLM analysis of cube-harness episodes.

Reads a completed experiment directory, decompresses the `*.msgpack.zst` step
files into a readable transcript, and invokes a coding-agent driver (Claude
Code SDK by default; terminal `claude -p` available) to produce a structured
judgment per episode.

The default judge recipe is `general_blame`. Other recipes (`profiling`,
`agent_scaffolding`) live under `use_cases/` and ship typed extensions of the
base shape.

Per-episode results are written into `episode_record.json` (sibling fields
`judge_output` and `judge_metadata`) and aggregated at the experiment root
into `experiment_judge_summary.json`, `experiment_judge_report.csv`, and
`experiment_judge_report.json`. Cross-experiment aggregation lives in
`cube_harness.analyze.cross_experiment`.

CLI: `ch-judge <experiment_dir> [options]` — see `cli.py` for flags.

This `__init__` re-exports each submodule's public surface via `import *`;
each submodule maintains its own `__all__`. The package-level surface is
therefore the union of those submodule lists — no separate `__all__` here to
drift out of sync.
"""

from cube_harness.analyze.judge.agent_driver import *  # noqa: F401, F403
from cube_harness.analyze.judge.audit import *  # noqa: F401, F403
from cube_harness.analyze.judge.benchmark_context_agent import *  # noqa: F401, F403
from cube_harness.analyze.judge.cli import main  # noqa: F401
from cube_harness.analyze.judge.context import *  # noqa: F401, F403
from cube_harness.analyze.judge.core import *  # noqa: F401, F403
from cube_harness.analyze.judge.episode_discovery import *  # noqa: F401, F403
from cube_harness.analyze.judge.meta_analysis import *  # noqa: F401, F403
from cube_harness.analyze.judge.parse import *  # noqa: F401, F403
from cube_harness.analyze.judge.recipe import *  # noqa: F401, F403

# Eager default. The use_cases catalog is built once when the package above
# imports; resolving the default here turns it into a real constant so callers
# can `from cube_harness.analyze.judge import DEFAULT_RECIPE` ergonomically.
from cube_harness.analyze.judge.recipe import JudgeRecipe, get_default_recipe  # noqa: E402
from cube_harness.analyze.judge.selectors import *  # noqa: F401, F403
from cube_harness.analyze.judge.transcript import *  # noqa: F401, F403
from cube_harness.analyze.judge.use_cases import RECIPE_CATALOG  # noqa: F401

DEFAULT_RECIPE: JudgeRecipe = get_default_recipe()
