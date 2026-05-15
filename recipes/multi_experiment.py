# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "swebench-verified-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# swebench-verified-cube = { path = "../cubes/swebench-verified", editable = true }
# ///
"""Reference recipe: more than one Experiment in a single file.

Name them `exp_<something>` and hand them all to `run`. Each canonical
config lookup returns an independent deep copy, so tweaking one experiment's
agent never affects the other.
"""

from swebench_verified_cube import SWEBENCH_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.infra import INFRA_CONFIGS
from cube_harness.recipe import run

_default = GENNY_CONFIGS["default"]
_swe = GENNY_CONFIGS["swe"]
_swe.budget.cost_limit = 4.0

exp_default = Experiment(
    name="genny-default",
    agent_config=_default,
    benchmark_config=SWEBENCH_CONFIGS["default"],
    infra=INFRA_CONFIGS["local"],
)

exp_swe = Experiment(
    name="genny-swe",
    agent_config=_swe,
    benchmark_config=SWEBENCH_CONFIGS["default"],
    infra=INFRA_CONFIGS["local"],
)

if __name__ == "__main__":
    run(exp_default, exp_swe)
