# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "webarena-verified-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# webarena-verified-cube = { path = "../cubes/webarena-verified", editable = true }
# ///
"""Reference recipe: ReAct on WebArena-Verified, infra auto-provisioned.

This file IS the config — copy it and edit the values. It is not a CLI.
Infra is named, not constructed: pick another entry from ~/.cube/infra.py
(see `cube_harness.infra`); "local" works with zero setup.
"""

from cube.tool import ToolboxConfig
from cube_browser_playwright import PlaywrightSessionConfig
from webarena_verified_cube.benchmark import WebArenaVerifiedBenchmarkConfig
from webarena_verified_cube.resources import WEBARENA_ALL
from webarena_verified_cube.tool import HarPlaywrightConfig, SubmitResponseConfig

from cube_harness.agents.react_configs import REACT_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.infra import INFRA_CONFIGS
from cube_harness.recipe import run

agent = REACT_CONFIGS["default"]

benchmark = WebArenaVerifiedBenchmarkConfig(
    tool_config=ToolboxConfig(
        tool_configs=[
            HarPlaywrightConfig(browser=PlaywrightSessionConfig(headless=True)),
            SubmitResponseConfig(),
        ]
    ),
    resources=[WEBARENA_ALL],
).subset_from_glob("sites", "*shopping_admin*")

exp = Experiment(
    name="webarena-verified",
    agent_config=agent,
    benchmark_config=benchmark,
    infra=INFRA_CONFIGS["local"],
    max_steps=30,
)

if __name__ == "__main__":
    run(exp)
