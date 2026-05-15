# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "miniwob-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# miniwob-cube = { path = "../cubes/miniwob", editable = true }
# ///
"""Reference recipe: ReAct on MiniWoB. The simplest end-to-end example.

This file IS the config — copy it and edit the values. It is not a CLI;
`run()` provides a fixed generic CLI (see `cube_harness.recipe`).
"""

from cube_browser_tool import PlaywrightConfig
from miniwob_cube.benchmark import MiniWobBenchmarkConfig

from cube_harness.agents.react_configs import REACT_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.recipe import run

agent = REACT_CONFIGS["default"]

exp = Experiment(
    name="miniwob",
    agent_config=agent,
    benchmark_config=MiniWobBenchmarkConfig(tool_config=PlaywrightConfig(use_screenshot=True, headless=True)),
    max_steps=10,
)

if __name__ == "__main__":
    run(exp)
