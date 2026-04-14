# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "workarena-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# workarena-cube = { path = "../cubes/workarena", editable = true }
# ///

"""Example recipe for running WorkArena benchmark with cube-harness.

This recipe demonstrates how to run WorkArena tasks using the BrowserGym tool.

Prerequisites:
    1. Install WorkArena: pip install browsergym-workarena
    2. Configure ServiceNow credentials via environment variables:
       - SNOW_INSTANCE_URL: ServiceNow instance URL
       - SNOW_INSTANCE_UNAME: ServiceNow username
       - SNOW_INSTANCE_PWD: ServiceNow password
       OR
       - HUGGING_FACE_HUB_TOKEN: For accessing gated instance pool

Usage:
    # Genny agent, debug mode (default)
    uv run recipes/workarena.py --debug

    # React agent, debug mode
    uv run recipes/workarena.py --debug --agent react

    # Full run with Genny (parallel with Ray)
    uv run recipes/workarena.py

    # Full run with React
    uv run recipes/workarena.py --agent react
"""

import argparse

from cube.tool import ToolboxConfig
from cube_browser_playwright.playwright_session import PlaywrightSessionConfig
from cube_chat_tool import ChatToolConfig
from workarena_cube.benchmark import WorkArenaBenchmark
from workarena_cube.tools import WorkArenaInfeasibleToolConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import GennyConfig
from cube_harness.agents.react import ReactAgentConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.tools.browsergym import BrowsergymConfig

_LLM = LLMConfig(model_name="gpt-5-mini", temperature=1.0)

AGENTS = {
    "genny": GennyConfig(
        llm_config=_LLM,
        max_actions=20,
        render_last_n_obs=1,
        tools_as_text=False,
        enable_summarize=False,
        summarize_cot_only=True,
    ),
    "react": ReactAgentConfig(
        llm_config=_LLM,
        render_last_n_steps=3,
        max_actions=20,
    ),
}


def main(debug: bool, agent: str, level: int) -> None:
    agent_config = AGENTS[agent]
    output_dir = make_experiment_output_dir(agent, "workarena", tag=f"l{level}")

    tools_configs = [
        BrowsergymConfig(
            browser=PlaywrightSessionConfig(headless=not debug, timeout=30000),
            use_screenshot=True,
            use_axtree=True,
            use_html=False,
        ),
        ChatToolConfig(),
    ]
    if level > 1:
        tools_configs.append(WorkArenaInfeasibleToolConfig())
    tool_config = ToolboxConfig(tool_configs=tools_configs)

    # Configure WorkArena benchmark
    benchmark = WorkArenaBenchmark(default_tool_config=tool_config, level=f"l{level}", n_seeds_l1=1)

    exp = Experiment(
        name=f"workarena_{agent}",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=15,
    )

    if debug:
        run_sequentially(exp, debug_limit=2)
    else:
        run_with_ray(exp, n_cpus=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WorkArena benchmark with cube-harness.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (headed browser, limited tasks)")
    parser.add_argument("--agent", choices=AGENTS, default="genny", help="Agent to use (default: genny)")
    parser.add_argument("--level", choices=[1, 2, 3], default=1, help="Level to run (default: 1)")
    args = parser.parse_args()
    main(debug=args.debug, agent=args.agent, level=args.level)
