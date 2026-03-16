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
    uv run recipes/workarena.py debug

    # React agent, debug mode
    uv run recipes/workarena.py debug react

    # Full run with Genny (parallel with Ray)
    uv run recipes/workarena.py

    # Full run with React
    uv run recipes/workarena.py react
"""

import sys

from cube_browser_playwright.playwright_session import PlaywrightSessionConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import GennyConfig
from cube_harness.agents.react import ReactAgentConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.tools.browsergym import BrowsergymConfig

try:
    from workarena_cube.benchmark import WorkArenaBenchmark
except ImportError:
    print("WorkArena benchmark requires 'workarena-cube'. Run `make install` to install all optional dependencies.")
    sys.exit(1)

_LLM = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)

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


def main(debug: bool, agent: str) -> None:
    agent_config = AGENTS[agent]
    output_dir = make_experiment_output_dir(agent, "workarena", tag="l1")

    tool_config = BrowsergymConfig(
        browser=PlaywrightSessionConfig(headless=not debug, timeout=30000),
        use_screenshot=True,
        use_axtree=True,
        use_html=False,
    )

    # Configure WorkArena benchmark
    benchmark = WorkArenaBenchmark(default_tool_config=tool_config, level="l1", n_seeds_l1=1)

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
    args = set(sys.argv[1:])
    main(debug="debug" in args, agent="react" if "react" in args else "genny")
