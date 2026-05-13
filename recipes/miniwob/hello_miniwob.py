"""Run MiniWoB benchmark with cube-harness.

Usage:
    # Genny agent, debug mode
    uv run --project recipes/miniwob recipes/miniwob/hello_miniwob.py --debug

    # React agent with Sonnet
    uv run --project recipes/miniwob recipes/miniwob/hello_miniwob.py --agent react --model anthropic/claude-sonnet-4-6

    # BrowserGym tool backend
    uv run --project recipes/miniwob recipes/miniwob/hello_miniwob.py --debug --browser browsergym

    # Full run with 8 workers
    uv run --project recipes/miniwob recipes/miniwob/hello_miniwob.py --n-cpus 8
"""

import argparse

from cube_browser_tool import PlaywrightConfig
from miniwob_cube.benchmark import MiniWobBenchmarkConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import GennyConfig
from cube_harness.agents.react import ReactAgentConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.tools.browsergym import BrowsergymConfig

_DEFAULT_MODEL = "gpt-5-mini"


def make_agents(model: str) -> dict[str, GennyConfig | ReactAgentConfig]:
    llm = LLMConfig(model_name=model, temperature=1.0)
    return {
        "genny": GennyConfig(llm_config=llm),
        "react": ReactAgentConfig(llm_config=llm),
    }


def _make_tool_config(browser: str, debug: bool) -> PlaywrightConfig | BrowsergymConfig:
    if browser == "browsergym":
        return BrowsergymConfig(use_screenshot=True, use_html=True, use_axtree=False)
    return PlaywrightConfig(use_screenshot=True, headless=not debug)


def main(debug: bool, agent: str, model: str, name: str | None, n_cpus: int, n_seeds: int, browser: str) -> None:
    agent_config = make_agents(model)[agent]
    model_short = model.split("/")[-1]
    exp_name = name if name is not None else f"miniwob/{agent}-{model_short}"
    output_dir = make_experiment_output_dir(agent, "miniwob")

    tool_config = _make_tool_config(browser, debug)
    benchmark_config = MiniWobBenchmarkConfig(tool_config=tool_config)

    exp = Experiment(
        name=exp_name,
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        max_steps=10,
        n_seeds=n_seeds,
    )

    if debug:
        run_sequentially(exp, debug_limit=2)
    else:
        run_with_ray(exp, n_cpus=n_cpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MiniWoB benchmark.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (headed browser, limited tasks)")
    parser.add_argument("--agent", choices=("genny", "react"), default="genny", help="Agent to use (default: genny)")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help=f"LLM model (default: {_DEFAULT_MODEL})")
    parser.add_argument("--name", default=None, help="Experiment name (default: auto-generated)")
    parser.add_argument("--n-cpus", type=int, default=4, help="Number of parallel Ray workers (default: 4)")
    parser.add_argument("--n-seeds", type=int, default=4, help="Number of seeds")
    parser.add_argument("--browser", choices=("playwright", "browsergym"), default="playwright",
                        help="Browser tool backend (default: playwright)")
    args = parser.parse_args()
    main(debug=args.debug, agent=args.agent, model=args.model, name=args.name, n_cpus=args.n_cpus,
         n_seeds=args.n_seeds, browser=args.browser)
