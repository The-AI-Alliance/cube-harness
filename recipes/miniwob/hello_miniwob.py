"""Run MiniWoB benchmark with cube-harness.

Usage:
    # Genny2, debug mode
    .venv/bin/python recipes/miniwob/hello_miniwob.py --debug

    # React agent with Sonnet
    .venv/bin/python recipes/miniwob/hello_miniwob.py --agent react --model anthropic/claude-sonnet-4-6

    # Full run with 8 workers
    .venv/bin/python recipes/miniwob/hello_miniwob.py --n-cpus 8
"""

import argparse

from cube.seed import BasicSeedGenerator
from cube_browser_playwright.playwright_session import PlaywrightSessionConfig
from miniwob_cube.benchmark import MiniWobBenchmarkConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny2 import Genny2Config
from cube_harness.agents.react import ReactAgentConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.tools.browsergym import BrowsergymConfig

_DEFAULT_MODEL = "openai/gpt-5-nano"


def make_agents(model: str) -> dict[str, Genny2Config | ReactAgentConfig]:
    llm = LLMConfig(model_name=model, temperature=1.0, set_cache_control="auto")
    return {
        "genny2": Genny2Config(llm_config=llm),
        "react": ReactAgentConfig(llm_config=llm),
    }


def main(debug: bool, agent: str, model: str, name: str | None, n_cpus: int, n_seeds: int) -> None:
    agent_config = make_agents(model)[agent]
    model_short = model.split("/")[-1]
    exp_name = name if name is not None else f"miniwob/{agent}-{model_short}"
    output_dir = make_experiment_output_dir(agent, "miniwob")

    tool_config = BrowsergymConfig(
        browser=PlaywrightSessionConfig(headless=not debug),
        use_screenshot=True,
        use_axtree=True,
        use_html=False,
    )
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
    parser.add_argument("--agent", choices=("genny2", "react"), default="genny2", help="Agent to use (default: genny2)")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help=f"LLM model (default: {_DEFAULT_MODEL})")
    parser.add_argument("--name", default=None, help="Experiment name (default: auto-generated)")
    parser.add_argument("--n-cpus", type=int, default=4, help="Number of parallel Ray workers (default: 4)")
    parser.add_argument("--n-seeds", type=int, default=4, help="Number of seeds")
    args = parser.parse_args()
    main(debug=args.debug, agent=args.agent, model=args.model, name=args.name, n_cpus=args.n_cpus, n_seeds=args.n_seeds)
