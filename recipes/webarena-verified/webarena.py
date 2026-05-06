"""Run WebArena-Verified benchmark with cube-harness.

Usage:
    # Shopping subset, debug mode
    .venv/bin/python recipes/webarena-verified/webarena.py --subset shopping --debug

    # Shopping admin with Sonnet
    .venv/bin/python recipes/webarena-verified/webarena.py --subset shopping_admin --model anthropic/claude-sonnet-4-6

    # Gitlab with custom name
    .venv/bin/python recipes/webarena-verified/webarena.py --subset gitlab --name my-experiment
"""

import argparse

from cube.infra_local import LocalInfraConfig
from cube.tool import ToolboxConfig
from cube_browser_playwright import PlaywrightSessionConfig
from webarena_verified_cube.benchmark import WebArenaVerifiedBenchmarkConfig
from webarena_verified_cube.resources import (
    WEBARENA_GITLAB,
    WEBARENA_MAP,
    WEBARENA_REDDIT,
    WEBARENA_SHOPPING,
    WEBARENA_SHOPPING_ADMIN,
    WEBARENA_WIKIPEDIA,
)
from webarena_verified_cube.tool import HarPlaywrightConfig, SubmitResponseConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny2 import Genny2Config
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig

_DEFAULT_MODEL = "openai/gpt-5.4-mini"

_SUBSETS = {
    "shopping": (WEBARENA_SHOPPING, "*shopping*"),
    "shopping_admin": (WEBARENA_SHOPPING_ADMIN, "*shopping_admin*"),
    "gitlab": (WEBARENA_GITLAB, "*gitlab*"),
    "reddit": (WEBARENA_REDDIT, "*reddit*"),
    "map": (WEBARENA_MAP, "*map*"),
    "wikipedia": (WEBARENA_WIKIPEDIA, "*wikipedia*"),
}


def main(debug: bool, subset: str, model: str, name: str | None, n_cpus: int) -> None:
    resource, sites_glob = _SUBSETS[subset]

    model_short = model.split("/")[-1]
    exp_name = name if name is not None else f"webarena-verified-{subset}/genny2-{model_short}"
    output_dir = make_experiment_output_dir("genny2", f"webarena-verified-{subset}")

    llm_config = LLMConfig(model_name=model, temperature=1.0, set_cache_control="auto")
    agent_config = Genny2Config(llm_config=llm_config)

    tool_config = ToolboxConfig(
        tool_configs=[
            HarPlaywrightConfig(browser=PlaywrightSessionConfig(headless=not debug)),
            SubmitResponseConfig(),
        ]
    )
    benchmark_config = (
        WebArenaVerifiedBenchmarkConfig(
            tool_config=tool_config,
            resources=[resource],
        )
        .subset_from_glob("sites", sites_glob)
    )

    exp = Experiment(
        name=exp_name,
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=LocalInfraConfig(),
        max_steps=30,
    )

    if debug:
        run_sequentially(exp, debug_limit=2)
    else:
        run_with_ray(exp, n_cpus=n_cpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WebArena-Verified benchmark.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (headed browser, limited tasks)")
    parser.add_argument("--subset", choices=list(_SUBSETS), required=True, help="Site subset to run")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help=f"LLM model (default: {_DEFAULT_MODEL})")
    parser.add_argument("--name", default=None, help="Experiment name (default: auto-generated)")
    parser.add_argument("--n-cpus", type=int, default=4, help="Number of parallel Ray workers (default: 4)")
    args = parser.parse_args()
    main(debug=args.debug, subset=args.subset, model=args.model, name=args.name, n_cpus=args.n_cpus)
