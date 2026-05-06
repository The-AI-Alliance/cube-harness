"""Run WebArena-Verified benchmark with cube-harness.

Usage:
    # BrowserGym test split (381 tasks, all sites) — the standard eval
    .venv/bin/python recipes/webarena-verified/webarena.py --bgym-test --model anthropic/claude-sonnet-4-6

    # Single site subset
    .venv/bin/python recipes/webarena-verified/webarena.py --subset shopping_admin --model anthropic/claude-sonnet-4-6

    # Debug mode
    .venv/bin/python recipes/webarena-verified/webarena.py --subset shopping --debug
"""

import argparse
import json
from importlib.resources import files

from cube.infra_local import LocalInfraConfig
from cube.tool import ToolboxConfig
from cube_browser_playwright import PlaywrightSessionConfig
from webarena_verified_cube.benchmark import WebArenaVerifiedBenchmarkConfig
from webarena_verified_cube.resources import (
    WEBARENA_ALL,
    WEBARENA_GITLAB,
    WEBARENA_MAP,
    WEBARENA_REDDIT,
    WEBARENA_SHOPPING,
    WEBARENA_SHOPPING_ADMIN,
    WEBARENA_WIKIPEDIA,
)
from webarena_verified_cube.tool import SubmitResponseConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny2 import Genny2Config
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.tools.browsergym import BrowsergymConfig

_DEFAULT_MODEL = "openai/gpt-5.4-mini"

_SUBSETS = {
    "shopping": (WEBARENA_SHOPPING, "*shopping*"),
    "shopping_admin": (WEBARENA_SHOPPING_ADMIN, "*shopping_admin*"),
    "gitlab": (WEBARENA_GITLAB, "*gitlab*"),
    "reddit": (WEBARENA_REDDIT, "*reddit*"),
    "map": (WEBARENA_MAP, "*map*"),
    "wikipedia": (WEBARENA_WIKIPEDIA, "*wikipedia*"),
}


def _load_bgym_test_ids() -> list[str]:
    data = files("webarena_verified_cube").joinpath("bgym_test_split.json").read_text()
    return json.loads(data)


def main(debug: bool, subset: str | None, bgym_test: bool, model: str, name: str | None, n_cpus: int) -> None:
    model_short = model.split("/")[-1]

    tool_config = ToolboxConfig(
        tool_configs=[
            BrowsergymConfig(
                browser=PlaywrightSessionConfig(headless=not debug),
                use_screenshot=True,
                use_axtree=True,
                use_html=False,
            ),
            SubmitResponseConfig(),
        ]
    )

    if bgym_test:
        tag = "bgym-test"
        exp_name = name if name is not None else f"webarena-verified-bgym-test/genny2-{model_short}"
        output_dir = make_experiment_output_dir("genny2", "webarena-verified-bgym-test")
        test_ids = _load_bgym_test_ids()
        benchmark_config = (
            WebArenaVerifiedBenchmarkConfig(
                tool_config=tool_config,
                resources=[WEBARENA_ALL],
            )
            .subset_from_list(test_ids)
        )
    else:
        assert subset is not None
        resource, sites_glob = _SUBSETS[subset]
        tag = subset
        exp_name = name if name is not None else f"webarena-verified-{subset}/genny2-{model_short}"
        output_dir = make_experiment_output_dir("genny2", f"webarena-verified-{subset}")
        benchmark_config = (
            WebArenaVerifiedBenchmarkConfig(
                tool_config=tool_config,
                resources=[resource],
            )
            .subset_from_glob("sites", sites_glob)
        )

    llm_config = LLMConfig(model_name=model, temperature=1.0, set_cache_control="auto")
    agent_config = Genny2Config(llm_config=llm_config)

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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subset", choices=list(_SUBSETS), help="Site subset to run")
    group.add_argument("--bgym-test", action="store_true", help="Run the 381-task BrowserGym test split (all sites)")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help=f"LLM model (default: {_DEFAULT_MODEL})")
    parser.add_argument("--name", default=None, help="Experiment name (default: auto-generated)")
    parser.add_argument("--n-cpus", type=int, default=4, help="Number of parallel Ray workers (default: 4)")
    args = parser.parse_args()
    main(debug=args.debug, subset=args.subset, bgym_test=args.bgym_test, model=args.model, name=args.name, n_cpus=args.n_cpus)
