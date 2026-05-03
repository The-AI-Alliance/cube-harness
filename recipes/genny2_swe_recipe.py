"""Genny2 SWE-bench Verified recipe — flat_history mode with final_step submission.

Runs Genny2 with flat_history=True (linear conversation, no injected scaffolding),
tool_choice=required, and final_step as the explicit submission signal.
Cost limit $3/episode, step limit 250.

Usage:
    # Debug run (2 oracle tasks, sequential):
    .venv/bin/python recipes/genny2_swe_recipe.py --debug

    # Full swebench-verified on Toolkit:
    .venv/bin/python recipes/genny2_swe_recipe.py --toolkit \\
        --eai-profile yul101 --eai-path ~/bin/eai --n-parallel 20

    # HAL-50 subset:
    .venv/bin/python recipes/genny2_swe_recipe.py --subset hal_mini --toolkit \\
        --eai-profile yul101 --eai-path ~/bin/eai --n-parallel 20

    # Specific model:
    .venv/bin/python recipes/genny2_swe_recipe.py haiku --debug
    .venv/bin/python recipes/genny2_swe_recipe.py gpt-5.4-mini --toolkit ...
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

_project_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_project_env if _project_env.exists() else Path.home() / ".env", override=True)

from cube_harness.agents.genny2 import Genny2Config  # noqa: E402
from cube_harness.exp_runner import run_sequentially, run_with_ray  # noqa: E402
from cube_harness.experiment import Experiment  # noqa: E402
from cube_harness.llm import LLMConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

# ---------------------------------------------------------------------------
# HAL-50 task subset (25 Django + 25 Sphinx — Princeton HAL leaderboard set)
# ---------------------------------------------------------------------------

_HAL_MINI_TASK_IDS: frozenset[str] = frozenset(
    [
        "django__django-11790",
        "django__django-11815",
        "django__django-11848",
        "django__django-11880",
        "django__django-11885",
        "django__django-11951",
        "django__django-11964",
        "django__django-11999",
        "django__django-12039",
        "django__django-12050",
        "django__django-12143",
        "django__django-12155",
        "django__django-12193",
        "django__django-12209",
        "django__django-12262",
        "django__django-12273",
        "django__django-12276",
        "django__django-12304",
        "django__django-12308",
        "django__django-12325",
        "django__django-12406",
        "django__django-12708",
        "django__django-12713",
        "django__django-12774",
        "django__django-9296",
        "sphinx-doc__sphinx-10323",
        "sphinx-doc__sphinx-10435",
        "sphinx-doc__sphinx-10466",
        "sphinx-doc__sphinx-10673",
        "sphinx-doc__sphinx-11510",
        "sphinx-doc__sphinx-7590",
        "sphinx-doc__sphinx-7748",
        "sphinx-doc__sphinx-7757",
        "sphinx-doc__sphinx-7985",
        "sphinx-doc__sphinx-8035",
        "sphinx-doc__sphinx-8056",
        "sphinx-doc__sphinx-8265",
        "sphinx-doc__sphinx-8269",
        "sphinx-doc__sphinx-8475",
        "sphinx-doc__sphinx-8548",
        "sphinx-doc__sphinx-8551",
        "sphinx-doc__sphinx-8638",
        "sphinx-doc__sphinx-8721",
        "sphinx-doc__sphinx-9229",
        "sphinx-doc__sphinx-9230",
        "sphinx-doc__sphinx-9281",
        "sphinx-doc__sphinx-9320",
        "sphinx-doc__sphinx-9367",
        "sphinx-doc__sphinx-9461",
        "sphinx-doc__sphinx-9698",
    ]
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = "You are a helpful assistant that can interact with a computer shell to solve programming tasks."

# Minimal instance template: task description + the two constraints that matter for evaluation.
# Environment details (cwd=/testbed, stateless subshells) are covered by the bash tool description.
_INSTANCE_TEMPLATE = """\
{{task}}

Do not modify test files (tests/ directory, test_*.py files) or configuration files.

When your fix is complete:
1. Verify: `git diff > patch.txt && cat patch.txt`
2. Confirm the patch only contains source file changes, then call `final_step`.\
"""

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, LLMConfig] = {
    "gpt-5.4-mini": LLMConfig(
        model_name="azure/gpt-5.4-mini",
        temperature=1.0,
        tool_choice="required",
        parallel_tool_calls=True,
    ),
    "gpt-5.4": LLMConfig(
        model_name="azure/gpt-5.4",
        temperature=1.0,
        tool_choice="required",
        parallel_tool_calls=True,
    ),
    "haiku": LLMConfig(
        model_name="anthropic/claude-haiku-4-5",
        temperature=0.0,
        tool_choice="required",
        parallel_tool_calls=False,
        set_cache_control="auto",
    ),
    "sonnet": LLMConfig(
        model_name="anthropic/claude-sonnet-4-6",
        temperature=0.0,
        tool_choice="required",
        parallel_tool_calls=True,
        set_cache_control="auto",
    ),
}

# ---------------------------------------------------------------------------
# Infra / benchmark helpers
# ---------------------------------------------------------------------------


def _make_infra(toolkit: bool, eai_profile: str, eai_path: str, preemptable: bool) -> object:
    if toolkit:
        from cube_infra_toolkit import ToolkitInfraConfig

        return ToolkitInfraConfig(
            profile=eai_profile,
            eai_path=eai_path,
            preemptable=preemptable,
            launch_timeout_seconds=3000,
        )
    from cube.infra_local import LocalInfraConfig

    return LocalInfraConfig()


def _make_benchmark_config(
    debug: bool,
    task_ids: list[str] | None,
    subset: str | None,
) -> object:
    from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmarkConfig
    from swebench_verified_cube.tool import BashOnlySWEBenchToolConfig

    if debug:
        from swebench_verified_cube.debug import get_debug_benchmark

        return get_debug_benchmark()

    config = SWEBenchVerifiedBenchmarkConfig(tool_config=BashOnlySWEBenchToolConfig())
    if subset == "hal_mini":
        config = config.subset_from_list(list(_HAL_MINI_TASK_IDS))
    elif subset:
        config = config.named_subset(subset)
    if task_ids:
        config = config.subset_from_list(task_ids)
    return config


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(
    model_key: str,
    *,
    debug: bool,
    task_ids: list[str] | None,
    subset: str | None,
    n_parallel: int,
    retry_dir: Path | None,
    toolkit: bool,
    eai_profile: str,
    eai_path: str,
    preemptable: bool,
    max_actions: int = 250,
    cost_limit: float = 3.0,
) -> None:
    llm_config = MODEL_CONFIGS[model_key]

    agent_config = Genny2Config(
        llm_config=llm_config,
        system_prompt=_SYSTEM_PROMPT,
        goal_template=_INSTANCE_TEMPLATE,
        flat_history=True,
        step_prompt="",
        cost_limit=cost_limit,
        max_format_errors=3,
        max_actions=max_actions,
    )

    infra = _make_infra(toolkit, eai_profile, eai_path, preemptable)
    benchmark_config = _make_benchmark_config(debug, task_ids, subset)

    output_dir = retry_dir if retry_dir is not None else None
    resume = retry_dir is not None

    infra_label = f"toolkit:{eai_profile}" if toolkit else "local"
    exp = Experiment(
        name=f"genny2-swe-{model_key}-{infra_label}",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=infra,
        max_steps=max_actions,
        resume=resume,
    )

    print(f"\n=== genny2 | {model_key} | {infra_label} | subset={subset or 'all'} ===")

    if debug:
        run_sequentially(exp)
    else:
        run_with_ray(exp, n_cpus=n_parallel)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genny2 SWE-bench Verified recipe")
    parser.add_argument("model", nargs="?", default="gpt-5.4-mini", choices=list(MODEL_CONFIGS))
    parser.add_argument("--debug", action="store_true", help="Run debug oracle tasks sequentially")
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs")
    parser.add_argument("--subset", default=None, help="hal_mini or any named subset")
    parser.add_argument("--n-parallel", type=int, default=5)
    parser.add_argument("--retry", metavar="DIR", default=None, help="Resume from output dir")
    parser.add_argument("--toolkit", action="store_true")
    parser.add_argument("--eai-profile", default="yul101")
    parser.add_argument("--eai-path", default="eai")
    parser.add_argument("--preemptable", action="store_true")
    parser.add_argument("--max-actions", type=int, default=250)
    parser.add_argument("--cost-limit", type=float, default=3.0)
    args = parser.parse_args()

    run(
        model_key=args.model,
        debug=args.debug,
        task_ids=[t.strip() for t in args.tasks.split(",")] if args.tasks else None,
        subset=args.subset,
        n_parallel=args.n_parallel,
        retry_dir=Path(args.retry) if args.retry else None,
        toolkit=args.toolkit,
        eai_profile=args.eai_profile,
        eai_path=args.eai_path,
        preemptable=args.preemptable,
        max_actions=args.max_actions,
        cost_limit=args.cost_limit,
    )
