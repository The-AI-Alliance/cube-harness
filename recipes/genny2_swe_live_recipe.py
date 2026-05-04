"""Genny2 SWE recipe — swe-bench-live.

Adapts genny2_swe_recipe.py for swe-bench-live:
- Uses SWEBenchLiveBenchmarkConfig (bash + read_file + write_file tool, no <returncode> wrapper)
- Submission via final_step (same as genny2 default)
- Subsets: test (1000), lite (300), verified (499), full (1887)

Usage:
    # Debug run (2 oracle tasks, sequential — no LLM needed):
    .venv/bin/python recipes/genny2_swe_live_recipe.py --debug

    # First N tasks from a subset (for initial benchmark debugging):
    .venv/bin/python recipes/genny2_swe_live_recipe.py --subset lite --n-tasks 5

    # Full 'lite' subset on Toolkit:
    .venv/bin/python recipes/genny2_swe_live_recipe.py --subset lite --toolkit \\
        --eai-profile yul101 --eai-path ~/bin/eai --n-parallel 20

    # Specific model:
    .venv/bin/python recipes/genny2_swe_live_recipe.py haiku --debug
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")
_project_env = Path(__file__).resolve().parents[1] / ".env"
if _project_env.exists():
    load_dotenv(_project_env, override=True)

from cube_harness.agents.genny2 import Genny2Config  # noqa: E402
from cube_harness.exp_runner import run_sequentially, run_with_ray  # noqa: E402
from cube_harness.experiment import Experiment  # noqa: E402
from cube_harness.llm import LLMConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

# ---------------------------------------------------------------------------
# Stratified 30-task sample from the lite subset (1 task per repo, seed=42).
# 30 repos out of 70 in lite — use with --subset live30 for initial signal runs.
# ---------------------------------------------------------------------------

_LIVE_30_SAMPLE: frozenset[str] = frozenset(
    [
        "conan-io__conan-17300",
        "aws-cloudformation__cfn-lint-3768",
        "deepset-ai__haystack-8609",
        "matplotlib__matplotlib-29258",
        "reflex-dev__reflex-4717",
        "instructlab__instructlab-3118",
        "run-llama__llama_deploy-458",
        "pydata__xarray-9636",
        "pvlib__pvlib-python-2400",
        "streamlink__streamlink-6242",
        "keras-team__keras-20443",
        "pdm-project__pdm-3250",
        "sphinx-doc__sphinx-12975",
        "python-babel__babel-1141",
        "projectmesa__mesa-2394",
        "pylint-dev__pylint-10240",
        "kozea__weasyprint-2300",
        "joke2k__faker-2142",
        "wemake-services__wemake-python-styleguide-3114",
        "privacyidea__privacyidea-4223",
        "sissbruecker__linkding-989",
        "kedro-org__kedro-4387",
        "stanfordnlp__dspy-1609",
        "yt-dlp__yt-dlp-11880",
        "pypsa__pypsa-1195",
        "beeware__briefcase-2214",
        "huggingface__smolagents-843",
        "dynaconf__dynaconf-1238",
        "ipython__ipython-14822",
        "jupyterlab__jupyter-ai-1022",
    ]
)

# ---------------------------------------------------------------------------
# System prompt — verbatim from upstream mini-swe-agent swebench.yaml
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = "You are a helpful assistant that can interact with a computer shell to solve programming tasks."

# ---------------------------------------------------------------------------
# Instance template — minimal: task + constraints + submit block.
# Earlier verbose mini-swe-agent template buried `final_step` 100+ lines deep
# and weaker models (haiku) never reached it before looping out at MAX_STEPS.
# ---------------------------------------------------------------------------

_INSTANCE_TEMPLATE = "{{task}}"

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
        temperature=1.0,
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
            launch_timeout_seconds=300,
        )
    from cube.infra_local import LocalInfraConfig

    return LocalInfraConfig()


def _make_benchmark_config(
    debug: bool,
    task_ids: list[str] | None,
    subset: str | None,
    n_tasks: int | None,
    solvable_from: Path | None = None,
) -> object:
    from swebench_live_cube.benchmark import SWEBenchLiveBenchmarkConfig

    if debug:
        from swebench_live_cube.debug import get_debug_benchmark

        return get_debug_benchmark()

    config = SWEBenchLiveBenchmarkConfig()
    if solvable_from is not None:
        task_ids = json.loads(solvable_from.read_text())
    if subset == "live30":
        config = config.subset_from_list(list(_LIVE_30_SAMPLE))
    elif subset:
        config = config.named_subset(subset)
    if task_ids:
        config = config.subset_from_list(task_ids)
    elif n_tasks:
        all_task_ids = list(config.task_metadata.keys())[:n_tasks]
        config = config.subset_from_list(all_task_ids)
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
    n_tasks: int | None,
    n_parallel: int,
    retry_dir: Path | None,
    toolkit: bool,
    eai_profile: str,
    eai_path: str,
    preemptable: bool,
    solvable_from: Path | None = None,
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
        obs_format="raw",
        cost_limit=cost_limit,
        budget_hint_interval_usd=1.0 if cost_limit is not None else None,
        max_format_errors=3,
        max_actions=max_actions,
        max_obs_chars=20_000,
        compact_threshold_chars=400_000,
    )

    infra = _make_infra(toolkit, eai_profile, eai_path, preemptable)
    benchmark_config = _make_benchmark_config(debug, task_ids, subset, n_tasks, solvable_from)

    output_dir = retry_dir if retry_dir is not None else None
    resume = retry_dir is not None

    infra_label = f"toolkit:{eai_profile}" if toolkit else "local"
    exp = Experiment(
        name=f"genny2-swe-live-{model_key}-{infra_label}",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=infra,
        max_steps=max_actions,
        resume=resume,
    )

    print(f"\n=== genny2-swe-live | {model_key} | {infra_label} | subset={subset or 'all'} ===")

    if debug:
        run_sequentially(exp)
    else:
        run_with_ray(exp, n_cpus=n_parallel)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genny2 swe-bench-live recipe")
    parser.add_argument("model", nargs="?", default="gpt-5.4-mini", choices=list(MODEL_CONFIGS))
    parser.add_argument("--debug", action="store_true", help="Run 2 oracle debug tasks sequentially")
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs")
    parser.add_argument(
        "--subset",
        default=None,
        choices=["test", "lite", "verified", "full", "live30"],
        help="Named subset: live30 (30-task stratified), lite (300), verified (499), full (1887), test (1000)",
    )
    parser.add_argument("--n-tasks", type=int, default=None, help="Take first N tasks from subset")
    parser.add_argument("--n-parallel", type=int, default=5)
    parser.add_argument("--retry", metavar="DIR", default=None, help="Resume from output dir")
    parser.add_argument("--toolkit", action="store_true")
    parser.add_argument("--eai-profile", default="yul101")
    parser.add_argument("--eai-path", default="eai")
    parser.add_argument("--preemptable", action="store_true")
    parser.add_argument("--max-actions", type=int, default=250)
    parser.add_argument("--cost-limit", type=float, default=3.0)
    parser.add_argument(
        "--solvable-from",
        metavar="PATH",
        default=None,
        help="JSON file of solvable task IDs (output of gold_patch_baseline_recipe --dump-solvable)",
    )
    args = parser.parse_args()

    run(
        model_key=args.model,
        debug=args.debug,
        task_ids=[t.strip() for t in args.tasks.split(",")] if args.tasks else None,
        subset=args.subset,
        n_tasks=args.n_tasks,
        n_parallel=args.n_parallel,
        retry_dir=Path(args.retry) if args.retry else None,
        toolkit=args.toolkit,
        eai_profile=args.eai_profile,
        eai_path=args.eai_path,
        preemptable=args.preemptable,
        solvable_from=Path(args.solvable_from) if args.solvable_from else None,
        max_actions=args.max_actions,
        cost_limit=args.cost_limit,
    )
