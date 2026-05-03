"""Gold-patch baseline — validates the swe-bench-live eval pipeline.

The oracle agent applies the gold patch from /tmp/gold_patch.diff (written by
reset() when oracle_mode=True) and calls final_step. Every task that passes
resolve criteria here confirms the eval pipeline is correct; any that do not
indicate a scoring or environment bug.

Run this before trusting any agent score.

Usage:
    # Debug 2 tasks (no LLM, fast):
    .venv/bin/python recipes/gold_patch_baseline_recipe.py --debug

    # 3 specific tasks (cfn-lint, conan, haystack):
    .venv/bin/python recipes/gold_patch_baseline_recipe.py \\
        --tasks conan-io__conan-17300,aws-cloudformation__cfn-lint-3768,deepset-ai__haystack-8609

    # Full live-30 stratified sample:
    .venv/bin/python recipes/gold_patch_baseline_recipe.py --subset live30

    # Toolkit:
    .venv/bin/python recipes/gold_patch_baseline_recipe.py --subset live30 \\
        --toolkit --eai-profile yul101
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")
_project_env = Path(__file__).resolve().parents[1] / ".env"
if _project_env.exists():
    load_dotenv(_project_env, override=True)

from cube.core import Action, ActionSchema, Observation  # noqa: E402

from cube_harness.agent import AgentConfig  # noqa: E402
from cube_harness.exp_runner import run_sequentially, run_with_ray  # noqa: E402
from cube_harness.experiment import Experiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

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
# Gold-patch agent — applies /tmp/gold_patch.diff then stops
# ---------------------------------------------------------------------------

_APPLY = Action(
    name="bash",
    arguments={
        "command": "git apply /tmp/gold_patch.diff 2>&1 || git apply --reject /tmp/gold_patch.diff 2>&1 || patch --batch --fuzz=5 -p1 -i /tmp/gold_patch.diff 2>&1"
    },
)
_STOP = Action(name="final_step", arguments={})


class GoldPatchAgent:
    """Deterministic agent: apply the oracle gold patch, then stop."""

    def __init__(self) -> None:
        self._step = 0

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        action = _APPLY if self._step == 0 else _STOP
        self._step += 1
        return action


class GoldPatchAgentConfig(AgentConfig):
    """Config for GoldPatchAgent — no LLM, no parameters."""

    def make(self) -> GoldPatchAgent:  # type: ignore[override]
        return GoldPatchAgent()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _make_infra(toolkit: bool, eai_profile: str, eai_path: str) -> object:
    if toolkit:
        from cube_infra_toolkit import ToolkitInfraConfig

        return ToolkitInfraConfig(profile=eai_profile, eai_path=eai_path, launch_timeout_seconds=300)
    from cube.infra_local import LocalInfraConfig

    return LocalInfraConfig()


def _make_benchmark(debug: bool, task_ids: list[str] | None, subset: str | None, n_tasks: int | None) -> object:
    from swebench_live_cube.benchmark import SWEBenchLiveBenchmarkConfig

    if debug:
        from swebench_live_cube.debug import get_debug_benchmark

        return get_debug_benchmark()

    config = SWEBenchLiveBenchmarkConfig(oracle_mode=True)
    if subset == "live30":
        config = config.subset_from_list(list(_LIVE_30_SAMPLE))
    elif subset:
        config = config.named_subset(subset)
    if task_ids:
        config = config.subset_from_list(task_ids)
    elif n_tasks:
        all_ids = list(config.task_metadata.keys())[:n_tasks]
        config = config.subset_from_list(all_ids)
    return config


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(
    *,
    debug: bool,
    task_ids: list[str] | None,
    subset: str | None,
    n_tasks: int | None,
    n_parallel: int,
    toolkit: bool,
    eai_profile: str,
    eai_path: str,
) -> None:
    agent_config = GoldPatchAgentConfig()
    infra = _make_infra(toolkit, eai_profile, eai_path)
    benchmark_config = _make_benchmark(debug, task_ids, subset, n_tasks)

    infra_label = f"toolkit:{eai_profile}" if toolkit else "local"
    exp = Experiment(
        name=f"gold-patch-baseline-{infra_label}",
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=infra,
        max_steps=5,
    )
    print(f"\n=== gold-patch baseline | {infra_label} | subset={subset or task_ids or 'debug'} ===")

    if debug:
        run_sequentially(exp)
    else:
        run_with_ray(exp, n_cpus=n_parallel)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gold-patch baseline — validates the eval pipeline")
    parser.add_argument("--debug", action="store_true", help="Run 2 oracle debug tasks (no LLM)")
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs")
    parser.add_argument("--subset", default=None, choices=["live30", "lite", "verified", "full", "test"])
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--n-parallel", type=int, default=10)
    parser.add_argument("--toolkit", action="store_true")
    parser.add_argument("--eai-profile", default="yul101")
    parser.add_argument("--eai-path", default="eai")
    args = parser.parse_args()

    run(
        debug=args.debug,
        task_ids=[t.strip() for t in args.tasks.split(",")] if args.tasks else None,
        subset=args.subset,
        n_tasks=args.n_tasks,
        n_parallel=args.n_parallel,
        toolkit=args.toolkit,
        eai_profile=args.eai_profile,
        eai_path=args.eai_path,
    )
