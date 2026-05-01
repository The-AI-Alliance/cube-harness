"""MiniAgent recipe — faithful port of mini-SWE-agent for SWE-bench Verified/Live.

Runs MiniAgent (flat message list, bash-only, 250 steps, temp=0) on a benchmark.
Default subset is the 50-task HAL leaderboard set (django + sphinx); compare results
against published GPT-5-medium baseline of 46% on that subset.

Usage:
    # Local debug run (2 oracle tasks, sequential):
    uv run recipes/mini_agent_recipe.py --debug

    # HAL-50 on Toolkit (compare to 40-55% target band):
    uv run recipes/mini_agent_recipe.py --subset hal_mini --toolkit --eai-profile yul101 --eai-path ~/bin/eai --n-parallel 20

    # Full 500-task run:
    uv run recipes/mini_agent_recipe.py --toolkit --eai-profile yul101 --eai-path ~/bin/eai --n-parallel 20

    # Specific tasks:
    uv run recipes/mini_agent_recipe.py --tasks django__django-12039,sphinx-doc__sphinx-8056 --n-parallel 2
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

_project_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_project_env if _project_env.exists() else Path.home() / ".env", override=True)

from cube_harness.agents.mini_agent import MiniAgentConfig  # noqa: E402
from cube_harness.exp_runner import run_sequentially, run_with_ray  # noqa: E402
from cube_harness.experiment import Experiment  # noqa: E402
from cube_harness.llm import LLMConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

MODEL_CONFIGS: dict[str, LLMConfig] = {
    "gpt-5.4-mini": LLMConfig(
        model_name="azure/gpt-5.4-mini",
        temperature=0.0,
        # NOTE: parallel_tool_calls=False (default). Upstream uses True.
        # Enabling it may improve performance; requires testing.
        parallel_tool_calls=False,
    ),
    "gpt-5.4": LLMConfig(
        model_name="azure/gpt-5.4",
        temperature=0.0,
        parallel_tool_calls=False,
    ),
    "sonnet": LLMConfig(
        model_name="anthropic/claude-sonnet-4-6",
        temperature=0.0,
        parallel_tool_calls=False,
    ),
}


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
    benchmark_name: str,
    debug: bool,
    task_ids: list[str] | None,
    subset: str | None,
) -> object:
    if benchmark_name == "swebench-verified":
        from swebench_verified_cube.benchmark import HAL_MINI_TASK_IDS, SWEBenchVerifiedBenchmarkConfig
        from swebench_verified_cube.tool import BashOnlySWEBenchToolConfig

        tool_config = BashOnlySWEBenchToolConfig()
        if debug:
            from swebench_verified_cube.debug import get_debug_benchmark

            bench = get_debug_benchmark()
            bench.tool_config = tool_config
            return bench
        config = SWEBenchVerifiedBenchmarkConfig(tool_config=tool_config)
        if subset == "hal_mini":
            config = config.subset_from_list(list(HAL_MINI_TASK_IDS))
        elif subset:
            config = config.named_subset(subset)
        if task_ids:
            config = config.subset_from_list(task_ids)
        return config
    if benchmark_name == "swebench-live":
        from swebench_live_cube.benchmark import SWEBenchLiveBenchmarkConfig
        from swebench_live_cube.tool import BashOnlySWEBenchToolConfig as LiveToolConfig

        tool_config = LiveToolConfig()
        if debug:
            from swebench_live_cube.debug import get_debug_benchmark

            bench = get_debug_benchmark()
            bench.tool_config = tool_config
            return bench
        config = SWEBenchLiveBenchmarkConfig(tool_config=tool_config)
        if subset and subset != "hal_mini":
            config = config.named_subset(subset)
        if task_ids:
            config = config.subset_from_list(task_ids)
        return config
    raise ValueError(f"mini_agent_recipe supports swebench-verified and swebench-live, got {benchmark_name!r}")


def run(
    benchmark_name: str,
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
) -> None:
    llm_config = MODEL_CONFIGS[model_key]
    agent_config = MiniAgentConfig(llm_config=llm_config)

    infra = _make_infra(toolkit, eai_profile, eai_path, preemptable)
    benchmark_config = _make_benchmark_config(benchmark_name, debug, task_ids, subset)

    output_dir = retry_dir if retry_dir is not None else None
    resume = retry_dir is not None
    infra_label = f"toolkit:{eai_profile}" if toolkit else "local"

    exp = Experiment(
        name=f"mini-agent-{benchmark_name}-{infra_label}",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=infra,
        # Outer Episode limit — set slightly above agent's step_limit so the agent's
        # own counter governs (agent stops cleanly, Episode doesn't force-cut it).
        max_steps=260,
        resume=resume,
    )

    subset_label = f"subset={subset}" if subset else "full"
    print(f"\n=== MiniAgent | {benchmark_name} | {model_key} | {infra_label} | {subset_label} ===")

    if debug:
        run_sequentially(exp)
    else:
        run_with_ray(exp, n_cpus=n_parallel)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MiniAgent recipe — bash-only, 250 steps, upstream mini-SWE-agent port"
    )
    parser.add_argument(
        "benchmark",
        nargs="?",
        default="swebench-verified",
        choices=["swebench-verified", "swebench-live"],
    )
    parser.add_argument("model", nargs="?", default="gpt-5.4-mini", choices=list(MODEL_CONFIGS))
    parser.add_argument("--debug", action="store_true", help="Oracle mode, sequential, 2 tasks")
    parser.add_argument(
        "--subset",
        default="hal_mini",
        help="Named subset: 'hal_mini' (HAL 50-task; default), or any benchmark-defined name",
    )
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs (overrides --subset)")
    parser.add_argument("--n-parallel", type=int, default=10)
    parser.add_argument("--retry", metavar="DIR", default=None)
    parser.add_argument("--toolkit", action="store_true")
    parser.add_argument("--eai-profile", default="yul101")
    parser.add_argument("--eai-path", default="eai")
    parser.add_argument("--preemptable", action="store_true")
    args = parser.parse_args()

    run(
        benchmark_name=args.benchmark,
        model_key=args.model,
        debug=args.debug,
        task_ids=[t.strip() for t in args.tasks.split(",")] if args.tasks else None,
        subset=args.subset if not args.tasks else None,
        n_parallel=args.n_parallel,
        retry_dir=Path(args.retry) if args.retry else None,
        toolkit=args.toolkit,
        eai_profile=args.eai_profile,
        eai_path=args.eai_path,
        preemptable=args.preemptable,
    )
