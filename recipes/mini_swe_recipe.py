"""Mini-SWE-agent style recipe — bash-only, 150 steps, reproduce-fix-verify workflow.

Inspired by mini-SWE-agent (github.com/SWE-agent/mini-swe-agent) which scores 74% on
SWE-bench Verified with a single bash tool and a simple ReAct loop.

Key differences from swe_agent_recipe.py:
  - Only bash tool exposed (no str_replace/view/read_file/write_file)
  - System prompt is minimal (delegates structure to instance prompt)
  - Instance prompt follows mini-SWE's: analyse → reproduce → fix → verify
  - max_actions=150 (vs 50)

Usage:
    uv run recipes/mini_swe_recipe.py                                  # swebench-verified, debug, gpt-5.4-mini
    uv run recipes/mini_swe_recipe.py swebench-verified gpt-5.4
    uv run recipes/mini_swe_recipe.py swebench-live gpt-5.4 --subset verified
    uv run recipes/mini_swe_recipe.py --toolkit --eai-path /bin/eai --n-parallel 20
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "meta_agent"))

_project_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_project_env if _project_env.exists() else Path.home() / ".env", override=True)

from hints import load_hints  # noqa: E402

from cube_harness.agents.genny import GennyConfig  # noqa: E402
from cube_harness.exp_runner import run_sequentially, run_with_ray  # noqa: E402
from cube_harness.experiment import Experiment  # noqa: E402
from cube_harness.llm import LLMConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

# ---------------------------------------------------------------------------
# Prompts — adapted from mini-SWE-agent's swebench.yaml
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful assistant that can interact with a computer shell to solve programming tasks."""

# The instance/act prompt is what Genny appends at every turn.
# We want the task description injected once (in the goal), and each turn just gets
# a lightweight "continue" nudge. The step counter [Step X/N] is prepended by Genny.
ACT_PROMPT = """\
Continue working on the task. For each response:
1. Include brief reasoning about what you're trying to do.
2. Issue at least one bash tool call.

Remember: each bash call runs in a fresh subshell — cd and exports do NOT persist.
When finished, call final_step."""

# The full task framing is injected as the goal (first user message).
# Genny uses task.description as the goal, which contains the problem statement.
# We augment it here via task_hints from the recipe, but the main instructions
# are baked into the system prompt below so they appear once per episode.

FULL_SYSTEM_PROMPT = """\
You are a helpful assistant that can interact with a computer shell to solve programming tasks.

You're a software engineer working on a real GitHub issue. The repository is already cloned \
at /testbed. Your task is to make changes to non-test source files in /testbed to fix the issue.

## Workflow (follow this order)

1. **Explore** — find and read relevant files (grep, cat, find)
2. **Reproduce** — write a short script (/tmp/repro.py) that triggers the reported bug; \
confirm it fails
3. **Fix** — edit the source code to resolve the issue (use sed, python inline, or heredoc)
4. **Verify** — run your repro script; confirm it passes; optionally run the repo's own tests
5. **Done** — call final_step

## Rules

- Only modify source files, NOT test files (tests/, test_*.py, *_test.py)
- Do NOT modify pyproject.toml, setup.cfg, or other config files
- Each bash call runs in a **fresh subshell** — cd and env changes do NOT persist; \
prefix every command with `cd /testbed &&` or use absolute paths
- Use `conda run -n testbed <cmd>` to run tests (the testbed conda env has all deps)
- PAGER=cat, MANPAGER=cat (pagers are disabled)

## Editing files

For small targeted edits, inline Python is reliable:
```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/testbed/src/module.py')
p.write_text(p.read_text().replace(
    'old exact text',
    'new text',
    1,
))
PY
```

For larger rewrites, use a heredoc:
```bash
cat > /testbed/src/module.py << 'EOF'
... full file content ...
EOF
```

## Step budget

Each step is labeled [Step X/N]. If you are past the halfway point and haven't located \
the root cause, make your best guess edit and call final_step — a partial fix is better \
than running out of steps with no change."""

MODEL_CONFIGS: dict[str, LLMConfig] = {
    "gpt-5.4-mini": LLMConfig(model_name="azure/gpt-5.4-mini"),
    "gpt-5.4": LLMConfig(model_name="azure/gpt-5.4"),
}


# ---------------------------------------------------------------------------
# Infra / benchmark factory (same as swe_agent_recipe)
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
    benchmark_name: str,
    debug: bool,
    task_ids: list[str] | None,
    subset: str | None,
) -> object:
    from cube.tools.terminal import TerminalToolConfig

    tool_config = TerminalToolConfig(working_dir="/testbed")
    if benchmark_name == "swebench-verified":
        if debug:
            from swebench_verified_cube.debug import get_debug_benchmark

            bench = get_debug_benchmark()
            bench.tool_config = tool_config
            return bench
        from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmarkConfig

        config = SWEBenchVerifiedBenchmarkConfig(tool_config=tool_config)
    elif benchmark_name == "swebench-live":
        if debug:
            from swebench_live_cube.debug import get_debug_benchmark

            bench = get_debug_benchmark()
            bench.tool_config = tool_config
            return bench
        from swebench_live_cube.benchmark import SWEBenchLiveBenchmarkConfig

        config = SWEBenchLiveBenchmarkConfig(tool_config=tool_config)
    else:
        raise ValueError(f"mini_swe_recipe supports swebench-verified and swebench-live, got {benchmark_name!r}")

    if subset == "live-golden-30":
        from swebench_live_cube.gold_patch_recipe import _LIVE_GOLDEN_30

        config = config.subset_from_list(list(_LIVE_GOLDEN_30))
    elif subset:
        config = config.named_subset(subset)
    if task_ids:
        config = config.subset_from_list(task_ids)
    return config


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(
    benchmark_name: str,
    model_key: str,
    *,
    debug: bool,
    use_hints: bool,
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
    task_hints = load_hints(benchmark_name) if use_hints else {}

    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=FULL_SYSTEM_PROMPT,
        act_prompt=ACT_PROMPT,
        max_actions=150,
        render_last_n_obs=2,
        task_hints=task_hints,
    )

    output_dir = retry_dir if retry_dir is not None else None
    resume = retry_dir is not None

    infra = _make_infra(toolkit, eai_profile, eai_path, preemptable)
    benchmark_config = _make_benchmark_config(benchmark_name, debug, task_ids, subset)

    infra_label = f"toolkit:{eai_profile}" if toolkit else "local"
    exp = Experiment(
        name=f"mini-genny-{benchmark_name}-{infra_label}",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=infra,
        max_steps=150,
        resume=resume,
    )

    label = f"hints={use_hints}" if benchmark_name == "swebench-verified" else ""
    print(f"\n=== mini-SWE | {benchmark_name} | {model_key} | {infra_label} | {label or 'no hints'} ===")

    if debug:
        run_sequentially(exp)
    else:
        run_with_ray(exp, n_cpus=n_parallel)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mini-SWE-agent style recipe (bash-only, 150 steps)")
    parser.add_argument(
        "benchmark", nargs="?", default="swebench-verified", choices=["swebench-verified", "swebench-live"]
    )
    parser.add_argument("model", nargs="?", default="gpt-5.4-mini", choices=list(MODEL_CONFIGS))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--hints", action="store_true")
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--n-parallel", type=int, default=5)
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
        use_hints=args.hints,
        task_ids=[t.strip() for t in args.tasks.split(",")] if args.tasks else None,
        subset=args.subset,
        n_parallel=args.n_parallel,
        retry_dir=Path(args.retry) if args.retry else None,
        toolkit=args.toolkit,
        eai_profile=args.eai_profile,
        eai_path=args.eai_path,
        preemptable=args.preemptable,
    )
