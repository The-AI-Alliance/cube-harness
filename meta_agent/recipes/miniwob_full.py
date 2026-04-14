"""MiniWob full evaluation — all 125 tasks, 1 seed.

Usage:
    uv run meta_agent/recipes/miniwob_full.py gpt-5.4                # no hints
    uv run meta_agent/recipes/miniwob_full.py gpt-5.4 hints          # with task hints
    uv run meta_agent/recipes/miniwob_full.py gpt-5.4-mini           # single model, no hints
    uv run meta_agent/recipes/miniwob_full.py debug                  # sequential, 1 task
    uv run meta_agent/recipes/miniwob_full.py headless-off           # force headless=False
    uv run meta_agent/recipes/miniwob_full.py retry /path/to/exp     # retry crashed episodes
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# meta_agent/ is not a Python package — add it to sys.path so we can import miniwob_hints.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env so credentials are available even when the shell didn't source ~/.zshrc.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from miniwob_cube.benchmark import MiniWobBenchmark
from miniwob_hints import MINIWOB_DEFAULT_HINT, MINIWOB_TASK_HINTS

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import GennyConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.tools.browsergym import BrowsergymConfig

MODEL_CONFIGS: dict[str, LLMConfig] = {
    "gpt-5.4-mini": LLMConfig(model_name="azure/gpt-5.4-mini", temperature=1.0),
    "gpt-5.4": LLMConfig(model_name="azure/gpt-5.4", temperature=1.0),
}


def make_agent(llm_config: LLMConfig, use_hints: bool = False) -> GennyConfig:
    return GennyConfig(
        llm_config=llm_config,
        max_actions=20,
        render_last_n_obs=3,
        hint=MINIWOB_DEFAULT_HINT if use_hints else "",
        task_hints=MINIWOB_TASK_HINTS if use_hints else {},
    )


def run_for_model(
    model_key: str,
    llm_config: LLMConfig,
    debug: bool,
    headless: bool,
    use_hints: bool,
    retry_dir: Path | None = None,
) -> None:
    # MiniWob: html exposes bid attributes on DOM elements (needed for clickable links
    # with <span class="alink"> that don't appear in the AXTree).
    tool_config = BrowsergymConfig(
        use_screenshot=True,
        use_axtree=True,
        use_html=True,
    )

    benchmark = MiniWobBenchmark(default_tool_config=tool_config)
    benchmark.setup()

    suffix = "hints" if use_hints else "nohints"
    if retry_dir is not None:
        output_dir = retry_dir
        retry_failed = True
        resume = True
    else:
        output_dir = make_experiment_output_dir("genny", f"miniwob-{suffix}-{model_key}")
        retry_failed = False
        resume = False

    exp = Experiment(
        name=f"miniwob-{suffix}-{model_key}",
        output_dir=output_dir,
        agent_config=make_agent(llm_config, use_hints=use_hints),
        benchmark=benchmark,
        max_steps=20,
        retry_failed=retry_failed,
        resume=resume,
    )

    if debug:
        run_sequentially(exp, debug_limit=1)
    else:
        run_with_ray(exp, n_cpus=10)


def main(debug: bool, headless: bool, models: list[str], use_hints: bool, retry_dir: Path | None) -> None:
    for model_key in models:
        llm_config = MODEL_CONFIGS[model_key]
        hint_label = "WITH hints" if use_hints else "NO hints"
        label = f"RETRY {retry_dir}" if retry_dir else hint_label
        print(f"\n=== {model_key} | {label} | headless={headless} ===")
        run_for_model(model_key, llm_config, debug, headless, use_hints, retry_dir)


if __name__ == "__main__":
    args = sys.argv[1:]
    args_set = set(args)
    debug = "debug" in args_set
    headless = not debug and "headless-off" not in args_set
    use_hints = "hints" in args_set

    retry_dir: Path | None = None
    if "retry" in args_set:
        retry_idx = args.index("retry")
        if retry_idx + 1 < len(args):
            retry_dir = Path(args[retry_idx + 1])
        else:
            print("ERROR: 'retry' flag requires a path argument", file=sys.stderr)
            sys.exit(1)

    selected = [k for k in MODEL_CONFIGS if k in args_set]
    if not selected:
        selected = ["gpt-5.4"]  # default to gpt-5.4

    main(debug=debug, headless=headless, models=selected, use_hints=use_hints, retry_dir=retry_dir)
