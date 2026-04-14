"""WorkArena L1 full evaluation — all tasks, 1 seed.

Usage:
    uv run recipes/workarena_l1_full.py gpt-5.4                # no hints (code fixes only)
    uv run recipes/workarena_l1_full.py gpt-5.4 hints          # with task hints
    uv run recipes/workarena_l1_full.py gpt-5.4-mini           # single model, no hints
    uv run recipes/workarena_l1_full.py debug                   # sequential, 1 task
    uv run recipes/workarena_l1_full.py headless-off            # force headless=False
    uv run recipes/workarena_l1_full.py retry /path/to/exp     # retry crashed episodes

Code fixes (always active):
    - AXTree: with_clickable=True, readonly/focusable visible
    - Action errors propagated to observation (last_action_error)
    - send_message hidden for non-chat tasks (sort/filter/create)
    - js_eval removed from action set
    - Tool docstrings updated (browser_type, browser_click, submit_form)

Hints (only with 'hints' flag):
    - Create: fill ALL fields, autocomplete workflow, submit_form
    - Sort/Filter: use filter UI, combobox bid+1 pattern
    - Chart: answer format (numeric-only vs label+count)
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# meta_agent/ is not a Python package — add it to sys.path so we can import workarena_hints.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env so credentials are available even when the shell didn't source ~/.zshrc.
# Ray workers inherit the parent process env, so this must run before ray.init().
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

from cube.tool import ToolboxConfig
from cube_browser_playwright.playwright_session import PlaywrightSessionConfig
from cube_chat_tool.chat_tool import ChatToolConfig
from workarena_cube.benchmark import WorkArenaBenchmark
from workarena_hints import WORKARENA_TASK_HINTS, WORKARENA_TASK_PRECISION

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
        max_actions=40,
        render_last_n_obs=1,
        # Task precision is always on — it compensates for under-defined task goals.
        task_precision=WORKARENA_TASK_PRECISION,
        # Task-specific hints are optional — they help the LLM but aren't strictly necessary.
        task_hints=WORKARENA_TASK_HINTS if use_hints else {},
    )


def run_for_model(
    model_key: str,
    llm_config: LLMConfig,
    debug: bool,
    headless: bool,
    use_hints: bool,
    retry_dir: Path | None = None,
) -> None:
    tool_config = ToolboxConfig(
        tool_configs=[
            BrowsergymConfig(
                browser=PlaywrightSessionConfig(headless=headless, timeout=60000),
                use_screenshot=True,
                use_axtree=True,
                use_html=False,
            ),
            ChatToolConfig(),
        ]
    )

    benchmark = WorkArenaBenchmark(level="l1", n_seeds_l1=1, default_tool_config=tool_config)
    benchmark.setup()

    suffix = "hints" if use_hints else "nohints"
    if retry_dir is not None:
        output_dir = retry_dir
        retry_failed = True
        resume = True
    else:
        output_dir = make_experiment_output_dir("genny", f"workarena-l1-{suffix}-{model_key}")
        retry_failed = False
        resume = False

    exp = Experiment(
        name=f"workarena-l1-{suffix}-{model_key}",
        output_dir=output_dir,
        agent_config=make_agent(llm_config, use_hints=use_hints),
        benchmark=benchmark,
        max_steps=40,
        retry_failed=retry_failed,
        resume=resume,
    )

    if debug:
        run_sequentially(exp, debug_limit=2)
    else:
        run_with_ray(exp, n_cpus=5)


def main(debug: bool, headless: bool, models: list[str], use_hints: bool, retry_dir: Path | None) -> None:
    for model_key in models:
        llm_config = MODEL_CONFIGS[model_key]
        hint_label = "WITH hints" if use_hints else "NO hints (code fixes only)"
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
