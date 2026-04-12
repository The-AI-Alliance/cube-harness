"""WorkArena L1 full evaluation — all tasks, 1 seed, v15 hints.

Usage:
    uv run recipes/workarena_l1_full.py                    # both models
    uv run recipes/workarena_l1_full.py gpt-5.4-mini       # single model
    uv run recipes/workarena_l1_full.py gpt-5.4            # single model
    uv run recipes/workarena_l1_full.py debug              # sequential, 1 task, gpt-5.4-mini
    uv run recipes/workarena_l1_full.py headless-off       # force headless=False (fixes keyboard_type)

Models:
    - azure/gpt-5.4-mini  (baseline, ~33 tasks)
    - azure/gpt-5.4       (stronger)

Hints (v15):
    - Filter: combobox button click pattern for field/operator selectors
    - Sort: combobox button for field selector, select_option for direction (native <select>)
    - Chart: numeric-only for value-retrieval, label+count for min-max
    - Create: keyboard_type_into for reference fields (fill bypasses autocomplete)
"""

import sys

from cube.tool import ToolboxConfig
from cube_browser_playwright.playwright_session import PlaywrightSessionConfig
from cube_chat_tool.chat_tool import ChatToolConfig
from workarena_cube.agent_hints import WORKARENA_DEFAULT_HINT, WORKARENA_TASK_HINTS
from workarena_cube.benchmark import WorkArenaBenchmark

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


def make_agent(llm_config: LLMConfig) -> GennyConfig:
    return GennyConfig(
        llm_config=llm_config,
        max_actions=40,
        render_last_n_obs=1,
        hint=WORKARENA_DEFAULT_HINT,
        task_hints=WORKARENA_TASK_HINTS,
    )


def run_for_model(model_key: str, llm_config: LLMConfig, debug: bool, headless: bool) -> None:
    tool_config = ToolboxConfig(
        tool_configs=[
            BrowsergymConfig(
                browser=PlaywrightSessionConfig(headless=headless, timeout=30000),
                use_screenshot=False,
                use_axtree=True,
                use_html=False,
            ),
            ChatToolConfig(),
        ]
    )

    benchmark = WorkArenaBenchmark(level="l1", n_seeds_l1=1, default_tool_config=tool_config)
    benchmark.setup()

    output_dir = make_experiment_output_dir("genny", f"workarena-l1-full-{model_key}")
    exp = Experiment(
        name=f"workarena-l1-full-{model_key}",
        output_dir=output_dir,
        agent_config=make_agent(llm_config),
        benchmark=benchmark,
        max_steps=40,
    )

    if debug:
        run_sequentially(exp, debug_limit=1)
    else:
        run_with_ray(exp, n_cpus=4)


def main(debug: bool, headless: bool, models: list[str]) -> None:
    for model_key in models:
        llm_config = MODEL_CONFIGS[model_key]
        print(f"\n=== Running model: {model_key} (headless={headless}) ===")
        run_for_model(model_key, llm_config, debug, headless)


if __name__ == "__main__":
    args = set(sys.argv[1:])
    debug = "debug" in args
    headless = not debug and "headless-off" not in args

    selected = [k for k in MODEL_CONFIGS if k in args]
    if not selected:
        selected = list(MODEL_CONFIGS.keys())

    main(debug=debug, headless=headless, models=selected)
