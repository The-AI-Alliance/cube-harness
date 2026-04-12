"""Test recipe for create-record tasks with submit_form() + keyboard_type_into().

Runs just the create tasks to verify the new submit_form() action and hints work.

Usage:
    uv run recipes/test_create_tasks.py           # all 5 create tasks, sequential
    uv run recipes/test_create_tasks.py incident  # only create-incident
"""

import sys

from cube.tool import ToolboxConfig
from cube_browser_playwright.playwright_session import PlaywrightSessionConfig
from cube_chat_tool import ChatToolConfig
from workarena_cube.agent_hints import WORKARENA_DEFAULT_HINT, WORKARENA_TASK_HINTS
from workarena_cube.benchmark import WorkArenaBenchmark

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import GennyConfig
from cube_harness.exp_runner import run_sequentially
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.tools.browsergym import BrowsergymConfig

_LLM = LLMConfig(model_name="azure/gpt-5.4-mini", temperature=1.0)

AGENT = GennyConfig(
    llm_config=_LLM,
    max_actions=40,
    render_last_n_obs=1,
    tools_as_text=False,
    enable_summarize=False,
    summarize_cot_only=True,
    hint=WORKARENA_DEFAULT_HINT,
    task_hints=WORKARENA_TASK_HINTS,
)

_CREATE_TASK_IDS = {
    "incident": "workarena.servicenow.create-incident",
    "hardware-asset": "workarena.servicenow.create-hardware-asset",
    "change-request": "workarena.servicenow.create-change-request",
    "user": "workarena.servicenow.create-user",
    "problem": "workarena.servicenow.create-problem",
}


def main(task_filter: str | None) -> None:
    output_dir = make_experiment_output_dir("genny", "workarena-create-test")

    tool_config = ToolboxConfig(
        tool_configs=[
            BrowsergymConfig(
                browser=PlaywrightSessionConfig(headless=False, timeout=30000),
                use_screenshot=False,
                use_axtree=True,
                use_html=False,
                pre_observation_delay=1.0,
            ),
            ChatToolConfig(),
        ]
    )

    benchmark = WorkArenaBenchmark(default_tool_config=tool_config, level="l1", n_seeds_l1=1)
    benchmark.setup()

    if task_filter is not None:
        task_id = _CREATE_TASK_IDS.get(task_filter)
        if task_id is None:
            print(f"Unknown task filter: {task_filter!r}. Choose from: {list(_CREATE_TASK_IDS.keys())}")
            sys.exit(1)
        benchmark = benchmark.subset_from_list([task_id])

    exp = Experiment(
        name="workarena-create-test",
        output_dir=output_dir,
        agent_config=AGENT,
        benchmark=benchmark,
        max_steps=25,
    )

    run_sequentially(exp)


if __name__ == "__main__":
    args = sys.argv[1:]
    task_filter = args[0] if args else None
    main(task_filter)
