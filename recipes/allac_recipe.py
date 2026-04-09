import sys

from cube_browser_playwright.playwright_session import PlaywrightSessionConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import GennyConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.tools.browsergym import BrowsergymConfig

try:
    from workarena_cube.benchmark import WorkArenaBenchmark
except ImportError:
    print("WorkArena benchmark requires 'workarena-cube'. Run `make install` to install all optional dependencies.")
    sys.exit(1)

# Targeted subset for fast iteration:
# - chart/knowledge/impersonation: validate send_msg_to_user now works (6 tasks)
# - 3 order tasks: control group (should pass)
# - 2 filter + 2 sort: verify bgym native actions work
_TARGET_TASK_IDS = [
    "workarena.servicenow.single-chart-value-retrieval",
    "workarena.servicenow.single-chart-min-max-retrieval",
    "workarena.servicenow.multi-chart-value-retrieval",
    "workarena.servicenow.multi-chart-min-max-retrieval",
    "workarena.servicenow.knowledge-base-search",
    "workarena.servicenow.impersonation",
    "workarena.servicenow.order-standard-laptop",
    "workarena.servicenow.order-apple-watch",
    "workarena.servicenow.order-sales-laptop",
    "workarena.servicenow.filter-incident-list",
    "workarena.servicenow.filter-asset-list",
    "workarena.servicenow.sort-incident-list",
    "workarena.servicenow.sort-asset-list",
]


def main(debug: bool) -> None:
    output_dir = make_experiment_output_dir("genny", "workarena", tag="l1-subset")

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = GennyConfig(
        llm_config=llm_config,
        max_actions=40,
        render_last_n_obs=1,
        tools_as_text=False,
        enable_summarize=False,
        summarize_cot_only=True,
    )

    tool_config = BrowsergymConfig(
        browser=PlaywrightSessionConfig(headless=True, timeout=30000),
        use_screenshot=False,
        use_axtree=True,
        use_html=False,
        pre_observation_delay=0.5,
    )

    benchmark = WorkArenaBenchmark(default_tool_config=tool_config, level="l1", n_seeds_l1=1)
    benchmark.setup()
    benchmark = benchmark.subset_from_list(_TARGET_TASK_IDS, benchmark_name_suffix="subset")

    exp = Experiment(
        name="workarena_l1_subset",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=100,
    )

    if debug:
        run_sequentially(exp, debug_limit=2)
    else:
        run_with_ray(exp, n_cpus=5)


if __name__ == "__main__":
    debug = sys.argv[-1] == "debug"
    main(debug)
