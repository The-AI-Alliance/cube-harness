"""Meta-agent recipe — starting point for the meta-agent improvement loop.

The meta-agent iteratively picks task subsets, runs evaluations, analyses failures,
and applies targeted fixes. This recipe is the entry point for each eval round.

Usage:
    uv run recipes/meta_agent_recipe.py          # full run
    uv run recipes/meta_agent_recipe.py debug    # sequential, 2 tasks max

Customising the task subset:
    Pass a list of task IDs to `task_ids` in `MiniWobSubset`. The meta-agent
    will update this list each iteration based on failure analysis.

Adding task hints:
    Populate `GennyConfig.task_hints` with task_id → hint text.
    The meta-agent adds entries here after diagnosing why a specific task fails.
    `GennyConfig.hint` can also be set as a general fallback for the whole subset.
"""

import sys

from cube_browser_tool import PlaywrightConfig
from miniwob_cube.benchmark import MiniWobBenchmark

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import GennyConfig
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig


class MiniWobSubset(MiniWobBenchmark):
    """MiniWob benchmark filtered to a specific subset of task IDs.

    The meta-agent targets a small, informative task set each iteration.
    Update `task_ids` to focus on tasks that are currently failing or
    have high diagnostic value.

    An empty list means all tasks (full benchmark).
    """

    task_ids: list[str] = []

    def get_task_configs(self):  # type: ignore[override]
        for tc in super().get_task_configs():
            if not self.task_ids or tc.task_id in self.task_ids:
                yield tc


def main(debug: bool) -> None:
    # --- Task subset ---
    # Update this list each meta-agent iteration to focus on failing tasks.
    # Example: ["click-button", "login-user", "search-engine"]
    task_ids: list[str] = [
        "click-button",
        "click-checkboxes",
    ]

    # --- Agent config ---
    # `task_hints` maps task_id → hint injected after the goal every step.
    # `hint` is a fallback applied to all tasks in this run when no specific match.
    # The meta-agent populates these after diagnosing failure traces.
    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = GennyConfig(
        llm_config=llm_config,
        hint="",  # general hint for this task subset
        task_hints={
            # "click-button": "The submit button always has id='subbtn'.",
        },
    )

    tool_config = PlaywrightConfig(use_screenshot=True, headless=True)
    benchmark = MiniWobSubset(default_tool_config=tool_config, task_ids=task_ids)

    output_dir = make_experiment_output_dir("genny", "miniwob-meta")
    exp = Experiment(
        name="miniwob-meta",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=10,
    )

    if debug:
        run_sequentially(exp, debug_limit=2)
    else:
        run_with_ray(exp, n_cpus=4)


if __name__ == "__main__":
    debug = sys.argv[-1] == "debug"
    main(debug)
