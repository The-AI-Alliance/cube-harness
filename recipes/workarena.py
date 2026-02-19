"""Example recipe for running WorkArena benchmark with AgentLab2.

This recipe demonstrates how to run WorkArena tasks using the BrowserGym tool.

Prerequisites:
    1. Install WorkArena: pip install browsergym-workarena
    2. Configure ServiceNow credentials via environment variables:
       - SNOW_INSTANCE_URL: ServiceNow instance URL
       - SNOW_INSTANCE_UNAME: ServiceNow username
       - SNOW_INSTANCE_PWD: ServiceNow password
       OR
       - HUGGING_FACE_HUB_TOKEN: For accessing gated instance pool

Usage:
    # Debug mode (2 tasks, sequential)
    uv run recipes/hello_workarena.py debug

    # Full run (parallel with Ray)
    uv run recipes/hello_workarena.py
"""

import sys

from agentlab2 import make_experiment_output_dir
from agentlab2.agents.react import ReactAgentConfig
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from agentlab2.tools.browsergym import BrowsergymConfig

try:
    from agentlab2.benchmarks.workarena import WorkArenaBenchmark
except ImportError:
    print(
        "WorkArena benchmark requires 'browsergym-workarena'. Run `make install` to install all optional dependencies."
    )
    sys.exit(1)


def main(debug: bool):
    output_dir = make_experiment_output_dir("react", "workarena", tag="l1")

    # Configure LLM
    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = ReactAgentConfig(
        render_last_n_steps=2,
        max_actions=20,
        llm_config=llm_config,
    )

    # Configure BrowserGym tool
    # Note: task_entrypoint and task_kwargs are set dynamically by WorkArenaTask.setup()
    tool_config = BrowsergymConfig(
        headless=not debug,  # Show browser in debug mode
        use_screenshot=True,
        use_axtree=True,
        use_html=False,
    )

    # Configure WorkArena benchmark
    benchmark = WorkArenaBenchmark(
        tool_config=tool_config,
        level="l1",
        n_seeds_l1=2 if debug else 5,  # Fewer seeds in debug mode
    )

    # Create experiment
    exp = Experiment(
        name="workarena",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    # Run experiment
    if debug:
        run_sequentially(exp, debug_limit=2)
    else:
        run_with_ray(exp, n_cpus=1)


if __name__ == "__main__":
    debug = sys.argv[-1] == "debug"
    main(debug)
