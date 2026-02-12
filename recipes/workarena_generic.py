"""Example recipe for running WorkArena benchmark with GenericAgent.

This recipe demonstrates how to run WorkArena tasks using the legacy GenericAgent
with text-based prompting and XML-like tags for structured output.

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
    uv run recipes/workarena_generic.py debug

    # Full run (parallel with Ray)
    uv run recipes/workarena_generic.py
"""

import sys
import time
from pathlib import Path

from agentlab2.agents.legacy_generic_agent import GenericAgentConfig, GenericPromptFlags, ObsFlags
from agentlab2.benchmarks.workarena import WorkArenaBenchmark
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from agentlab2.tools.browsergym import BrowsergymConfig


def main(debug: bool) -> None:
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path.home() / "agentlab_results" / "al2" / f"workarena_generic_{current_datetime}"

    # Configure LLM
    llm_config = LLMConfig(
        model_name="azure/gpt-5-mini",
        temperature=1.0,
        reasoning_effort="medium",
        parallel_tool_calls=True,
    )

    # Observation flags - these control what the agent uses in prompts
    # IMPORTANT: BrowsergymConfig must provide at least what ObsFlags expects
    use_html = False
    use_axtree = True
    use_screenshot = False

    # Configure GenericAgent with flags matching FLAGS_CUSTOM from original agentlab
    agent_config = GenericAgentConfig(
        llm_config=llm_config,
        max_actions=20,
        flags=GenericPromptFlags(
            obs=ObsFlags(
                use_html=use_html,
                use_ax_tree=use_axtree,
                use_focused_element=True,
                use_error_logs=True,
                use_history=True,
                use_past_error_logs=False,
                use_action_history=True,
                use_think_history=True,
                use_diff=False,
                html_type="pruned_html",
                use_screenshot=use_screenshot,
                use_som=False,
                extract_visible_tag=True,
                extract_clickable_tag=True,
                extract_coords="False",
                filter_visible_elements_only=False,
            ),
            use_plan=False,
            use_criticise=False,
            use_thinking=True,
            use_memory=False,
            use_concrete_example=True,
            use_abstract_example=True,
            use_hints=True,
            enable_chat=False,
            max_prompt_tokens=40_000,
            be_cautious=True,
            extra_instructions=None,
        ),
    )

    # Configure BrowserGym tool
    # Note: task_entrypoint and task_kwargs are set dynamically by WorkArenaTask.setup()
    # These flags must be >= the agent's ObsFlags to provide required observations
    tool_config = BrowsergymConfig(
        headless=not debug,  # Show browser in debug mode
        use_screenshot=True,  # Always save screenshot in trajectory for debug
        use_axtree=use_axtree,
        use_html=use_html,
    )

    # Configure WorkArena benchmark
    benchmark = WorkArenaBenchmark(
        tool_config=tool_config,
        level="l1",
        n_seeds_l1=2 if debug else 5,  # Fewer seeds in debug mode
    )

    # Create experiment
    exp = Experiment(
        name="workarena_generic",
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
