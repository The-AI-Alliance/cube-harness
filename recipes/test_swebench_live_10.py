"""Test SWE-bench Live cube: 10 tasks in parallel with React agent.

Usage
-----
    DAYTONA_API_KEY=... OPENAI_API_KEY=... uv run recipes/test_swebench_live_10.py
"""

import logging
from pathlib import Path

from cube.backends.daytona import DaytonaContainerBackend

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.exp_runner import run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from swebench_live_cube.benchmark import SWEBenchLiveBenchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

SWE_SYSTEM_PROMPT = """\
You are an autonomous coding agent. You have access to a Linux sandbox with the repository already cloned at /testbed.
Your task is to resolve the GitHub issue described below. Use the provided tools to explore the codebase, \
understand the problem, and implement a fix.
Start by exploring the repository structure and reading relevant files before making changes.
When you are confident the fix is correct, call final_step to submit."""

backend = DaytonaContainerBackend()

benchmark = SWEBenchLiveBenchmark(
    container_backend=backend,
    max_tasks=10,
    split="lite",
)

agent_config = ReactAgentConfig(
    llm_config=LLMConfig(model_name="gpt-4.1-mini"),
    system_prompt=SWE_SYSTEM_PROMPT,
)

experiment = Experiment(
    name="swebench_live_10",
    output_dir=Path("results/swebench_live_10"),
    agent_config=agent_config,
    benchmark=benchmark,
    max_steps=30,
)

results = run_with_ray(experiment, n_cpus=10, episode_timeout=1800.0)
