"""Minimal recipe to run SWE-bench Verified tasks.

Usage
-----
    DAYTONA_API_KEY=... OPENAI_API_KEY=... uv run recipes/hello_swebench.py
"""

import logging
from pathlib import Path

from cube.backends.daytona import DaytonaContainerBackend

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.exp_runner import run_sequentially
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

SWE_SYSTEM_PROMPT = """\
You are an autonomous coding agent. You have access to a Linux sandbox with the repository already cloned at /testbed.
Your task is to resolve the GitHub issue described below. Use the provided tools to explore the codebase, \
understand the problem, and implement a fix.
Start by exploring the repository structure and reading relevant files before making changes.
When you are confident the fix is correct, call final_step to submit."""

backend = DaytonaContainerBackend()

benchmark = SWEBenchVerifiedBenchmark(
    container_backend=backend,
    max_tasks=2,
    repo_filter="django/django",
)

agent_config = ReactAgentConfig(
    llm_config=LLMConfig(model_name="gpt-4.1-mini"),
    system_prompt=SWE_SYSTEM_PROMPT,
)

experiment = Experiment(
    name="swebench_verified_test",
    output_dir=Path("results/swebench_verified_test"),
    agent_config=agent_config,
    benchmark=benchmark,
    max_steps=100,
)

run_sequentially(experiment)
