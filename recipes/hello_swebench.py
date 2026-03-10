"""Minimal recipe to run SWE-bench Verified tasks.

Usage
-----
    DAYTONA_API_KEY=... uv run recipes/hello_swebench.py
"""

import logging
from pathlib import Path

from cube.backends.daytona import DaytonaContainerBackend
from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmark

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.exp_runner import run_sequentially
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

backend = DaytonaContainerBackend()

benchmark = SWEBenchVerifiedBenchmark(
    container_backend=backend,
    max_tasks=2,
    repo_filter="django/django",
)
benchmark.install()
benchmark.setup()

agent_config = ReactAgentConfig(
    llm_config=LLMConfig(model_name="gpt-4o"),
)

experiment = Experiment(
    name="swebench_verified_test",
    output_dir=Path("results/swebench_verified_test"),
    agent_config=agent_config,
    benchmark=benchmark,
    max_steps=100,
)

run_sequentially(experiment)
