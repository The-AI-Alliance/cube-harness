"""Test SWE-bench Verified cube: 10 tasks in parallel with React agent.

Usage
-----
    DAYTONA_API_KEY=... OPENAI_API_KEY=... uv run recipes/test_swebench_10.py
"""

import logging
from pathlib import Path

from cube.backends.daytona import DaytonaContainerBackend

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.exp_runner import run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

backend = DaytonaContainerBackend()

benchmark = SWEBenchVerifiedBenchmark(
    container_backend=backend,
    max_tasks=10,
)

agent_config = ReactAgentConfig(
    llm_config=LLMConfig(model_name="gpt-4o-mini"),
)

experiment = Experiment(
    name="swebench_verified_10",
    output_dir=Path("results/swebench_verified_10"),
    agent_config=agent_config,
    benchmark=benchmark,
    max_steps=30,
)

results = run_with_ray(experiment, n_cpus=10, episode_timeout=1800.0)
