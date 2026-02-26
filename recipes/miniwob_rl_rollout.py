"""Run a single MiniWob rollout with vllm model and save the training data to JSON."""

import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.miniwob.benchmark import MiniWobBenchmark
from agentlab2.llm import LLMConfig
from agentlab2.rl.rollout import rollout
from agentlab2.tools.playwright import PlaywrightConfig

load_dotenv()

LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d %(funcName)s() - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

VLLM_BASE_URL = os.environ["VLLM_BASE_URL"]
VLLM_API_KEY = os.environ["VLLM_API_KEY"]
# model = "openai/ServiceNow-AI/Apriel-1.6-15b-Thinker"
model = "openai/Qwen/Qwen3-4B-Thinking-2507"


def main(task_index: int = 0) -> None:
    llm_config = LLMConfig(
        model_name=model,
        api_base=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        temperature=1.0,
        num_retries=0,
    )
    agent_config = ReactAgentConfig(llm_config=llm_config)

    tool_config = PlaywrightConfig(use_screenshot=False, headless=True)
    benchmark = MiniWobBenchmark(tool_config=tool_config)

    benchmark.setup()
    try:
        env_configs = benchmark.env_configs()
        env_config = env_configs[task_index]
        print(f"Running rollout on task: {env_config.task.id}")

        result = rollout(agent_config=agent_config, env_config=env_config)

        current_datetime = time.strftime("%Y%m%d_%H%M%S")
        output_path = (
            Path.home() / "agentlab_results" / "rollouts" / f"rollout_{env_config.task.id}_{current_datetime}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(result), indent=2) + "\n")
        print(f"Saved rollout result to {output_path}")
        print(f"Reward: {result.reward}, Text pairs: {len(result.text_pairs)}")
    finally:
        benchmark.close()


if __name__ == "__main__":
    task_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(task_index=task_idx)
