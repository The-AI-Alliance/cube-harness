# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "math-tool-use",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# math-tool-use = { path = "../cubes/math-tool-use", editable = true }
# ///

"""Run math-tool-use with a self-hosted OpenAI-compatible vLLM endpoint.

Usage:
    uv run recipes/hello_math_tool.py debug
    uv run recipes/hello_math_tool.py full --model openai/Qwen/Qwen2.5-Math-1.5B
    uv run recipes/hello_math_tool.py full --base-url http://localhost:8000/v1
"""

import argparse

from cube import tool


from math_tool_use import MathToolUseBenchmark, MathToolUseToolConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.react import ReactAgentConfig
from cube_harness.exp_runner import run_sequentially
from cube_harness.experiment import Experiment
from cube_harness.llm import RLCollectorConfig

def main(mode: str, model: str, base_url: str) -> None:
    api_key = "EMPTY"

    # # LiteLLM reads OPENAI_API_BASE / OPENAI_API_KEY for openai/* models.
    # os.environ["OPENAI_API_BASE"] = base_url
    # os.environ["OPENAI_API_KEY"] = api_key

    # # Keep OpenAI python client compatibility if anything else in-process uses it.
    # os.environ["OPENAI_BASE_URL"] = base_url

    output_dir = make_experiment_output_dir("react", "math-tool-use", tag="vllm")

    instructions = (
            "Workflow (required):\\n"
            "1. Draft a brief plan in plain text.\\n"
            "2. Execute one run_python_code call to compute the result and print it.\\n"
            "3. Finalize by calling MathAnswer with the LaTeX-formatted answer (use \\\\boxed{...})."
        )

    llm_config = RLCollectorConfig(
        model_name=model,
        api_key=api_key,
        api_base=base_url,
        temperature=1.0,
        max_completion_tokens=2048,
        logprobs=True,
        top_logprobs=True,
        extra_body={"return_token_ids": True},
    )

    agent_config = ReactAgentConfig(llm_config=llm_config, react_prompt=instructions)
    tool_config = MathToolUseToolConfig(sandbox_endpoint="http://dns-24e3447c-506e-4b21-92df-156e18db5087-sandboxfusion")
    benchmark = MathToolUseBenchmark(default_tool_config=tool_config)
    benchmark.install()
    benchmark.setup()
    if mode == "debug":
        benchmark = benchmark.subset_from_list(["q_0", "q_1"], benchmark_name_suffix="debug")

    exp = Experiment(
        name="math-tool-use",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
        max_steps=3,
    )

    run_sequentially(exp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run math-tool-use against a self-hosted vLLM endpoint")
    parser.add_argument("mode", nargs="?", default="debug", choices=["debug", "full"])
    parser.add_argument("--model", default="openai/Qwen2.5-7B-Instruct")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    args = parser.parse_args()

    main(mode=args.mode, model=args.model, base_url=args.base_url)


# vllm serve /mnt/llmd/base_models/Qwen2.5-7B-Instruct \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --api-key EMPTY \
#   --enable-auto-tool-choice \
#   --tool-call-parser hermes \
#   --served-model-name Qwen2.5-7B-Instruct
#   --return-tokens-as-token-ids