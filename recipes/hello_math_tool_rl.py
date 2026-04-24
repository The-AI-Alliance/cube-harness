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
    uv run recipes/hello_math_tool.py full --model openai/Qwen2.5-7B-Instruct
    uv run recipes/hello_math_tool.py full --base-url http://localhost:8000/v1
"""

import argparse
import os

from math_tool_use import MathToolUseBenchmark, MathToolUseToolConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.tir import TirAgentConfig
from cube_harness.exp_runner import run_sequentially
from cube_harness.experiment import Experiment
from cube_harness.episode import Episode, MAX_STEPS
from cube_harness.llm import LLMConfig

_SYSTEM_PROMPT = """You are a math-focused AI Agent. Solve problems by combining clear symbolic reasoning
with short, deterministic Python code.
Keep your replies concise and direct. Prioritize clarity and avoid over-elaboration.
Always present the final answer in LaTeX \\boxed{}.
Do not express emotions or opinions about user questions.

Workflow:
1. Draft a brief plan in plain text.
2. Execute one run_python_code call to compute or verify the result.
3. Finalize by calling MathAnswer with the LaTeX-formatted answer.

Python execution policy (run_python_code):
- Use Python strictly for pure computation to verify and validate the final answer.
- No network, file system, OS or environment access.
- Keep snippets minimal and self-contained; print only the final result.

Validation:
- Cross-check results (alternative derivation, invariants, higher precision) before finalizing.
- If execution fails, propose the minimal fix and retry.
Always verify with run_python_code before invoking MathAnswer."""


def main(mode: str, model: str, base_url: str, sandbox_endpoint: str, max_completion_tokens: int) -> None:
    api_key = "EMPTY"
    # output_dir = make_experiment_output_dir("tir", "math-tool-use", tag="vllm")

    llm_config = LLMConfig(
        model_name=model,
        api_key=api_key,
        api_base=base_url,
        temperature=1.0,
        max_completion_tokens=max_completion_tokens,
        ## Important for accurate RL data collection:
        training=True,  # ensure cache is disabled for accurate RL data collection
        logprobs=True,
        top_logprobs=1,
        extra_body={"return_token_ids": True},
    )

    agent_config = TirAgentConfig(
        llm_config=llm_config,
        system_prompt=_SYSTEM_PROMPT,
        max_actions=3,
    )

    tool_config = MathToolUseToolConfig(sandbox_endpoint=sandbox_endpoint)
    benchmark = MathToolUseBenchmark(default_tool_config=tool_config)
    benchmark.install()
    benchmark.setup()

    if mode == "debug":
        benchmark = benchmark.subset_from_list(["q_0", "q_1", "q_2", "q_3"], benchmark_name_suffix="debug")

    # exp = Experiment(
    #     name="math-tool-use",
    #     output_dir=output_dir,
    #     agent_config=agent_config,
    #     benchmark=benchmark,
    #     max_steps=3,
    # )

    # run_sequentially(exp)

    task_configs = list(benchmark.get_task_configs())
    episodes = [
        Episode(
            id=i,
            output_dir="",
            agent_config=agent_config,
            task_config=tc,
            exp_name="default",
            max_steps=MAX_STEPS,
            persist_episode=False,
            runtime_context=benchmark._runtime_context,
            container_backend=benchmark.container_backend,
        )
        for i, tc in enumerate(task_configs)
    ]
    for episode in episodes:
        tj = episode.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run math-tool-use against a self-hosted vLLM endpoint")
    parser.add_argument("mode", nargs="?", default="debug", choices=["debug", "full"])
    parser.add_argument("--model", default="openai/Qwen2.5-7B-Instruct")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--sandbox-endpoint", default="http://dns-24e3447c-506e-4b21-92df-156e18db5087-sandboxfusion")
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    args = parser.parse_args()

    main(
        mode=args.mode,
        model=args.model,
        base_url=args.base_url,
        sandbox_endpoint=args.sandbox_endpoint,
        max_completion_tokens=args.max_completion_tokens,
    )


# Example vLLM launch for parity with PipelineRL config:
# vllm serve /mnt/llmd/base_models/Qwen2.5-7B-Instruct \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --api-key EMPTY \
#   --enable-auto-tool-choice \
#   --tool-call-parser rl_tool \
#   --tool-parser-plugin pipelinerl/rl_tool_parser_plugin.py \
#   --served-model-name Qwen2.5-7B-Instruct \
#   --return-tokens-as-token-ids
