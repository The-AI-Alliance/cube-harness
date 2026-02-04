"""Hello Terminal-Bench recipe.

Terminal-Bench evaluates agents on real-world terminal tasks.

Usage:
    uv run recipes/hello_tbench.py debug                    # 1 task, sequential (Daytona)
    uv run recipes/hello_tbench.py oracle                   # 10 tasks with oracle agent
    uv run recipes/hello_tbench.py oracle --tool docker     # 10 tasks with oracle, local Docker
    uv run recipes/hello_tbench.py oracle_full              # All tasks with oracle agent (30 Ray workers)
    uv run recipes/hello_tbench.py easy                     # 4 easy tasks with react agent
    uv run recipes/hello_tbench.py easy --tool docker       # 4 easy tasks, local Docker
    uv run recipes/hello_tbench.py                          # Full run with Ray

Tracing (Jaeger):
    uv run recipes/hello_tbench.py debug --trace            # Traces to localhost:4318
    uv run recipes/hello_tbench.py debug --trace http://jaeger:4318/v1/traces  # Custom endpoint
"""

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agentlab2.agents.oracle import OracleAgentConfig  # noqa: E402
from agentlab2.agents.react import ReactAgentConfig  # noqa: E402
from agentlab2.benchmarks.terminalbench import TerminalBenchBenchmark  # noqa: E402
from agentlab2.exp_runner import run_sequentially, run_with_ray  # noqa: E402
from agentlab2.experiment import Experiment  # noqa: E402
from agentlab2.llm import LLMConfig  # noqa: E402
from agentlab2.tools.daytona import DaytonaSWEToolConfig  # noqa: E402
from agentlab2.tools.docker import DockerSWEToolConfig  # noqa: E402

VALID_MODES = ("debug", "oracle", "oracle_full", "easy", "full")

SYSTEM_PROMPT = """You are an expert software engineer working in a Linux terminal.
Work in /app directory. Read existing files, test your solutions before declaring completion."""


def create_tool_config(tool: str) -> DaytonaSWEToolConfig | DockerSWEToolConfig:
    """Create the appropriate tool config based on the tool parameter."""
    if tool == "daytona":
        api_key = os.getenv("DAYTONA_API_KEY")
        if not api_key:
            raise ValueError("DAYTONA_API_KEY environment variable is required for Daytona tool")
        return DaytonaSWEToolConfig(api_key=api_key)
    elif tool == "docker":
        return DockerSWEToolConfig(
            pull_policy="missing",  # Pull images only if not present locally
            remove_on_close=True,
            network_mode="bridge",
        )
    else:
        raise ValueError(f"Unknown tool: {tool}. Use 'daytona' or 'docker'")


def main(mode: str, tool: str, otlp_endpoint: str | None = None) -> None:
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/terminalbench") / f"tbench_{mode}_{tool}_{current_datetime}"
    trace_output = str(output_dir / "traces") if otlp_endpoint else None

    tool_config = create_tool_config(tool)
    print(f"Using tool: {tool} ({type(tool_config).__name__})")

    if mode in ("oracle", "oracle_full"):
        agent_config = OracleAgentConfig()
    else:
        llm_config = LLMConfig(model_name="openai/gpt-5-nano", tool_choice="required")
        agent_config = ReactAgentConfig(
            llm_config=llm_config,
            system_prompt=SYSTEM_PROMPT,
            max_actions=100,
            max_obs_chars=200000,
            max_history_tokens=240000,
        )

    benchmark = TerminalBenchBenchmark(
        tool_config=tool_config,
        shuffle=mode != "oracle_full",
        shuffle_seed=42,
        max_tasks={"debug": 1, "oracle": 10, "easy": 4}.get(mode),
        difficulty_filter="easy" if mode == "easy" else None,
        oracle_mode=mode in ("oracle", "oracle_full"),
    )

    exp = Experiment(
        name="terminalbench",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    if mode in ("debug", "oracle", "easy"):
        run_sequentially(
            exp,
            debug_limit=1 if mode == "debug" else None,
            trace_output=trace_output,
            otlp_endpoint=otlp_endpoint,
        )
    elif mode == "oracle_full":
        run_with_ray(exp, n_cpus=30, trace_output=trace_output, otlp_endpoint=otlp_endpoint)
    else:
        run_with_ray(exp, n_cpus=4, trace_output=trace_output, otlp_endpoint=otlp_endpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Terminal-Bench experiments")
    parser.add_argument(
        "mode",
        nargs="?",
        default="full",
        choices=VALID_MODES,
        help="Execution mode (default: full)",
    )
    parser.add_argument(
        "--tool",
        default="daytona",
        choices=["daytona", "docker"],
        help="Tool backend: 'daytona' (cloud) or 'docker' (local). Default: daytona",
    )
    parser.add_argument(
        "--trace",
        nargs="?",
        const="http://localhost:4318/v1/traces",
        default=None,
        metavar="ENDPOINT",
        help="Enable tracing to Jaeger. Default endpoint: http://localhost:4318/v1/traces",
    )
    args = parser.parse_args()
    main(args.mode, args.tool, otlp_endpoint=args.trace)
