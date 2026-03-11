"""Hello OSWorld - Example script for running OSWorld benchmark.

OSWorld evaluates agents on real desktop automation tasks using Docker+QEMU VMs.
This recipe uses cube-computer-tool for VM management (no desktop_env dependency).

Prerequisites:
1. Docker installed and running
2. /dev/kvm available (for hardware acceleration on Linux)
3. pip install -e osworld-cube/  (installs this package + its dependencies)

First-time run:
- OSWorld repo is cloned to ~/.agentlab2/benchmarks/osworld/OSWorld
- Ubuntu VM disk image (~23GB) is downloaded to ~/.cube/osworld/ on first make()
- Everything is cached for subsequent runs

Usage:
    # Debug mode (2 tasks, sequential, verbose logging)
    python recipes/hello_osworld.py debug

    # Full run (all tasks, parallel with Ray)
    python recipes/hello_osworld.py
"""

import sys
import time
from pathlib import Path

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from osworld_cube import OSWORLD_SYSTEM_PROMPT_COMPUTER_13, OSWorldBenchmark, OSWorldComputerConfig


def main(debug: bool):
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path.home() / "agentlab_results" / "al2" / f"osworld_{current_datetime}"

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = ReactAgentConfig(
        llm_config=llm_config,
        max_actions=15,
        system_prompt=OSWORLD_SYSTEM_PROMPT_COMPUTER_13,
    )

    # Configure Computer tool for OSWorld
    # vm_image_path is auto-downloaded from HuggingFace if not provided
    tool_config = OSWorldComputerConfig(
        headless=True,
        require_a11y_tree=True,
        observe_after_action=True,
    )

    # Configure OSWorld benchmark
    benchmark = OSWorldBenchmark(
        tool_config=tool_config,
        tasks_file="osworld-cube/src/osworld_cube/osworld_tasks.json",
        domain="all",
        shuffle=True,
        max_turns=15,
    )

    exp = Experiment(
        name="osworld",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    if debug:
        print("\n" + "=" * 60)
        print("DEBUG MODE: Running 2 tasks sequentially")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Domain: {benchmark.domain}")
        print("VM will start, wait 60s, then agent can interact")
        print("=" * 60 + "\n")
        run_sequentially(exp, debug_limit=2)
    else:
        print("\nRunning OSWorld benchmark with Ray (4 CPUs)")
        print(f"Output directory: {output_dir}\n")
        run_with_ray(exp, n_cpus=4)


if __name__ == "__main__":
    debug = sys.argv[-1] == "debug"
    main(debug)
