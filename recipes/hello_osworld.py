"""Hello OSWorld - Example script for running OSWorld benchmark.

OSWorld evaluates agents on real desktop automation tasks using VMs/Docker.

Prerequisites:
1. Install desktop_env: pip install desktop-env
2. That's it! OSWorld repo clones automatically on first run.

First-time run:
- OSWorld repo is cloned to ~/.agentlab2/benchmarks/osworld/OSWorld
- VM images (~23GB Ubuntu) download automatically on first task
- Everything is cached for subsequent runs

Usage:
    # Debug mode (2 tasks, sequential, verbose logging)
    python recipes/hello_osworld.py debug

    # Full run (all tasks, parallel with Ray)
    python recipes/hello_osworld.py

Optional pre-installation:
    from agentlab2.benchmarks.osworld.benchmark import OSWorldBenchmark
    OSWorldBenchmark().install()  # Pre-clone repo (auto-runs anyway)

Note: Without desktop_env installed, this will fail with ImportError.
"""

import sys
import time
from pathlib import Path

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.osworld.benchmark import OSWorldBenchmark
from agentlab2.exp_runner import run_sequentially, run_with_ray
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from agentlab2.tools.computer import ComputerConfig


def main(debug: bool):
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path.home() / "agentlab_results" / "al2" / f"osworld_{current_datetime}"

    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = ReactAgentConfig(llm_config=llm_config)

    # Configure Computer tool for OSWorld (VM/desktop automation)
    tool_config = ComputerConfig(
        provider="docker",  # Use Docker (fast) or "vmware" (full desktop)
        headless=True,
        require_a11y_tree=True,
        screen_size=(1920, 1080),
        observe_after_action=True,  # Default Fase, set to True for debugging/visualization (adds ~1-2s per action)
    )

    # Configure OSWorld benchmark (paths are automatic now)
    benchmark = OSWorldBenchmark(
        tool_config=tool_config,
        domain="all",  # or specific: "chrome", "os", "libreoffice"
        shuffle=True,
        test_set_name="test_all.json",  # or "test_small.json" for quick testing
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
        print(f"Provider: {tool_config.provider}")
        print(f"Domain: {benchmark.domain}")
        print(f"VM will start, wait 60s, then agent can interact")
        print("=" * 60 + "\n")
        run_sequentially(exp, debug_limit=2)
    else:
        print(f"\nRunning OSWorld benchmark with Ray (4 CPUs)")
        print(f"Output directory: {output_dir}\n")
        run_with_ray(exp, n_cpus=4)


if __name__ == "__main__":
    debug = sys.argv[-1] == "debug"
    main(debug)
