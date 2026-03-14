# cube-harness

Open source harness for building and evaluating AI agents using the [CUBE Standard](https://github.com/The-AI-Alliance/cube-standard).

**[CUBE Standard](https://github.com/The-AI-Alliance/cube-standard)** defines the benchmark protocol. **cube-harness** is the evaluation runtime: it runs agents against any CUBE-compatible benchmark, records trajectories, and scales execution with Ray.

> [!NOTE]
> **cube-harness is in active development (alpha).** Interfaces may change. We welcome early adopters and contributors who want to shape the framework, not just use it.
> See our [Roadmap](ROADMAP.md) and [Contributing Guide](CONTRIBUTING.md). Serious contributors can [apply here](https://forms.gle/JFiBi4ynfVLMghAH8).

<!-- [Published Documentation](https://the-ai-alliance.github.io/cube-harness/) -->

## Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/The-AI-Alliance/cube-harness.git
cd cube-harness

# Install dependencies
make install
```

### API Keys

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your-key-here
```

Any [LiteLLM-supported provider](https://docs.litellm.ai/docs/providers) works — just change `model_name` in the recipe.

### Run Tests

```bash
make test
```

### Run Hello Example

The [`hello_miniwob`](recipes/hello_miniwob.py) recipe demonstrates running a ReAct agent on the MiniWob benchmark.

**Start here** — 2 tasks, sequential (fast, no Ray required):

```bash
make debug
# or: uv run recipes/hello_miniwob.py debug
```

Full benchmark (all 125 tasks, parallel via Ray):

```bash
make hello
```

This will:
1. Launch a headless browser environment
2. Run a ReAct agent powered by GPT-5.2-mini on MiniWob tasks
3. Save trajectories and results to `~/cube_harness_results/al2/hello_miniwob/`

### Configuration

Recipes are the configuration. Copy one from [`recipes/`](recipes/), edit what you need, and run it. Config objects are typed Pydantic models — serialized to disk with every experiment so results are always reproducible.

See **[docs/configuration.md](docs/configuration.md)** for the full philosophy, a comparison with Hydra/YAML/CLI approaches, and how to run sweeps.

## Experiment Viewer

cube-harness includes a Gradio-based UI for exploring experiment results and trajectories:

```bash
make viewer
# or: uv run ch-viewer
```

The viewer displays:
- **Trajectory list** — all runs with task ID, steps, reward, and duration
- **Visual timeline** — color-coded steps (blue=environment, green=agent) with duration-based widths
- **Screenshots** — environment state at each step
- **Step details** — observations, agent actions, and LLM reasoning
- **Debug data** — raw JSON, LLM calls, and tool configurations

![cube-harness Viewer Screenshot](docs/assets/images/al2_viewer.png)

## Architecture Overview

cube-harness is a **universal evaluation platform** for agentic benchmarks and an **RL data generation** framework built on top of the CUBE Standard.

### Core Components

![cube-harness Overview](docs/assets/images/al2_overview.png)

- **Agent** — LLM-powered decision maker that receives observations and produces actions
- **Environment** — Executes actions, provides observations and rewards (tool + task composition)
- **Tool** — Modular action provider that exposes an action space, reusable across benchmarks
- **ActionSpace** — Defines the set of possible actions a tool can execute
- **Task** — Defines goals, validation logic, and action subsets
- **Benchmark** — Collection of tasks; produces env configs for episodes
- **Episode** — Single agent-environment loop for one task; records a trajectory
- **Trajectory** — Stores interaction history (observations, actions, rewards)
- **Experiment** — Coordinates execution of multiple episodes across a benchmark
- **ExpRunner** — Execution runtime (sequential or parallel via Ray)

### Design Goals

1. **Benchmark Agnostic** — Plug in any CUBE-standard benchmark (MiniWob, WebArena, OSWorld, …) via the `Benchmark` interface
2. **Agent Agnostic** — Support any agent architecture by implementing the `Agent` protocol
3. **RL-Ready** — Trajectory format designed for training data generation with full LLM call logging
4. **Scalable** — Ray integration for parallel episode execution across multiple workers
5. **Observable** — Structured trajectory output for analysis and debugging

## Development

```bash
make format    # Format code
make lint      # Lint and auto-fix
make help      # Show all commands
make test      # Run tests
make coverage  # Run tests with coverage report
```

## Project Structure

```
cube-harness/
├── src/cube_harness/   # Source code for the framework
├── tests/              # Test suite
├── recipes/            # Example recipes and configurations
├── docs/               # Project documentation
└── Makefile            # Common task shortcuts
```

## Getting Involved

All contributions are welcome — open an issue, submit a PR, or wrap a new benchmark. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development guide, DCO requirements, and RFC process.

Want deeper involvement? Join the core team, shape the roadmap, and get credit for what you build. [Apply here](https://forms.gle/JFiBi4ynfVLMghAH8).

For general AI Alliance contribution guidelines, see the [community repo](https://github.com/The-AI-Alliance/community/).
