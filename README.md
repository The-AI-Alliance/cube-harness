# AgentLab2

Open source framework for building and evaluating UI agents.
<!-- [Published Documentation](https://the-ai-alliance.github.io/AgentLab2/) -->
Published Documentation - Coming soon.

## Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/The-AI-Alliance/AgentLab2.git
cd AgentLab2

# Install dependencies
make install
```

### Run Hello Example

The [`hello_miniwob`](recipes/hello_miniwob.py) recipe demonstrates running a ReAct agent on the MiniWob benchmark:

```bash
# Run full benchmark (all 125 tasks, parallel execution with Ray)
uv run recipes/hello_miniwob.py
```
or you can use the Makefile shortcut:
```bash
make hello
```

For quicker debugging, you can run a smaller subset of tasks sequentially:
```bash
# Run in debug mode (2 tasks, sequential execution)
make debug
# or: uv run recipes/hello_miniwob.py debug
```

This will:
1. Launch a headless browser environment
2. Run a ReAct agent powered by GPT-5-mini on MiniWob tasks
3. Save trajectories and results to `~/agentlab_results/al2/hello_miniwob/`

### Configuration

You can customize the experiment by editing recipe [`recipes/hello_miniwob.py`](recipes/hello_miniwob.py) or making your own recipe. Key configuration options include:
- `LLMConfig` - model name, temperature, max tokens
- `BrowserEnvConfig` - env class and its options: headless mode, screenshot capture, html pruning etc.
- `ReactAgentConfig` - agent behavior
- `MiniWobBenchmark` - bencmark selection

## Architecture Overview

AgentLab2 is designed as a **universal evaluation platform** for multiple agentic benchmarks and serves as the foundation for **RL data generation** pipelines.

### Core Components

![AgentLab2 Overview](docs/assets/images/al2_overview.png)

- **Agent** - LLM-powered decision maker that receives observations and produces actions
- **Environment** - Provides observations, executes actions, returns rewards
- **Tool** - Modular action provider; environments (e.g., ToolboxEnv) compose multiple tools to build their action space
- **Task** - Defines goals, validation logic, and action filtering
- **Benchmark** - Collection of tasks with shared environment configuration
- **Episode** - Single agent-task execution with trajectory recording
- **Trajectory** - Stores interaction history (observations, actions, rewards) for RL training
- **Experiment** - Coordinates multiple episodes across a benchmark
- **ExpRunner** - Execution runtime (sequential or parallel via Ray)

### Key Abstractions

| Module | Purpose |
|--------|---------|
| `core.py` | Data structures: Action, Observation, Trajectory, AgentOutput |
| `agent.py` | Agent protocol and configuration |
| `environment.py` | Environment, Task, and ToolboxEnv abstractions |
| `benchmark.py` | Benchmark interface for task collections |
| `llm.py` | LLM wrapper using LiteLLM (supports OpenAI, Anthropic, etc.) |
| `tool.py` | Tool abstraction for action spaces |
| `episode.py` | Episode execution and trajectory persistence |
| `experiment.py` | Experiment configuration and statistics |
| `exp_runner.py` | Sequential and Ray-based parallel execution |

### Design Goals

1. **Benchmark Agnostic** - Plug in any benchmark (MiniWob, WebArena, OSWorld, etc.) via the `Benchmark` interface
2. **Agent Agnostic** - Support any agent architecture by implementing the `Agent` protocol
3. **RL-Ready** - Trajectory format designed for training data generation with full LLM call logging
4. **Scalable** - Ray integration for parallel episode execution across multiple workers
5. **Observable** - Structured trajectory output for analysis and debugging

## Development

```bash
make format    # Format code
make lint      # Lint and auto-fix
make help      # Show all commands
```

## Project Structure

```
AgentLab2/
├── src/agentlab2/       # Source code for the framework
├── tests/               # Test suite
├── recipes/             # Example recipes and configurations
├── docs/                # Project documentation
├── pyproject.toml       # Package metadata and dependencies
├── Makefile             # Common development tasks
└── .vscode/             # VS Code settings (Ruff formatter/linter)
```

The rest of this README provides information for contributors, developers, and users of this project repo.

<!-- ## Getting Involved

We welcome contributions as PRs. Please see our [Alliance community repo](https://github.com/The-AI-Alliance/community/) for general information about contributing to any of our projects. This section provides some specific details you need to know.

In particular, see the AI Alliance [CONTRIBUTING](https://github.com/The-AI-Alliance/community/blob/main/CONTRIBUTING.md) instructions. You will need to agree with the AI Alliance [Code of Conduct](https://github.com/The-AI-Alliance/community/blob/main/CODE_OF_CONDUCT.md).

All _code_ contributions are licensed under the [Apache 2.0 LICENSE](https://github.com/The-AI-Alliance/community/blob/main/LICENSE.Apache-2.0) (which is also in this repo, [LICENSE.Apache-2.0](LICENSE.Apache-2.0)).

All _documentation_ contributions are licensed under the [Creative Commons Attribution 4.0 International](https://github.com/The-AI-Alliance/community/blob/main/LICENSE.CC-BY-4.0) (which is also in this repo, [LICENSE.CC-BY-4.0](LICENSE.CC-BY-4.0)).

All _data_ contributions are licensed under the [Community Data License Agreement - Permissive - Version 2.0](https://github.com/The-AI-Alliance/community/blob/main/LICENSE.CDLA-2.0) (which is also in this repo, [LICENSE.CDLA-2.0](LICENSE.CDLA-2.0)).

### We use the "Developer Certificate of Origin" (DCO).

> [!WARNING]
> Before you make any git commits with changes, understand what's required for DCO.

See the Alliance contributing guide [section on DCO](https://github.com/The-AI-Alliance/community/blob/main/CONTRIBUTING.md#developer-certificate-of-origin) for details. In practical terms, supporting this requirement means you must use the `-s` flag with your `git commit` commands.

## Documentation

Project documentation is available in the [docs/](docs/) directory. -->
