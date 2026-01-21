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

### API Keys
* Set up your api keys in your `.env` of this project
* Current hello world recipe is using an azure endpoint. You may change it for your needs (don't commit)
* For azure, you need to provide these:
```bash
export AZURE_API_KEY=
export AZURE_API_BASE=
export AZURE_API_VERSION=
```

### Run Tests
```bash
make test
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
- **Environment** - Executes actions, provides observations and rewards. Created through composition of the tool and the task
- **Tool** - Modular action provider that exposes an action space and can be reused across different benchmarks (e.g., web browser, OS container)
- **ActionSpace** - Defines the set of possible actions that a tool can execute
- **Task** - Defines goals, validation logic, and action subsets. Sets up the tool for a specific scenario
- **Benchmark** - Collection of tasks with methods for common setup/teardown; produces env configs for episodes
- **Episode** - Single agent-environment execution loop for one task, records trajectory
- **Trajectory** - Stores interaction history (observations, actions, rewards)
- **Experiment** - Coordinates execution of multiple episodes across a benchmark and collects results
- **ExpRunner** - Execution runtime (sequential or parallel via Ray)


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
make test      # Run tests
make coverage  # Run tests with coverage report
```

## Project Structure

```
AgentLab2/
├── src/agentlab2/       # Source code for the framework
├── tests/               # Test suite
├── recipes/             # Example recipes and configurations
├── docs/                # Project documentation
└── Makefile             # Common task shortcuts
```