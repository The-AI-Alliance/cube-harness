# AgentLab2 - Project Structure for Claude Code

AgentLab2 is an open-source framework for building and evaluating UI agents. It serves as a universal evaluation platform for agentic benchmarks and as the foundation for RL data generation pipelines.

## Directory Structure

```
AgentLab2/
├── src/agentlab2/           # Core framework source code
│   ├── core.py              # Data structures: Action, Observation, Trajectory
│   ├── agent.py             # Agent protocol and configuration
│   ├── environment.py       # Environment, Task, and ToolboxEnv abstractions
│   ├── tool.py              # Tool abstraction for action spaces
│   ├── benchmark.py         # Benchmark interface for task collections
│   ├── llm.py               # LLM wrapper using LiteLLM
│   ├── episode.py           # Episode execution and trajectory persistence
│   ├── experiment.py        # Experiment configuration and statistics
│   ├── exp_runner.py        # Sequential and Ray-based parallel execution
│   ├── storage.py           # Trajectory file storage (JSONL + metadata)
│   ├── viewer.py            # Gradio-based experiment viewer UI
│   ├── utils.py             # HTML pruning utilities
│   ├── agents/              # Agent implementations
│   │   └── react.py         # ReAct agent with LLM-based reasoning
│   ├── envs/                # Environment implementations
│   │   └── browser.py       # Browser environment using Playwright
│   ├── tools/               # Tool implementations
│   │   ├── playwright.py    # Sync/Async Playwright browser tools
│   │   ├── browsergym.py    # BrowserGym integration
│   │   └── computer.py      # Computer use tools
│   ├── action_spaces/       # Action space protocols
│   │   └── browser_action_space.py  # Browser action protocol
│   └── benchmarks/          # Benchmark implementations
│       └── miniwob/         # MiniWob benchmark
│           ├── benchmark.py # MiniWobBenchmark class
│           ├── task.py      # MiniWobTask implementation
│           └── miniwob_tasks.json  # Task definitions
├── recipes/                 # Example experiment recipes
│   └── hello_miniwob.py     # Sample MiniWob experiment
├── tests/                   # Test suite
└── docs/                    # Documentation and assets
```

## Core Abstractions

### Data Flow

```
Benchmark → Episode(s) → Agent ↔ Environment → Trajectory
                            ↑        ↓
                           LLM   Tool(s)
```

### Module Dependencies

```
core.py (base data structures)
    ↑
llm.py (LLM wrapper, used by agents)
    ↑
tool.py (Tool abstraction)
    ↑
environment.py (Environment, Task, ToolboxEnv - uses tools)
    ↑
agent.py (Agent protocol - receives observations, produces actions)
    ↑
benchmark.py (task collections with env configs)
    ↑
episode.py (runs single agent-task execution)
    ↑
experiment.py (coordinates multiple episodes)
    ↑
exp_runner.py (sequential or Ray parallel execution)
```

## Key Classes

### core.py - Data Structures
- **ActionSchema**: Function specification for LLM tool calls (name, description, parameters)
- **Action**: Represents a function call with id, name, and arguments
- **AgentOutput**: Contains actions list and llm_calls for logging
- **Content**: Piece of content (text, image, dict) in an observation, supports tool_call_id
- **Observation**: List of Contents, convertible to LLM messages
- **EnvironmentOutput**: Result of env step (obs, reward, done, info)
- **TrajectoryStep**: Single step with output and timing info
- **Trajectory**: Full interaction history with metadata and reward_info

### agent.py - Agent Protocol
- **AgentConfig**: Abstract base for agent configuration, has `make()` method
- **Agent**: Abstract base with `step(obs) -> AgentOutput` method

### environment.py - Environment Abstractions
- **EnvironmentConfig**: Abstract config with `_task` field and `make()` method
- **Environment**: Abstract base with `setup()`, `step(action)`, `close()`, `action_set()`
- **Task**: Abstract task with `setup(env)`, `validate_task()`, `filter_actions()`
- **ToolboxEnv**: Composes multiple Tools, routes actions to appropriate tool
- **STOP_ACTION**: Special action to signal task completion

### tool.py - Tool Abstraction
- **AbstractTool**: Base with `execute_action()`, `action_set()`, `reset()`, `close()`
- **Tool**: Protocol-based implementation using `action_space` attribute

### benchmark.py - Benchmark Interface
- **Benchmark**: Abstract with `setup()`, `close()`, `env_configs()`, optional `install()`/`uninstall()`

### llm.py - LLM Integration
- **Prompt**: Messages + tools for LLM call
- **LLMConfig**: Model config (name, temperature, max_tokens, retry strategy)
- **LLM**: Wrapper around LiteLLM completion API
- **LLMCall**: Logged LLM call with timestamp, config, prompt, and output

### episode.py - Episode Execution
- **Episode**: Manages agent-task execution, saves trajectory incrementally via Storage

### experiment.py - Experiment Management
- **ExpResult**: Results container with trajectories and failures
- **Experiment**: Holds agent_config, benchmark, creates episodes, prints stats

### exp_runner.py - Execution Runtimes
- **run_with_ray()**: Parallel execution using Ray workers
- **run_sequentially()**: Sequential execution with optional debug_limit

### storage.py - Trajectory Persistence
- **Storage**: Protocol for trajectory storage
- **FileStorage**: File-based storage
  - `trajectories/{id}.metadata.json` - trajectory metadata
  - `trajectories/{id}.jsonl` - steps as JSON lines
  - `llm_calls/{step_id}_{call_id}.json` - extracted LLM calls
- **LLMCallRef**: Reference to external LLM call file

### viewer.py - Experiment Viewer
- Gradio-based UI for exploring results
- Timeline visualization with step navigation
- Screenshot display, step details, raw JSON, LLM calls inspection
- Entry point: `al2-viewer` or `make viewer`

## Implementations

### agents/react.py - ReAct Agent
- **ReactAgentConfig**: Config with llm_config, system/react prompts, history limits
- **ReactAgent**: Implements ReAct framework
  - Maintains conversation history
  - Auto-compacts history when exceeding token limit via LLM summarization
  - Parses tool_calls from LLM output into Actions

### envs/browser.py - Browser Environment
- **BrowserEnvConfig**: Config with PWConfig for Playwright
- **BrowserEnv**: ToolboxEnv using single browser tool
  - Appends page_obs after each step
  - Provides `goto()` and `evaluate_js()` helpers

### tools/playwright.py - Playwright Tool
- **PWConfig**: Browser config (headless, use_html, use_screenshot, use_axtree, prune_html)
- **SyncPlaywrightTool**: Synchronous Playwright implementation
  - Implements BrowserActionSpace protocol
  - Actions: click, type, press_key, drag, hover, select_option, mouse_click_xy, wait, back, forward
  - Observations: page_html(), page_screenshot(), page_axtree()
- **AsyncPlaywrightTool**: Async version

### benchmarks/miniwob/ - MiniWob Benchmark
- **MiniWobBenchmark**: Manages local HTTP server for MiniWob HTML
  - Loads tasks from miniwob_tasks.json
  - Creates env_configs with MiniWobTask instances
- **MiniWobTask**: Individual MiniWob task
  - Sets up task via JS initialization
  - Validates via JS reward function
  - Filters actions to browser actions only

## Common Patterns

### Creating a New Agent
```python
from agentlab2.agent import Agent, AgentConfig
from agentlab2.core import AgentOutput, Observation

class MyAgentConfig(AgentConfig):
    # Add config fields
    def make(self) -> "MyAgent":
        return MyAgent(config=self)

class MyAgent(Agent):
    def step(self, obs: Observation) -> AgentOutput:
        # Process observation, call LLM, return actions
        return AgentOutput(actions=[...], llm_calls=[...])
```

### Creating a New Tool
```python
from agentlab2.tool import Tool, AbstractTool
from agentlab2.core import ActionSchema

class MyActionSpace(Protocol):
    def my_action(self, arg: str) -> None: ...

class MyTool(Tool):
    action_space = MyActionSpace

    def my_action(self, arg: str):
        # Implementation
        return "result"
```

### Creating a New Benchmark
```python
from agentlab2.benchmark import Benchmark
from agentlab2.environment import EnvironmentConfig

class MyBenchmark(Benchmark):
    env_config: EnvironmentConfig

    def setup(self): ...
    def close(self): ...
    def env_configs(self) -> list[EnvironmentConfig]:
        # Return list of env configs with tasks assigned
        return [self.env_config.model_copy(update=dict(_task=task)) for task in tasks]
```

### Running an Experiment
```python
from agentlab2.experiment import Experiment
from agentlab2.exp_runner import run_sequentially, run_with_ray

exp = Experiment(
    name="my_exp",
    output_dir="/path/to/output",
    agent_config=my_agent_config,
    benchmark=my_benchmark,
)

# Sequential (debugging)
run_sequentially(exp, debug_limit=5)

# Parallel (production)
run_with_ray(exp, n_cpus=8)
```

## Output Structure

```
output_dir/
├── experiment_config.json     # Full experiment configuration
├── trajectories/
│   ├── {task_id}_ep{n}.metadata.json  # Trajectory metadata
│   └── {task_id}_ep{n}.jsonl          # Steps as JSON lines
├── llm_calls/
│   └── {trajectory_id}_step{n}_{call_id}.json  # Individual LLM calls
└── ray_logs/                  # Ray worker logs (if using Ray)
```

## Development Commands

```bash
make install   # Install dependencies (uv sync + pip install -e .)
make hello     # Run MiniWob benchmark (full, parallel with Ray)
make debug     # Run 2 tasks sequentially for debugging
make viewer    # Launch experiment viewer with hot reload
make format    # Format code with Ruff
make lint      # Lint and auto-fix with Ruff
make test      # Run pytest tests
```

## Project Configuration

### Package Management
- **Tool**: `uv` (fast Python package manager)
- **Python**: >= 3.13 required
- **Virtual env**: `.venv/` in project root
- **Source layout**: `src/agentlab2/` (src-layout)

### Code Style (Ruff)
- **Line length**: 120 characters
- **Indent**: 4 spaces
- **Quotes**: Double quotes (`"`)

When writing code, follow these conventions:
```python
# Line length: 120 chars max
# Use double quotes for strings
# Imports are auto-sorted by ruff
from agentlab2.core import Action, ActionSchema, Observation  # sorted alphabetically
```

### Key Dependencies
| Package | Purpose |
|---------|---------|
| `pydantic` | Data validation and serialization for all config/data classes |
| `litellm` | Unified LLM API (OpenAI, Anthropic, Azure, etc.) |
| `playwright` | Browser automation for web agents |
| `ray` | Distributed parallel execution of episodes |
| `gradio` | Experiment viewer UI |
| `miniwob` | MiniWob benchmark HTML files |
| `beautifulsoup4` | HTML parsing and pruning |
| `Pillow` | Image handling for screenshots |

### Entry Points
- `al2-viewer` - CLI command to launch experiment viewer (defined in pyproject.toml)

### VSCode Setup
The project includes `.vscode/settings.json` with:
- Ruff as default Python formatter with format-on-save
- Python interpreter: `.venv/bin/python`
- Extra paths: `./src` for import resolution
- Type checking: standard mode (Pylance)

### Running Commands
Always use `uv run` to execute Python scripts:
```bash
uv run recipes/hello_miniwob.py        # Run a recipe
uv run pytest tests/ -v                 # Run tests
uv run al2-viewer                       # Launch viewer
uv run python -c "import agentlab2"    # Quick import test
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| LLM API keys | Set in `.env` or environment (e.g., `OPENAI_API_KEY`, `AZURE_API_KEY`) |

## Viewer CLI Options

```bash
uv run al2-viewer --help

# Options:
#   --results-dir PATH   Path to results directory (default: ~/agentlab_results/al2)
#   --debug              Enable hot reloading on source changes
#   --port PORT          Server port number (default: auto-select)
#   --share              Enable Gradio share link for remote access
```

## Testing

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_core.py -v

# Run with coverage (if configured)
uv run pytest tests/ --cov=agentlab2
```
