# AgentLab2 - Project Structure for Coding Agents

AgentLab2 is an open-source framework for building and evaluating UI agents. It serves as a universal evaluation platform for agentic benchmarks and as the foundation for RL data generation pipelines.

## Directory Structure

```
AgentLab2/
├── src/agentlab2/           # Core framework source code
│   ├── __init__.py          # Package metadata and version
│   ├── base.py              # TypedBaseModel for serialization with type info
│   ├── core.py              # Data structures: Action, Observation, Trajectory, Task
│   ├── agent.py             # Agent protocol and configuration
│   ├── environment.py       # Environment and EnvConfig abstractions
│   ├── tool.py              # Tool abstraction for action spaces
│   ├── benchmark.py         # Benchmark interface for task collections
│   ├── llm.py               # LLM wrapper using LiteLLM
│   ├── episode.py           # Episode execution and trajectory persistence
│   ├── experiment.py        # Experiment configuration and statistics
│   ├── exp_runner.py        # Sequential and Ray-based parallel execution
│   ├── utils.py             # HTML pruning utilities
│   ├── agents/              # Agent implementations
│   │   └── react.py         # ReAct agent with LLM-based reasoning
│   ├── tools/               # Tool implementations
│   │   ├── __init__.py      # Tools package
│   │   ├── playwright.py    # Sync/Async Playwright browser tools
│   │   ├── toolbox.py       # Composite tool for multiple tools
│   │   ├── browsergym.py    # BrowserGym integration (stub)
│   │   └── computer.py      # Computer use tools (stub)
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
base.py (TypedBaseModel for serialization)
    ↑
llm.py (LLM wrapper, uses TypedBaseModel)
    ↑
core.py (data structures: Action, Observation, Trajectory, Task)
    ↑
tool.py (Tool, ToolConfig, AbstractTool)
    ↑
environment.py (Environment, EnvConfig - composes task + tool)
    ↑
agent.py (Agent protocol - receives observations, produces actions)
    ↑
benchmark.py (task collections with tool_config)
    ↑
episode.py (runs single agent-task execution)
    ↑
experiment.py (coordinates multiple episodes)
    ↑
exp_runner.py (sequential or Ray parallel execution)
```

## Key Classes

### base.py - Base Classes
- **TypedBaseModel**: Pydantic base that serializes/deserializes with `_type` field for polymorphism

### core.py - Data Structures
- **ActionSchema**: Function specification for LLM tool calls (name, description, parameters)
- **Action**: Represents a function call with id, name, and arguments
- **AgentOutput**: Contains actions list and llm_calls for logging
- **Content**: Piece of content (text, image, dict, BaseModel) in an observation, supports tool_call_id
- **Observation**: List of Contents, convertible to LLM messages
- **EnvironmentOutput**: Result of env step (obs, reward, done, info)
- **Trajectory**: Full interaction history with steps and metadata
- **ActionSpace**: Protocol base class for action spaces
- **ActionSubset**: Type alias for tuple of action methods (for filtering)
- **Task**: Abstract task with `setup(tool)`, `validate_task()`, `filter_actions()`, `obs_postprocess()`

### agent.py - Agent Protocol
- **AgentConfig**: Abstract base for agent configuration, has `make(action_set)` method
- **Agent**: Abstract base with `step(obs) -> AgentOutput` method

### environment.py - Environment Abstractions
- **EnvConfig**: Runtime config holding task and tool_config, has `make()` method
- **AbstractEnvironment**: Abstract base with `setup()`, `step(action)`, `close()`, `action_set`
- **Environment**: Concrete implementation that composes Task + Tool
- **STOP_ACTION**: Special action to signal task completion

### tool.py - Tool Abstraction
- **AbstractTool**: Abstract base with `execute_action()`, `action_set`, `reset()`, `close()`
- **ToolConfig**: Abstract base for tool configurations with `make()` method
- **Tool**: Protocol-based implementation using `action_space` attribute

### benchmark.py - Benchmark Interface
- **Benchmark**: Abstract with `setup()`, `close()`, `load_tasks()`, `env_configs()`, optional `install()`/`uninstall()`
  - Contains `tool_config` field for creating tools

### llm.py - LLM Integration
- **Prompt**: Messages + tools for LLM call
- **LLMConfig**: Model config (name, temperature, max_tokens, reasoning_effort, retry strategy)
- **LLM**: Wrapper around LiteLLM completion API
- **LLMCall**: Logged LLM call with timestamp, config, prompt, and output

### episode.py - Episode Execution
- **Episode**: Manages agent-task execution, saves trajectory incrementally to JSONL files

### experiment.py - Experiment Management
- **ExpResult**: Results container with trajectories and failures
- **Experiment**: Holds agent_config, benchmark, creates episodes, prints stats

### exp_runner.py - Execution Runtimes
- **run_with_ray()**: Parallel execution using Ray workers
- **run_sequentially()**: Sequential execution with optional debug_limit

## Implementations

### agents/react.py - ReAct Agent
- **ReactAgentConfig**: Config with llm_config, system/react prompts, history limits
- **ReactAgent**: Implements ReAct framework
  - Maintains conversation history
  - Auto-compacts history when exceeding token limit via LLM summarization
  - Parses tool_calls from LLM output into Actions
  - Supports `get_training_pairs()` for extracting input/output pairs

### tools/playwright.py - Playwright Tool
- **PlaywrightConfig**: Browser config (headless, use_html, use_screenshot, use_axtree, prune_html)
- **SyncPlaywrightTool**: Synchronous Playwright implementation
  - Implements BrowserActionSpace protocol
  - Actions: browser_click, browser_type, browser_press_key, browser_drag, browser_hover, browser_select_option, browser_mouse_click_xy, browser_wait, browser_back, browser_forward, noop
  - Observations: page_html(), page_screenshot(), page_axtree()
  - Helpers: goto(), evaluate_js()
- **AsyncPlaywrightTool**: Async version with same interface

### tools/toolbox.py - Composite Tool
- **ToolboxConfig**: Config holding list of tool_configs
- **Toolbox**: Composite tool that combines multiple tools
  - Routes actions to appropriate tool by action name
  - Provides `find_tool(cls)` helper to retrieve specific tool

### tools/browsergym.py - BrowserGym Tool (stub)
- **BrowsergymTool**: Placeholder for BrowserGym integration

### tools/computer.py - Computer Use Tool (stub)
- **Computer**: Docker-based computer interaction tool
  - Methods: mouse_click_xy, mouse_hover_xy, mouse_drag_xy, keyboard_type, run_cli_command, get_screenshot, get_current_window_axtree

### action_spaces/browser_action_space.py - Browser Actions
- **BrowserActionSpace**: Protocol defining browser actions
  - browser_press_key, browser_type, browser_click, browser_drag, browser_hover
  - browser_select_option, browser_mouse_click_xy, browser_wait, browser_back, browser_forward, noop

### benchmarks/miniwob/ - MiniWob Benchmark
- **MiniWobBenchmark**: Manages local HTTP server for MiniWob HTML
  - Loads tasks from miniwob_tasks.json via `load_tasks()`
  - Uses `tool_config` to create PlaywrightConfig for env_configs
- **MiniWobTask**: Individual MiniWob task
  - Sets up task via JS initialization
  - Validates via JS reward function (`validate_per_step=True`)
  - Filters actions to browser actions via `supported_actions`
  - Post-processes screenshots to crop to MiniWob viewport (332x214)

## Common Patterns

### Creating a New Agent
```python
from agentlab2.agent import Agent, AgentConfig
from agentlab2.core import ActionSchema, AgentOutput, Observation

class MyAgentConfig(AgentConfig):
    # Add config fields
    def make(self, action_set: list[ActionSchema]) -> "MyAgent":
        return MyAgent(config=self, action_set=action_set)

class MyAgent(Agent):
    def __init__(self, config: MyAgentConfig, action_set: list[ActionSchema]):
        self.config = config
        self.tools = [a.as_dict() for a in action_set]

    def step(self, obs: Observation) -> AgentOutput:
        # Process observation, call LLM, return actions
        return AgentOutput(actions=[...], llm_calls=[...])
```

### Creating a New Tool
```python
from typing import Protocol
from agentlab2.tool import Tool, ToolConfig

class MyActionSpace(Protocol):
    def my_action(self, arg: str) -> None: ...

class MyToolConfig(ToolConfig):
    # Add config fields
    def make(self) -> "MyTool":
        return MyTool(config=self)

class MyTool(Tool):
    action_space = MyActionSpace

    def __init__(self, config: MyToolConfig):
        self.config = config

    def my_action(self, arg: str):
        # Implementation
        return "result"
```

### Creating a New Task
```python
from agentlab2.core import ActionSchema, Observation, Task
from agentlab2.tool import AbstractTool

class MyTask(Task):
    id: str
    validate_per_step: bool = False

    def __init__(self, id: str, ...):
        self.id = id

    def setup(self, tool: AbstractTool) -> tuple[Observation, dict]:
        self._tool = tool
        # Initialize task state
        return Observation.from_text("Task goal"), {"info": "..."}

    def validate_task(self, obs: Observation) -> tuple[float, dict]:
        # Check if task is completed
        return reward, {"done": done}

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        # Return subset of actions allowed for this task
        return actions
```

### Creating a New Benchmark
```python
from agentlab2.benchmark import Benchmark
from agentlab2.core import Task
from agentlab2.tool import ToolConfig

class MyBenchmark(Benchmark):
    tool_config: ToolConfig  # Required field

    def setup(self):
        # Start any required services (servers, containers, etc.)
        pass

    def close(self):
        # Clean up services
        pass

    def load_tasks(self) -> list[Task]:
        # Load and return task instances
        return [MyTask(id="task1"), MyTask(id="task2")]
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
│   ├── run{id}_task_{task_id}.metadata.json  # Trajectory metadata
│   └── run{id}_task_{task_id}.jsonl          # Steps as JSON lines (AgentOutput, EnvironmentOutput)
└── ray_logs/                  # Ray worker logs (if using Ray)
```

## Development Commands

```bash
make install   # Install dependencies (uv sync + pip install -e .)
make hello     # Run MiniWob benchmark (full, parallel with Ray)
make debug     # Run 2 tasks sequentially for debugging
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
| `miniwob` | MiniWob benchmark HTML files |
| `beautifulsoup4` | HTML parsing and pruning |
| `Pillow` | Image handling for screenshots |
| `termcolor` | Colored terminal output for logging |

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
uv run python -c "import agentlab2"    # Quick import test
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| LLM API keys | Set in `.env` or environment (e.g., `OPENAI_API_KEY`, `AZURE_API_KEY`) |

## Testing

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_core.py -v

# Run with coverage (if configured)
uv run pytest tests/ --cov=agentlab2
```


## Development Notes
- do not use imports inside the function or class, all imports should be at the top of the module!
- always add type hints for function parameters and return types, including for test functions.