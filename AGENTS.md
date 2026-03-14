# cube-harness - Project Structure for Coding Agents

cube-harness is an open-source framework for building and evaluating UI agents. It serves as a universal evaluation platform for agentic benchmarks and as the foundation for RL data generation pipelines.

## Directory Structure

```
cube-harness/
├── src/cube_harness/           # Core framework source code
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
│   ├── storage.py           # Trajectory storage (Protocol + FileStorage)
│   ├── utils.py             # HTML pruning utilities
│   ├── viewer.py            # Gradio-based experiment/trajectory viewer
│   ├── agents/              # Agent implementations
│   │   ├── react.py         # ReAct agent with LLM-based reasoning
│   │   └── legacy_generic_agent.py  # XML-tag-based generic agent with prompt building
│   ├── tools/               # Tool implementations
│   │   ├── base.py          # BrowserTaskTool protocol
│   │   ├── playwright.py    # Sync/Async Playwright browser tools
│   │   ├── toolbox.py       # Composite tool for multiple tools
│   │   ├── browsergym.py    # BrowserGym integration (BidBrowserActionSpace)
│   │   └── computer.py      # Computer use tools (Docker-based)
│   ├── action_spaces/       # Action space protocols
│   │   └── browser_action_space.py  # BrowserActionSpace + BidBrowserActionSpace
│   ├── benchmarks/          # Benchmark implementations
│   │   ├── miniwob/         # MiniWob benchmark
│   │   │   ├── benchmark.py # MiniWobBenchmark class
│   │   │   ├── task.py      # MiniWobTask implementation
│   │   │   └── miniwob_tasks.json  # Task definitions
│   │   └── workarena/       # WorkArena ServiceNow benchmark
│   │       ├── benchmark.py # WorkArenaBenchmark class
│   │       └── task.py      # WorkArenaTask implementation
│   └── metrics/             # Telemetry and tracing
│       ├── models.py        # SpanRecord data model
│       ├── tracer.py        # Agent tracer (OpenTelemetry-based)
│       ├── processor.py     # Trace export and processing
│       ├── store.py         # JSONL span writer
│       └── disk_exporter.py # Disk-based span exporter
├── recipes/                 # Example experiment recipes
├── tests/                   # Test suite
└── docs/                    # Documentation and assets
```

## Data Flow

```
Benchmark → Episode(s) → Agent ↔ Environment → Trajectory
                            ↑        ↓
                           LLM   Tool(s)
```

## Key Classes

### base.py - Base Classes
- **TypedBaseModel**: Pydantic base that serializes/deserializes with `_type` field for polymorphism

### core.py - Data Structures
- **ActionSchema**: Function specification for LLM tool calls (name, description, parameters)
- **Action**: Represents a function call with id, name, and arguments
- **StepError**: Represents an error that occurred during a step execution
- **AgentOutput**: Contains actions list and llm_calls for logging
- **Content**: Piece of content (text, image, dict, BaseModel) in an observation, supports tool_call_id
- **Observation**: List of Contents, convertible to LLM messages
- **EnvironmentOutput**: Result of env step (obs, reward, done, info)
- **TrajectoryStep**: Single step pairing AgentOutput + EnvironmentOutput
- **Trajectory**: Full interaction history with steps and metadata
- **ActionSpace**: A frozenset of action callables for filtering
- **Task**: Abstract task with `setup(tool)`, `validate_task()`, `filter_actions()`, `obs_postprocess()`

### agent.py - Agent Protocol
- **AgentConfig**: Abstract base for agent configuration, has `make(action_set)` method
- **Agent**: Abstract base with `step(obs) -> AgentOutput` method

### environment.py - Environment Abstractions
- **EnvConfig**: Runtime config holding task and tool_config, has `make()` method
- **AbstractEnvironment**: Abstract base with `setup()`, `step(action)`, `close()`, `action_set`
- **Environment**: Concrete implementation that composes Task + Tool

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
- **Usage**: Token usage information from LLM response
- **LLMResponse**: Response from LLM containing message and usage info
- **LLM**: Wrapper around LiteLLM completion API
- **LLMCall**: Logged LLM call with timestamp, config, prompt, and output

### episode.py - Episode Execution
- **EpisodeConfig**: Configuration for an episode that can be saved and reloaded
- **Episode**: Manages agent-task execution, saves trajectory incrementally to JSONL files

### experiment.py - Experiment Management
- **ExpResult**: Results container with trajectories and failures
- **Experiment**: Holds agent_config, benchmark, creates episodes, prints stats

### exp_runner.py - Execution Runtimes
- **run_with_ray()**: Parallel execution using Ray workers
- **run_sequentially()**: Sequential execution with optional debug_limit

### storage.py - Trajectory Storage
- **Storage**: Protocol for trajectory storage backends
- **FileStorage**: File-based storage with JSONL step appending and separate LLM call extraction
- **LLMCallRef**: Reference to an LLM call stored in a separate file

### viewer.py - Experiment Viewer
- **ViewerState**: State for the Gradio viewer application
- **run_viewer()**: Launch Gradio UI for exploring trajectories and agent outputs

## Implementations

### agents/react.py - ReAct Agent
- **ReactAgentConfig**: Config with llm_config, system/react prompts, history limits
- **ReactAgent**: Implements ReAct framework
  - Maintains conversation history
  - Auto-compacts history when exceeding token limit via LLM summarization
  - Parses tool_calls from LLM output into Actions
  - Supports `get_training_pairs()` for extracting input/output pairs

### agents/legacy_generic_agent.py - Generic Agent
- **GenericAgentConfig**: Config with llm_config, prompt flags (GenericPromptFlags)
- **GenericAgent**: XML-tag-based agent with structured prompt building
  - Prompt elements: Think, Plan, Memory, Criticise, BeCautious, Hints
  - Configurable observation flags (use_html, use_axtree, use_screenshot)
  - Token-aware prompt fitting via observation shrinking (HTML/AXTree truncation)
  - Retry mechanism for LLM parsing errors
  - Screenshot support with multimodal messages
  - Extended thinking integration (reasoning_effort)
  - Draft-then-criticise pattern

### tools/base.py - Browser Tool Protocol
- **BrowserTaskTool**: Protocol for browser tools (reset, goto, evaluate_js, page_obs, page)

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

### tools/browsergym.py - BrowserGym Tool
- **BrowsergymConfig**: Config for BrowserGym environment
- **BrowsergymTool**: Full BrowserGym integration implementing BidBrowserActionSpace
  - Wraps BrowserGym's BrowserEnv for task setup and validation
  - Implements BrowserTaskTool protocol (goto, evaluate_js, page_obs)
  - Frame/iframe navigation for BID-based element access
  - Checkbox/radio fallback to JavaScript when needed
  - Observation conversion from BrowserGym to cube-harness format

### tools/computer.py - Computer Use Tool
- **Computer**: Docker-based computer interaction tool
  - Methods: mouse_click_xy, mouse_hover_xy, mouse_drag_xy, keyboard_type, run_cli_command, get_screenshot, get_current_window_axtree

### action_spaces/browser_action_space.py - Browser Actions
- **BrowserActionSpace**: Protocol defining browser actions using CSS selectors
  - browser_press_key, browser_type, browser_click, browser_drag, browser_hover
  - browser_select_option, browser_mouse_click_xy, browser_wait, browser_back, browser_forward, noop
- **BidBrowserActionSpace**: Protocol defining browser actions using Browser IDs (BIDs)
  - Same action set as BrowserActionSpace but uses BID-based element identification

### benchmarks/miniwob/ - MiniWob Benchmark
- **MiniWobBenchmark**: Manages local HTTP server for MiniWob HTML
  - Loads tasks from miniwob_tasks.json via `load_tasks()`
  - Uses `tool_config` to create PlaywrightConfig for env_configs
- **MiniWobTask**: Individual MiniWob task
  - Sets up task via JS initialization
  - Validates via JS reward function (`validate_per_step=True`)
  - Filters actions to browser actions via `supported_actions`
  - Post-processes screenshots to crop to MiniWob viewport (332x214)

### benchmarks/workarena/ - WorkArena Benchmark
- **WorkArenaBenchmark**: Benchmark for WorkArena ServiceNow tasks
  - Integrates with BrowserGym task classes
  - Configurable task level (l1, l2, l3)
- **WorkArenaTask**: Task wrapper for WorkArena BrowserGym tasks
  - Initializes BrowserGym environment with specific task class and seed
  - Validates via BrowserGym's reward function
  - Handles task teardown and resource cleanup

### metrics/ - Telemetry and Tracing
- **_AgentTracer**: OpenTelemetry-based tracer for benchmark, episode, and step spans
- **TraceProcessor**: Exports episode spans to structured JSON hierarchy
- **JsonlSpanWriter**: JSONL-based span storage
- **DiskSpanExporter**: Disk-based OpenTelemetry span exporter
- **get_tracer()**: Factory function for creating tracers (returns no-op if deps missing)

## Common Patterns

### Creating a New Agent
```python
from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import ActionSchema, AgentOutput, Observation

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
from cube_harness.tool import Tool, ToolConfig

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
from cube_harness.core import ActionSchema, Observation, Task
from cube_harness.tool import AbstractTool

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
from cube_harness.benchmark import Benchmark
from cube_harness.core import Task
from cube_harness.tool import ToolConfig

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
from cube_harness.experiment import Experiment
from cube_harness.exp_runner import run_sequentially, run_with_ray

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

## Development Commands
We're using make commands for most tasks. Look into `Makefile` for development commands.

## Project Configuration

### Package Management
- **Tool**: `uv` (fast Python package manager)
- **Python**: >= 3.13 required
- **Virtual env**: `.venv/` in project root
- **Source layout**: `src/cube_harness/` (src-layout)

### Code Style
- **Formatter**: Ruff
- **Line length**: 120 characters
- **Indent**: 4 spaces
- **Quotes**: Double quotes (`"`)

### Running Commands
Always use `uv run` to execute Python scripts:
```bash
uv sync --all-extras                   # install all optional dependencies
uv run recipes/hello_miniwob.py        # Run a recipe
uv run pytest tests/ -v                 # Run tests
uv run python -c "import cube_harness"    # Quick import test
```

## Environment Variables
All vars are set in `.env` file.

## Testing

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_core.py -v

# Run with coverage (if configured)
uv run pytest tests/ --cov=cube_harness
```

## Development Notes
- do not use imports inside the function or class, all imports should be at the top of the module!
- always add type hints for function parameters and return types, including for test functions.


## Constitution: Code Review Rules

All PRs are reviewed against the [cube-harness Constitution](/.claude/rules/constitution.md).

See `.claude/rules/` for the full constitution and detailed review rules with examples.