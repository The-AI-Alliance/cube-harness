# Task / Environment → cube.Task Migration Plan

## Context

This document covers step 2 of the cube → AgentLab2 integration (see `cube-integration-plan.md`).

**Goal**: Enable new benchmark authors to write cube-based tasks/benchmarks inside AL2, runnable with the existing `Episode`/`Experiment` machinery — **without breaking or rewriting MiniWob/WorkArena**.

**Out of scope**: Migrating MiniWob or WorkArena. Those benchmarks will eventually move to a `cubes/` repo. Until then they keep working as-is, but their code paths are clearly marked deprecated.

---

## Summary of Changes

| File | Change type | What | Status |
| --- | --- | --- | --- |
| `src/agentlab2/benchmark.py` | Deprecation | `__init_subclass__` warning; deprecation docstring on `load_tasks()`; runtime warning on `env_configs()`; removed `install()`/`uninstall()` | ✅ Done |
| `src/agentlab2/core.py` | Deprecation | `__init_subclass__` warning + docstring on `Task` | ✅ Done |
| `src/agentlab2/environment.py` | Deprecation | Constructor warnings + docstrings on `EnvConfig`, `Environment`, `AbstractEnvironment` | ✅ Done |
| `src/agentlab2/experiment.py` | Extend | Accept both AL2 `Benchmark` and `cube.benchmark.Benchmark`; dispatch to `get_task_configs()` or `env_configs()` with deprecation warning | TODO |
| `src/agentlab2/episode.py` | Extend | Accept `task_config: TaskConfig` (new) OR `env_config: EnvConfig` (deprecated); `EpisodeConfig` gains `task_config` field; new `_run_cube_task()` loop | TODO |
| `cubes/arithmetic/` | New | Toy POC benchmark using `cube.benchmark.Benchmark` directly | TODO |
| `tests/test_cube_episode.py` | New | Tests for the new cube task path through `Episode` | TODO |

**Untouched**: `benchmarks/miniwob/`, `benchmarks/workarena/`, all existing tests.

---

## Architectural Decisions

### A. Dual-path Episode (new cube path + deprecated legacy path)

`Episode.__init__` accepts *either* `task_config: TaskConfig` (new) or `env_config: EnvConfig` (deprecated). Exactly one must be provided. The public `run()` method dispatches:

```python
def run(self) -> Trajectory:
    if self.config.task_config is not None:
        return self._run_cube_task()   # new cube path
    else:
        warnings.warn("Running via env_config is deprecated...", DeprecationWarning)
        return self._run_legacy()      # existing logic, unchanged
```

MiniWob and WorkArena continue to use `env_config=` unchanged. New benchmarks use `task_config=`.

### B. Experiment accepts both benchmark types; dispatches accordingly

New benchmarks subclass `cube.benchmark.Benchmark` directly (has `get_task_configs()`). Legacy benchmarks (MiniWob, WorkArena) subclass AL2's deprecated `agentlab2.benchmark.Benchmark` (has `env_configs()`). `Experiment` accepts either and dispatches:

```text
cube.benchmark.Benchmark  →  get_task_configs()  →  Episode(task_config=...)    [new path]
agentlab2.benchmark.Benchmark  →  env_configs()  →  Episode(env_config=...)     [deprecated, warns]
```

`Experiment.benchmark` is typed as `agentlab2.benchmark.Benchmark | cube.benchmark.Benchmark`. A deprecation warning fires at episode-creation time when the benchmark is an AL2 instance. The dispatch catches both `AttributeError` (AL2's `Benchmark` has no `get_task_configs()`) and `NotImplementedError` as the signal to fall back.

### C. EpisodeConfig gains task_config field

`EpisodeConfig` gets a new optional `task_config: TaskConfig | None = None` field. The old `task_id: str` and `tool_config: ToolConfig` fields become `| None` (backward-compatible on-disk format). Existing stored configs deserialize fine because Pydantic ignores the new None-defaulted field.

`TypedBaseModel` already serializes with `_type` (full dotted class name) and reconstructs the correct subclass on load — polymorphic `TaskConfig` in `EpisodeConfig` is round-trip safe with no extra work.

---

## Detailed File Changes

### 1. `src/agentlab2/episode.py`

#### `EpisodeConfig` changes

```python
# Add field:
task_config: TaskConfig | None = None

# Make optional (backward compat):
task_id: str | None = None
tool_config: ToolConfig | None = None
```

Full new `EpisodeConfig`:

```python
class EpisodeConfig(TypedBaseModel):
    id: int
    agent_config: AgentConfig
    exp_name: str
    output_dir: Path
    max_steps: int
    # New cube path:
    task_config: TaskConfig | None = None
    # Deprecated legacy path:
    task_id: str | None = None
    tool_config: ToolConfig | None = None
```

#### `Episode.__init__` changes

```python
def __init__(
    self,
    id: int,
    output_dir: Path,
    agent_config: AgentConfig,
    task_config: TaskConfig | None = None,   # new
    env_config: EnvConfig | None = None,     # deprecated
    exp_name: str = "default",
    max_steps: int = MAX_STEPS,
    storage: Storage | None = None,
) -> None:
    if task_config is None and env_config is None:
        raise ValueError("Provide either task_config (new) or env_config (deprecated).")
    if task_config is not None and env_config is not None:
        raise ValueError("Provide only one of task_config or env_config.")

    if env_config is not None:
        warnings.warn(
            "env_config is deprecated. Pass task_config with a cube.Task instead.",
            DeprecationWarning, stacklevel=2,
        )

    self.config = EpisodeConfig(
        id=id,
        agent_config=agent_config,
        exp_name=exp_name,
        output_dir=output_dir,
        max_steps=max_steps,
        task_config=task_config,
        task_id=env_config.task.id if env_config is not None else None,
        tool_config=env_config.tool_config if env_config is not None else None,
    )
    self._env_config = env_config   # kept for _run_legacy()
    self.storage = storage or FileStorage(output_dir)
    self.allow_overwrite = False
```

#### `Episode.run()` dispatch

```python
def run(self) -> Trajectory:
    if self.config.task_config is not None:
        return self._run_cube_task()
    else:
        warnings.warn(
            "Running via env_config is deprecated. Use task_config with a cube.Task.",
            DeprecationWarning, stacklevel=2,
        )
        return self._run_legacy()
```

#### `Episode._run_cube_task()` (new method)

Contains the new clean run loop (mirrors `_run_legacy` but calls `task.reset()` / `task.step()` / `task.close()`):

```python
def _run_cube_task(self) -> Trajectory:
    """Run loop for cube.Task-based episodes."""
    tracer = get_tracer(self.config.exp_name)
    task = self.config.task_config.make()          # lazy: creates tool here
    agent = self.config.agent_config.make(task.action_set)
    try:
        with tracer.episode(task.id, experiment=self.config.exp_name) as episode_span:
            start_time = time.time()
            obs, info = task.reset()               # was: env.setup()
            env_output = EnvironmentOutput(obs=obs, info=info)
            trajectory = Trajectory(
                id=f"{task.id}_ep{self.config.id}",
                steps=[TrajectoryStep(output=env_output, start_time=start_time, end_time=time.time())],
                metadata={"task_id": task.id, **info},
                start_time=start_time,
            )
            self.storage.save_trajectory(trajectory, allow_overwrite=self.allow_overwrite)
            turns = 0
            while not env_output.done and turns < self.config.max_steps:
                with tracer.step(f"turn_{turns}") as span:
                    ts = time.time()
                    try:
                        agent_output = agent.step(env_output.obs)
                    except Exception as e:
                        # ... same error handling as existing run() ...
                        raise
                    env_ts = time.time()
                    try:
                        env_output = task.step(agent_output.actions)  # was: env.step()
                    except Exception as e:
                        # ... same error handling ...
                        raise
                    # ... same trajectory recording ...
                    turns += 1
            trajectory.end_time = time.time()
            trajectory.reward_info = {"reward": env_output.reward, "done": env_output.done, **env_output.info}
            self.storage.save_trajectory(trajectory)
            # ... same status recording ...
    except Exception as e:
        raise
    finally:
        task.close()                               # was: env.close()
    return trajectory
```

#### `Episode._run_legacy()` (renamed from existing `run()`)

Move the **entire existing `run()` body** verbatim into `_run_legacy()`. No other changes.

#### `Episode.load_episode_from_config()` changes

Support both paths:

```python
@classmethod
def load_episode_from_config(cls, config_path: Path, benchmark=None) -> Self:
    output_dir = config_path.parent.parent
    storage = FileStorage(output_dir)
    episode_config = storage.load_episode_config(config_path)

    if episode_config.task_config is not None:
        # New cube path — fully self-contained, no benchmark needed
        return cls(
            id=episode_config.id,
            output_dir=episode_config.output_dir,
            agent_config=episode_config.agent_config,
            task_config=episode_config.task_config,
            exp_name=episode_config.exp_name,
            max_steps=episode_config.max_steps,
            storage=storage,
        )
    else:
        # Legacy path — needs benchmark to reconstruct Task
        if benchmark is None:
            raise ValueError(
                "benchmark is required when loading a legacy EpisodeConfig "
                "(no task_config stored). Pass the benchmark instance."
            )
        # ... existing reconstruction logic unchanged ...
```

---

### 2. `src/agentlab2/experiment.py`

#### `Experiment._create_all_episodes()` changes

```python
def _create_all_episodes(self) -> list[Episode]:
    # Try new cube path first
    try:
        task_configs = self.benchmark.get_task_configs()
        episodes = [
            Episode(
                id=i,
                output_dir=self.output_dir,
                agent_config=self.agent_config,
                task_config=tc,               # new
                exp_name=self.name,
                max_steps=self.max_steps,
            )
            for i, tc in enumerate(task_configs)
        ]
    except (AttributeError, NotImplementedError):
        # Fall back to deprecated env_configs() path for legacy benchmarks.
        # AttributeError: AL2's Benchmark has no get_task_configs() at all.
        # NotImplementedError: cube.benchmark.Benchmark subclass that doesn't override it.
        warnings.warn(
            f"{type(self.benchmark).__name__} does not implement get_task_configs(). "
            "Falling back to deprecated env_configs(). "
            "Implement get_task_configs() to use the cube.Task path.",
            DeprecationWarning, stacklevel=2,
        )
        episodes = [
            Episode(
                id=i,
                output_dir=self.output_dir,
                agent_config=self.agent_config,
                env_config=ec,                # deprecated
                exp_name=self.name,
                max_steps=self.max_steps,
            )
            for i, ec in enumerate(self.benchmark.env_configs())
        ]
    for episode in episodes:
        episode.storage.save_episode_config(episode.config)
    return episodes
```

**Import to add**: `import warnings`

---

### 3. `cubes/arithmetic/` — new POC

A self-contained arithmetic benchmark demonstrating the full cube pattern with no external dependencies.

#### Directory structure

```text
cubes/
└── arithmetic/
    ├── __init__.py
    ├── tool.py          # ArithmeticTool with @tool_action submit_answer()
    ├── task.py          # ArithmeticTask(cube.Task) + ArithmeticTaskConfig
    └── benchmark.py     # ArithmeticBenchmark(Benchmark)
```

#### `cubes/arithmetic/tool.py`

```python
from cube.tool import ToolConfig
from cube.core import Observation
from agentlab2.tool import ToolWithTelemetry
from cube.tool import tool_action


class ArithmeticToolConfig(ToolConfig):
    def make(self, container=None) -> "ArithmeticTool":
        return ArithmeticTool()


class ArithmeticTool(ToolWithTelemetry):
    """Simple tool that accepts a numeric answer submission."""

    def __init__(self):
        self._submitted: int | None = None

    @tool_action
    def submit_answer(self, answer: int) -> str:
        """Submit your answer to the arithmetic question.

        Args:
            answer: Your integer answer.

        Returns:
            Confirmation message.
        """
        self._submitted = answer
        return f"Answer {answer} submitted."

    def reset(self) -> None:
        self._submitted = None

    @property
    def last_answer(self) -> int | None:
        return self._submitted
```

#### `cubes/arithmetic/task.py`

```python
import operator
from cube.task import Task, TaskConfig, TaskMetadata
from cube.core import Observation, EnvironmentOutput
from cube.tool import ToolConfig
from agentlab2.benchmarks.arithmetic.tool import ArithmeticTool, ArithmeticToolConfig


OPERATIONS = {"+": operator.add, "-": operator.sub, "*": operator.mul}


class ArithmeticTask(Task):
    """Task: solve a simple arithmetic problem by calling submit_answer()."""

    a: int
    b: int
    op: str  # one of "+", "-", "*"

    @property
    def tool(self) -> ArithmeticTool:  # type: ignore[override]
        return self._tool  # type: ignore[return-value]
        # TODO: clarify the BrowserTaskTool Protocol / cube AbstractTool relationship

    @property
    def _expected(self) -> int:
        return OPERATIONS[self.op](self.a, self.b)

    def reset(self) -> tuple[Observation, dict]:
        self.tool.reset()
        question = f"What is {self.a} {self.op} {self.b}?"
        return Observation.from_text(question), {"question": question, "expected": self._expected}

    def evaluate(self, obs: Observation) -> tuple[float, dict]:
        answer = self.tool.last_answer
        correct = answer == self._expected
        return (1.0 if correct else 0.0), {"answer": answer, "expected": self._expected, "correct": correct}

    def finished(self, obs: Observation) -> bool:
        return self.tool.last_answer is not None


class ArithmeticTaskConfig(TaskConfig):
    a: int
    b: int
    op: str = "+"

    def make(self, runtime_context=None, container_backend=None) -> ArithmeticTask:
        return ArithmeticTask(
            metadata=TaskMetadata(
                id=self.task_id,
                abstract_description=f"Solve: {self.a} {self.op} {self.b}",
            ),
            tool_config=self.tool_config or ArithmeticToolConfig(),
            a=self.a,
            b=self.b,
            op=self.op,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
```

#### `cubes/arithmetic/benchmark.py`

New benchmarks inherit from `cube.benchmark.Benchmark` directly, not from AL2's deprecated `Benchmark`.

```python
from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.task import TaskMetadata
from cubes.arithmetic.task import ArithmeticTaskConfig
from cubes.arithmetic.tool import ArithmeticToolConfig


class ArithmeticBenchmark(Benchmark):
    """Toy benchmark: simple arithmetic problems. No setup/teardown needed."""

    benchmark_metadata = BenchmarkMetadata(name="arithmetic", version="1.0", description="Simple arithmetic tasks")
    task_metadata = {
        "add_3_4":   TaskMetadata(id="add_3_4"),
        "sub_10_3":  TaskMetadata(id="sub_10_3"),
        "mul_6_7":   TaskMetadata(id="mul_6_7"),
        "add_100_1": TaskMetadata(id="add_100_1"),
    }
    task_config_class = ArithmeticTaskConfig

    def _setup(self) -> None:
        pass  # no shared infrastructure needed

    def close(self) -> None:
        pass
```

#### Usage example (smoke test / demo)

```python
from agentlab2.benchmarks.arithmetic.benchmark import ArithmeticBenchmark
from agentlab2.benchmarks.arithmetic.task import ArithmeticTaskConfig

benchmark = ArithmeticBenchmark()
configs = benchmark.get_task_configs()

config = configs[0]                  # ArithmeticTaskConfig(task_id="add_3_4", ...)
task = config.make()                 # ArithmeticTask(a=3, b=4, op="+") — tool created here
obs, info = task.reset()             # "What is 3 + 4?"
print(obs)

from cube.core import Action
action = Action(name="submit_answer", arguments={"answer": 7})
env_out = task.step(action)          # evaluate() called, reward=1.0
print(env_out.reward)                # 1.0
task.close()
```

---

### 4. `tests/test_cube_episode.py` — new

Tests that the new cube path through `Episode` and `Experiment` works end-to-end.

```python
"""Tests for the cube.Task path through Episode and Experiment."""
import pytest
from agentlab2.benchmarks.arithmetic.benchmark import ArithmeticBenchmark
from agentlab2.benchmarks.arithmetic.task import ArithmeticTaskConfig
from agentlab2.episode import Episode
from agentlab2.experiment import Experiment
from tests.conftest import MockAgentConfig   # reuse existing mock agent

def test_episode_accepts_task_config(tmp_dir, mock_agent_config):
    config = ArithmeticTaskConfig(task_id="add_3_4", a=3, b=4, op="+")
    episode = Episode(id=0, output_dir=tmp_dir, agent_config=mock_agent_config, task_config=config)
    assert episode.config.task_config == config
    assert episode.config.task_id is None     # legacy field not set

def test_episode_run_cube_task(tmp_dir, mock_agent_config):
    config = ArithmeticTaskConfig(task_id="add_3_4", a=3, b=4, op="+")
    episode = Episode(id=0, output_dir=tmp_dir, agent_config=mock_agent_config, task_config=config)
    trajectory = episode.run()
    assert trajectory is not None

def test_experiment_with_arithmetic_benchmark(tmp_dir, mock_agent_config):
    benchmark = ArithmeticBenchmark()
    exp = Experiment(name="test", output_dir=tmp_dir, agent_config=mock_agent_config, benchmark=benchmark)
    episodes = exp.get_episodes_to_run()
    assert len(episodes) == 4

def test_episode_config_round_trip(tmp_dir, mock_agent_config):
    """EpisodeConfig with task_config serializes and deserializes correctly."""
    config = ArithmeticTaskConfig(task_id="add_3_4", a=3, b=4, op="+")
    episode = Episode(id=0, output_dir=tmp_dir, agent_config=mock_agent_config, task_config=config)
    episode.storage.save_episode_config(episode.config)
    # Load back — no benchmark needed since task_config is self-contained
    restored = Episode.load_episode_from_config(
        tmp_dir / "episode_configs" / "episode_0_task_add_3_4.json"
    )
    assert restored.config.task_config.task_id == "add_3_4"
```

---

## What is kept only for MiniWob/WorkArena (deprecated)

| Symbol | File | Reason kept | Deprecation |
| --- | --- | --- | --- |
| `Task` (AL2) | `core.py` | MiniWob + WorkArena Task base class | `__init_subclass__` warning |
| `ActionSpace` | `core.py` | MiniWob + WorkArena action filtering | docstring note |
| `Environment` | `environment.py` | Episode legacy run path | constructor warning |
| `AbstractEnvironment` | `environment.py` | MiniWob/WorkArena don't use it directly | constructor warning |
| `EnvConfig` | `environment.py` | Episode legacy run path | constructor warning |
| `Benchmark.load_tasks()` | `benchmark.py` | Implemented by MiniWob + WorkArena | no warning (overridden) |
| `Benchmark.env_configs()` | `benchmark.py` | Called by Experiment fallback path | runtime warning |
| `EpisodeConfig.task_id` | `episode.py` | Old stored configs on disk | — |
| `EpisodeConfig.tool_config` | `episode.py` | Old stored configs on disk | — |
| `Episode._run_legacy()` | `episode.py` | Invoked when `env_config` is passed | run() warning |

---

## Verification

```bash
# 1. cube-standard must still pass (no changes there)
cd cube-standard && make test

# 2. AL2 tests must still pass (old benchmarks unaffected)
cd AgentLab2 && make test

# 3. Smoke test: new cube path
python -c "
from cubes.arithmetic.benchmark import ArithmeticBenchmark
b = ArithmeticBenchmark()
tc = b.get_task_configs()[0]
task = tc.make()
obs, _ = task.reset()
print(obs)
from cube.core import Action
out = task.step(Action(name='submit_answer', arguments={'answer': 7}))
print(out.reward)  # 1.0
task.close()
"
```

---

## Status

| Step | File(s) | Status |
| --- | --- | --- |
| Deprecation warning on AL2 `Task` | `core.py` | ✅ Done — `__init_subclass__` warning + docstring |
| Deprecation warnings on `Environment` classes | `environment.py` | ✅ Done — constructor warnings on `EnvConfig`, `AbstractEnvironment`, `Environment` + docstrings |
| Deprecations on AL2 `Benchmark` | `benchmark.py` | ✅ Done — `__init_subclass__` warning + docstring; runtime warning on `env_configs()`; deprecation docstring on `load_tasks()`; `install()`/`uninstall()` removed (no cube equivalent, only used in tests) |
| Dual-path `Episode` + `EpisodeConfig` | `episode.py` | TODO |
| Dispatch in `Experiment` | `experiment.py` | TODO |
| Arithmetic toy cube (tool, task, config, benchmark) | `cubes/arithmetic/` | TODO |
| New tests for cube path | `tests/test_cube_episode.py` | TODO |
| Update `cube-integration-plan.md` status | `cube-integration-plan.md` | TODO |

### Deviations from original plan

- **`benchmark.py`**: Did not add `get_task_configs()` to AL2's `Benchmark`. New benchmarks use `cube.benchmark.Benchmark` directly — there is no reason to forward the method through the deprecated AL2 class.
- **`benchmark.py`**: Did not make `tool_config` optional. Only legacy benchmarks (MiniWob/WorkArena) use this class and they always supply it.
- **`benchmark.py`**: Removed `install()`/`uninstall()` entirely (no cube equivalent, only referenced in tests). Updated `conftest.py` `MockBenchmark` and `tests/test_benchmark.py` accordingly.

### Note on test deprecation warnings

`MockTask` (inherits `agentlab2.core.Task`) and `MockBenchmark` (inherits `agentlab2.benchmark.Benchmark`) in `conftest.py` now emit `DeprecationWarning` at class-definition time. These warnings are intentionally left in place — they serve as a reminder that the legacy test infrastructure itself needs to be migrated once MiniWob and WorkArena move to the cube API.
