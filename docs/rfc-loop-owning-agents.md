# RFC: Loop-Owning Agents ("run" interface)

**Branch**: `feat/meta-agent`  
**Status**: Draft — for discussion  
**Date**: 2026-04-14

---

## 1. Motivation

The current `Agent` contract has one method: `step(obs) -> AgentOutput`. The harness drives the loop.

This works well for classic ReAct/Genny-style agents. But it is a poor fit for:

- **Agentic frameworks** (LangGraph, smolagents, AutoGen, CrewAI) where the agent *is* the loop — it decides when to call tools, when to stop, and how to manage its own state.
- **Claude SDK `run` calls**, where a single SDK call expands into many internal tool invocations.
- **Meta-agent architectures** where an outer LLM plans and dispatches inner agents — the outer agent needs to drive sub-loops.

The goal is to support a `run(task_as_tool)` interface alongside the existing `step` interface, with:

1. **Full trajectory recording** — same `TrajectoryStep` / `FileStorage` pipeline as today.
2. **Eval** — same reward/done signals from `task.step()` and `task.close()`.
3. **Backward compatibility** — existing `step`-based agents and recipes continue to work unchanged.
4. **No code duplication** — shared infrastructure between both modes.

---

## 2. Current Architecture (reference)

```
Episode._run_loop()
  ├── setup_fn()               → EnvironmentOutput (initial obs)
  ├── loop:
  │     agent.step(obs)        → AgentOutput       [harness calls agent]
  │     storage.save_step(...)
  │     task.step(actions)     → EnvironmentOutput [harness calls env]
  │     storage.save_step(...)
  └── close_fn()
```

Key invariants the new design must preserve:

- Each `(AgentOutput, EnvironmentOutput)` pair is saved incrementally as `NNN_act.msgpack.zst` / `NNN_obs.msgpack.zst`.
- `Trajectory.reward_info` reflects the final `EnvironmentOutput.reward` / `EnvironmentOutput.done`.
- OpenTelemetry spans wrap each turn.
- `SummaryProcessor` receives every step.

---

## 3. Proposed Design

### 3.1 New agent interface

`run()` becomes the primary method on `Agent`. Its default implementation calls `step()` in the same loop the harness runs today, so **existing agents require no changes**. Agents that want to own the loop simply override `run()` instead.

```python
class Agent(ABC):
    # Still the atomic decision unit — override this for step-mode agents
    def step(self, obs: Observation) -> AgentOutput:
        raise NotImplementedError

    # Primary entry point — harness always calls this
    # Default: drives the step() loop, mirroring the harness loop today
    def run(self, task_tool: "TaskTool") -> AgentOutput:
        all_llm_calls: list[LLMCall] = []
        while not task_tool.done:
            agent_output = self.step(task_tool.last_obs)
            all_llm_calls.extend(agent_output.llm_calls)
            if not agent_output.actions:
                break
            task_tool.call(agent_output)
        return AgentOutput(llm_calls=all_llm_calls)
```

`AgentConfig.make()` is unchanged. No detection, no `owns_loop()` — the harness always calls `run()`.

### 3.2 TaskTool — the cube task as a tool

The agent receives one object: a `TaskTool`. It is a monitored wrapper around the cube `Task` that:

- Exposes the task's actions as callable methods (same `ActionSchema` the harness already computes).
- Intercepts every call and records `(AgentOutput, EnvironmentOutput)` pairs to `Storage`.
- Propagates `done` / `reward` so the agent can detect task completion.

```python
class TaskTool:
    """Wraps a cube Task, monitoring every action for trajectory recording.

    Passed to Agent.run(). Each call() records an (AgentOutput, EnvironmentOutput)
    pair using the same FileStorage pipeline as the harness step loop.
    """

    @property
    def action_schemas(self) -> list[ActionSchema]: ...

    @property
    def last_obs(self) -> Observation: ...

    @property
    def done(self) -> bool: ...

    @property
    def reward(self) -> float: ...

    def call(self, agent_output: AgentOutput) -> EnvironmentOutput:
        """Execute the actions in agent_output, record both sides, return env output."""
        ...
```

The agent passes a fully-formed `AgentOutput` (actions + llm_calls + thoughts — same as today). `TaskTool.call()` then:

1. Saves the `AgentOutput` as an `act` step.
2. Calls `task.step(agent_output.actions)`.
3. Saves the resulting `EnvironmentOutput` as an `obs` step.
4. Updates internal `done` / `reward`.

The default `Agent.run()` implementation passes each `step()` result directly to `task_tool.call()`, so the on-disk format is identical for both modes.

This means the on-disk format is **identical** to the step-mode format. The viewer and analysis code require zero changes.

### 3.3 Episode dispatching — simplified

`Episode._run_loop` collapses to a single path: create a `TaskTool`, call `agent.run(task_tool)`, finalize.

```python
def _run_loop(...) -> Trajectory:
    env_output = setup_fn()
    # record initial obs as step 0
    task_tool = TaskTool(task, storage, trajectory, summary_proc, tracer, max_steps=self.config.max_steps)
    agent.run(task_tool)
    # finalize trajectory with task_tool.reward / task_tool.done
    close_fn()
    return trajectory
```

`_compute_summary_stats`, `_obs_with_validation_message`, `SummaryProcessor`, tracer spans — all remain in their current homes, called from `TaskTool.call()` instead of the loop body. The harness loop itself disappears.

---

## 4. What agents look like

### Step-mode agent (existing — no changes needed)

```python
class ReactAgent(Agent):
    # step() unchanged — run() is inherited from Agent base and drives the loop
    def step(self, obs: Observation) -> AgentOutput:
        ...
```

### Loop-owning agent (new — overrides run())

```python
class MyFrameworkAgent(Agent):
    def run(self, task_tool: TaskTool) -> AgentOutput:
        while not task_tool.done:
            llm_response = self.llm.call(...)
            action = self._parse_action(llm_response)
            agent_output = AgentOutput(
                actions=[action],
                llm_calls=[llm_response.as_llm_call()],
            )
            task_tool.call(agent_output)
        return AgentOutput()
```

### Claude SDK agent

```python
class ClaudeSDKAgent(Agent):
    def run(self, task_tool: TaskTool) -> AgentOutput:
        # SDK manages its internal loop; task_tool bridges each tool call to cube
        sdk.run(tools=[task_tool.as_sdk_tool()])
        return AgentOutput()
```

The key difference between step-mode and loop-owning: step-mode agents only need to reason about one observation at a time. Loop-owning agents manage history, multi-step planning, and termination themselves.

---

## 5. Alignment with external frameworks

| Framework | How it maps |
|---|---|
| **smolagents** | Agent implements `run(task_as_tool)`. The `TaskTool` acts as the single managed tool in `agent.toolbox`. |
| **LangGraph** | Each LangGraph node that calls an environment action routes through `task_tool.call()`. The graph replaces the harness loop. |
| **AutoGen / CrewAI** | The top-level agent `run()` is the "crew.kickoff()". Each crew action that touches the task environment uses `task_tool.call()`. |
| **Claude Agent SDK** | `agent.run()` wraps `sdk.run(tools=[task_tool.as_mcp_tool()])`. The SDK manages its internal loop; the harness sees the outer `run()` call. |
| **OpenAI Agents SDK** | Same pattern as Claude SDK — `Agent.run()` wraps `openai_agents.Runner.run(agent, tools=[task_tool])`. |

The key insight: all these frameworks want to own the loop and call *tools*. `TaskTool` is exactly that — the cube task exposed as a single tool, with monitoring baked in.

---

## 6. MCP integration (optional / future)

`TaskTool` could expose itself as an MCP server. The existing `cube_harness/mcp/` module already has conversion utilities. An MCP-native agent would connect to the task as a local MCP server and call actions via MCP protocol. The `TaskTool` MCP adapter would intercept those calls and record them identically.

This is optional for the first iteration but worth keeping in mind so the interface doesn't close it off.

---

## 7. What does NOT change

- `Agent.step()` and all existing agents (`ReactAgent`, `GennyConfig`) — unchanged, `run()` default handles them.
- `Episode`, `Experiment`, `exp_runner` public API — unchanged.
- `FileStorage` and on-disk format — identical for both modes.
- Recipes — existing recipes work without modification.
- `AgentConfig.make()` signature — unchanged.

---

## 8. Open questions for discussion

1. **`TaskTool.call()` signature**: Should the agent pass a full `AgentOutput` per action, or should `TaskTool` also accept raw `Action` + `list[LLMCall]` separately? The former mirrors the existing data model cleanly.

2. **Multiple actions per step**: `task.step()` today accepts `list[Action]`. Should `TaskTool.call()` also accept batches? Or enforce one-action-at-a-time for clarity?

3. **Error handling**: If the agent's `run()` raises, should the harness still finalize the trajectory with partial steps? Today `_run_loop` re-raises after saving the error. Same behavior makes sense.

4. **Max steps enforcement**: Today the harness enforces `max_steps`. With a loop-owning agent, `TaskTool` would track the call count and raise (or set `done=True`) when the limit is hit. Is that the right place?

5. **`task_tool.call()` vs direct LLM tool call**: Some frameworks (smolagents, Claude SDK) call tools by returning a structured JSON from the LLM. The `TaskTool` needs to be exposable as an `ActionSchema` / MCP tool description. This is already possible since `action_schemas` mirrors the existing `ActionSchema` list.

6. **Naming**: `TaskTool` vs `MonitoredTask` vs `InstrumentedTask`? The word "tool" emphasizes how the agent sees it; "monitored" emphasizes what the harness does. Worth aligning with the broader `cube` vocabulary.

---

## 9. Minimal implementation plan (not for this branch yet)

1. Add `TaskTool` class (new file `src/cube_harness/task_tool.py`) — wraps task + storage + tracer + summary_proc.
2. Add default `Agent.run(task_tool)` to `agent.py` — calls `self.step()` in a loop via `task_tool`.
3. Simplify `Episode._run_loop` to construct a `TaskTool` and call `agent.run(task_tool)`.
4. Verify parity: run `ReactAgent` on MiniWob with the new path, diff the trajectories against the old path.
5. Write a minimal `LoopOwningAgent` test that overrides `run()` — confirms a loop-owning agent produces the same on-disk format.
6. Write a recipe using a real framework (smolagents or Claude SDK) to confirm ergonomics.
