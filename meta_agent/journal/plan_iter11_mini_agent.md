# Plan: Iter 11 — MiniAgent clone (faithful upstream port, reproducible subset)

**Date:** 2026-05-01
**Status:** plan locked, ready for implementation

## Why this plan exists

Iter 8–10 explored Genny variants (windowed obs, max_actions bumps, mini-SWE-style
prompt). Best result so far: 38.7% on our 54-task subset with `gpt-5.4-mini`. Public
data on the same model class:

- **Vals.ai** (mini-SWE-agent bash-only, 500 tasks): `gpt-5.4-mini` likely lands ~70%+
  (time-bucketed: 86% <15min, 71% 15m–1h). Specific 73.3% headline is derived, not
  directly published.
- **swebench.com** official bash-only leaderboard: GPT-5 mini (older, 2025-08-07) at
  56–60%. gpt-5.4-mini should beat that.
- **HAL Princeton mini leaderboard** (50-task subset): GPT-5 medium 46%, o4-mini-low
  54%. No gpt-5-mini-tier number, but band is 40–55%.

Our 38.7% is well below all anchors, even being conservative. **The gap is in our
scaffold, not the model.** Conclusion from iter9–10: rather than keep tuning Genny,
build a faithful port of mini-SWE-agent and make it pass on a publicly-fixed subset.

## Goal

Reach the published mini-SWE-agent number (or within 5pp) on a fixed reproducible
subset, using a stripped-down `MiniAgent` that bypasses Genny entirely.

## Reproducible target subset

**HAL "SWE-bench Verified Mini"** — 50 fixed tasks (25 django + 25 sphinx).

- Task ID list: https://raw.githubusercontent.com/princeton-pli/hal-harness/main/hal/benchmarks/swebench_verified_mini_task_ids.txt
- HF mirror: `MariusHobbhahn/swe-bench-verified-mini` (parquet, pinned)
- Used by: HAL leaderboard, mini-SWE-agent reproducibility runs, 13+ models with
  published scores

**Caveat:** the 50 are all django + sphinx, not a random sample of Verified. Gives
narrow generalization but maximal comparability with public numbers.

**Initial target band for `gpt-5.4-mini`: 40–55% on HAL-50.**
**Stretch (if our setup is healthy): 55–65%.**

Once HAL-50 is healthy, scale to full 500 for a Vals.ai/swebench.com comparison.

## Architecture decisions (resolved)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Extend `cube_harness.Agent` base class** (option a, not standalone runner) | Gets parallelism, XRay, trajectory persistence, summary stats for free. Loop maps cleanly: `model.query()` ≡ `Agent.act()`, exec ≡ tool, append ≡ trajectory step. |
| 2 | **No cube-standard API changes** | `SWEBenchVerifiedTask.evaluate()` doesn't reset `/testbed` — runs tests on whatever the agent edited. Magic-string is purely a "stop early" signal; agent's edits already on disk. Workspace-based eval is the cube norm anyway. |
| 3 | **Skip cost-limit** for now | Not blocking. LiteLLM callback later if needed. |
| 4 | **Use `BashOnlySWEBenchTool`** as-is | Already maps to `container.exec(cmd, workdir=/testbed)` — fresh subshell per call, matches upstream `subprocess.run` semantics. Already concatenates stdout+stderr. |

## MiniAgent design (verbatim upstream port)

**Reference:** github.com/SWE-agent/mini-swe-agent, branch `main`.

### Loop (from `src/minisweagent/agents/default.py`)

```
messages = [system, instance]              # both Jinja-rendered with {{task}}, etc.
while True:
    response = model.query(messages, tools=[BASH_TOOL])    # native tool calls
    messages.append(response)
    for tool_call in response.tool_calls:
        cmd = tool_call.arguments["command"]
        out = container.exec(cmd, workdir="/testbed", timeout=60)
        if first_line(out.stdout) == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and out.exit_code == 0:
            return done=True   # signal episode end
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": render(out)})
    if step_count >= step_limit:
        return done=True
```

### Bash tool schema (from `src/minisweagent/models/utils/actions_toolcall.py`)

```python
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}
```

### Prompts (from `src/minisweagent/config/benchmarks/swebench.yaml`)

**System (verbatim):**
> You are a helpful assistant that can interact with a computer shell to solve programming tasks.

**Instance template:** copy verbatim from upstream YAML. Includes `<pr_description>{{task}}</pr_description>`,
workflow steps (analyze → reproduce → fix → verify), and submission instructions
mandating `git diff > patch.txt && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`.

### Observation rendering

- If `len(output) < 10000` chars → emit full block.
- Else → head 5000 + `<elided_chars: N>` + tail 5000 + nudge to use head/tail/grep.

### Defaults

- `step_limit=250` (model turns)
- `temperature=0`
- `parallel_tool_calls=true`
- `drop_params=true`
- `bash_timeout=60s` (consider bumping to 600 for slow test suites — upstream issue)
- bash env: `PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1`

### Submission detection

```python
def is_submission(stdout: str, exit_code: int) -> bool:
    if exit_code != 0:
        return False
    lines = stdout.lstrip().splitlines()
    return bool(lines) and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
```

The first stripped line of stdout matters; everything after is the diff (we ignore it
since `evaluate()` reads `/testbed` directly).

## File plan

1. **`src/cube_harness/agents/mini_agent.py`** (NEW, ~200 lines)
   - `MiniAgentConfig(AgentConfig)` — Pydantic config: prompts, step_limit, temp, bash_timeout
   - `MiniAgent(Agent)` — owns `self.messages` per episode, implements `act()` returning a `bash` action, handles tool-call parsing and submission detection.
   - Submission detection lives here (not in tool); when triggered, agent returns a `final_step` action which Episode interprets as done.

2. **`recipes/mini_agent_recipe.py`** (NEW, ~150 lines)
   - Wires `MiniAgent` + `BashOnlySWEBenchTool` + `SWEBenchVerifiedBenchmarkConfig`.
   - CLI args mirror `mini_swe_recipe.py` (`--toolkit`, `--n-parallel`, `--subset hal_mini`, etc.).
   - Runs via `run_with_ray`.

3. **`cubes/swebench-verified-cube/src/swebench_verified_cube/benchmark.py`** (EDIT)
   - Add `hal_mini` named subset. Either fetch IDs from HAL repo at install-time, or
     embed the 50 IDs as a constant (preferred — stable, no network dependency).

4. **`cubes/swebench-verified-cube/tests/test_benchmark.py`** (EDIT)
   - Add test for `hal_mini` subset (50 tasks, all django/sphinx).

## Open questions to resolve during implementation

- **Action-space integration:** how does `MiniAgent.act()` return a bash command in
  cube-harness's action-set framework? Probably a single `bash` `ActionSchema` plus
  `final_step`. Verify against `cube_harness.action_spaces`.
- **Episode loop and tool execution:** does the harness Episode call
  `tool.bash(cmd)` directly, or does it go through `Action(name="bash", arguments={...})`?
  Confirm path so submission detection happens at the right layer.
- **Multi-tool-call per turn:** upstream allows multiple `bash` tool calls per LLM
  response. cube-harness's Episode is `act() → step()` (one action per env step).
  Either flatten to one-per-step (recommended for simplicity) or buffer multiple
  actions. Default: one bash call per turn — drop the upstream multi-call feature
  unless it's needed for parity.
- **Tool-call message threading:** LiteLLM tool-call responses need
  `tool_call_id`-keyed reply messages. Verify our LLM wrapper preserves these.

## Implementation sequence

1. **Add `hal_mini` subset** (~30 min). Hardcode 50 IDs as constant in benchmark.py.
   Add unit test. Confirm `mini_swe_recipe.py --subset hal_mini` lists 50 tasks.
2. **Skeleton `MiniAgent`** (~half day). Bare loop, system+instance prompts, single
   bash tool, magic-string detection, no truncation. Pass linting. Add unit tests for
   `is_submission()` and prompt rendering.
3. **Add `mini_agent_recipe.py`** (~1h). Mirror `mini_swe_recipe.py` shape.
4. **Local smoke test** (~30 min): `--debug` against 1–2 tasks with magic-string
   exit. Verify trajectory storage, evaluate() runs, reward returns.
5. **HAL-50 run on Toolkit, gpt-5.4-mini, n-parallel 20** (~45 min wall clock).
   Compare to 40–55% target band.
6. **Iterate on gaps.** If <40%: investigate prompt fidelity, tool-call threading,
   container env vars. If 40–55%: scale to full 500-task SWE-bench Verified.
7. **(Stretch) Cost-limit hook + bash_timeout=600 + scaling experiments.**

## Anti-goals

- Do NOT modify Genny.
- Do NOT change cube-standard / cube-harness public APIs.
- Do NOT add features beyond upstream parity in v1 (no rolling summary, no windowed
  obs, no chain-of-thought extraction, no per-step linting).
- Do NOT spend time on prompt tuning until baseline upstream port is confirmed
  working. Verbatim first, optimize second.

## Success criteria

- `MiniAgent + mini_agent_recipe.py` + `hal_mini` subset all merged behind a feature
  flag (separate recipe, no behavior change for existing flows).
- HAL-50 with `gpt-5.4-mini`: ≥40% (matches GPT-5 medium 46% within band).
- If HAL-50 ≥55%: scale to full 500, target ≥60% (within 10pp of swebench.com bash
  leaderboard for similar tier).

## Notes for downstream work

- Once MiniAgent is solid, the same pattern works for **swebench-live-cube**
  (already has `BashOnlySWEBenchTool`). Just point the recipe at swebench-live.
- Cost-limit + bash-timeout-bump are easy follow-ups.
- Multi-tool-call-per-turn could be re-enabled if it materially helps (upstream uses
  it for parallel test runs).
- Eventually consider letting the agent own its own loop (Alex's note) — would
  require a cube-harness API extension. Out of scope for iter11.
