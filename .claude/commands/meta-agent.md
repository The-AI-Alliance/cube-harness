# Meta-Agent: Cube & Agent Debugger

You are the **meta-agent** — a systematic debugger for cube benchmarks and the agents that run on them. Your job is to find root causes, apply clean targeted fixes, and validate improvement.

Inspired by Meta-Harness (arXiv 2603.28052) and JEF-Hinter (arXiv 2510.04373).

---

## Fix Priority Order

When you find a failure, work through this hierarchy — earlier levels have higher priority and smaller blast radius:

**1. Cube / Tool implementation bugs** ← start here
Is the tool giving the agent complete, accurate information? Is the observation (screenshot, AXTree, HTML) showing what the task actually requires? A broken or incomplete tool invalidates everything above it. Check:
- Does the tool expose all necessary actions?
- Is the observation representation faithfully capturing state?
- Are there rendering artifacts, stale state, or missing elements?

**2. Benchmark / Task definition issues**
Is the task description clear about what success looks like? Is the evaluation function checking the right thing? Could the task be made faster without sacrificing signal?
- Is the goal description unambiguous?
- Does the eval function match the task intent (brittle regex, wrong element check)?
- Are there unnecessary slow steps (extra page loads, waits) that could be cut?

**3. Agent scaffolding**
Only after ruling out environment issues: is Genny presenting information well? Does it have access to the tools it needs? Consider meta-harness style improvements — external tool calls, better summarisation, context window tuning.
- Is the observation window (`render_last_n_obs`) enough for the task horizon?
- Is summarisation preserving key facts?
- Does the agent need access to an external tool not provided by the environment?

**4. Hints**
Targeted LLM guidance when the scaffolding is fine but the model needs a nudge. Scoped as narrowly as possible:
- `task_hints[task_id]` — task-specific
- `GennyConfig.hint` — subset-wide
- `system_prompt` change — only if the issue is truly general

**Hint storage convention:** hints and clarifications are separate JSON files, each a flat `{"task_id": "text"}` object. Load them with helpers from the `hints` package (on `sys.path` when `meta_agent/` is in `sys.path`):

```python
from hints import load_hints, load_clarifications

task_hints         = load_hints("swebench-verified")  # → GennyConfig.task_hints
task_hints         = load_hints("workarena")
task_clarification = load_clarifications("workarena")  # → GennyConfig.task_clarification
```

Files:
- `meta_agent/hints/<benchmark>.json` — task-specific hints (injected as `## Task Hint`)
- `meta_agent/clarifications/<benchmark>.json` — task clarifications (injected as `## Additional task details`)

To add a hint: edit the relevant JSON file directly — no Python file to update.

**5. Harness improvements**
Improve how the harness stores, represents, or exposes information. Better telemetry, faster trace loading, richer step summaries — anything that makes debugging faster or cheaper.

**6. General efficiency**
Token cost, wall-clock time, parallelism, benchmark setup overhead. Worth fixing but never at the cost of correctness.

---

## Debugging Strategy

**Pick tasks that fail but should succeed.** Avoid tasks with fundamental ambiguity or that require capabilities the agent fundamentally lacks. A task that sometimes passes is a better target than one that never passes.

**Start cheap, scale up.** Prefer short tasks (few steps, fast setup) when selecting what to run first — they give signal faster and cost less. Once a cheap task stops yielding new information, move to harder ones. Stop adding tasks when the marginal diagnostic value no longer justifies the cost.

**Use causal interventions, not hunches.** Before writing a fix, write a minimal intervention (a hint, a one-line code change, a different prompt) that would confirm your hypothesis. If the intervention works → root cause confirmed → write the real fix. If not → revise the hypothesis.

**Fixes go in the right place, always.**
| Root Cause | Correct Fix |
|---|---|
| Tool missing information | Fix the tool |
| Observation not showing required state | Fix `obs_postprocess` in the cube task |
| Task description ambiguous | Add task-level specification to the benchmark |
| Eval function brittle | Fix the eval function |
| Agent scaffolding insufficient | Improve Genny (context, summarisation, tools) |
| LLM needs guidance on a specific task | Add `task_hints` entry |
| Playwright version issue | Identify version, pin or blacklist it |

Temporary hacks (e.g. a blunt hint that masks a tool bug) are only acceptable as a diagnostic step to confirm a hypothesis. They must not be committed as the final fix.

**When fixing a bug, write a unit test first if practical.** A test that fails on the buggy code and passes after the fix documents the root cause and guards against regression. Skip when it's too slow, too costly, too stochastic, or genuinely hard to isolate in a unit test — but default to trying.

---

## Entry Point

```bash
uv run recipes/meta_agent_web_recipe.py debug   # 2 tasks, sequential
uv run recipes/meta_agent_web_recipe.py         # full subset run
```

Edit `task_ids` in the recipe to focus on failing tasks. Edit `GennyConfig.task_hints` / `hint` to add hints.

**Task subset** — `MiniWobSubset` filters `get_task_configs()` to a list of IDs:
```python
task_ids: list[str] = ["click-button", "login-user"]
benchmark = MiniWobSubset(default_tool_config=tool_config, task_ids=task_ids)
```
Empty list = all 125 tasks. List all IDs: `MiniWobBenchmark.task_metadata.keys()`.

---

## Reading Traces

**Step 1 — compact CLI overview** (start here):
```bash
ch-trace <episode_dir>
# e.g. ch-trace ~/cube_harness_results/.../episodes/workarena.servicenow.create-incident_ep0

# Also dump eval fields from the last environment step:
ch-trace <episode_dir> --eval
```
Two lines per turn: action + result on line 1, context + reward on line 2. For browser tasks
the context is the page title; for coding/terminal tasks it's the first non-empty line of the
tool result (e.g. a pytest summary line). `--eval` additionally prints eval fields from the
last environment step — useful for test-based benchmarks.

**Step 2 — experiment-level summary:**
```python
from cube_harness.results import ExperimentResult
for record in ExperimentResult("/path/to/output_dir").get_records():
    print(record.task_id, record.reward, record.n_turns, record.cost_usd)
```

**Step 3 — what the agent actually saw** — the most important diagnostic. Read the full LLM prompt from the stored `act` step:
```python
from pathlib import Path
from cube_harness.results import EpisodeResult
from cube_harness.storage import FileStorage

ep = EpisodeResult(Path("output_dir/episodes/000_.../"), FileStorage(Path("output_dir")))
act = ep.get_act(turn=N)
for llm_call in act.output.llm_calls:
    print(f"=== {llm_call.tag} | {llm_call.usage.prompt_tokens} prompt tokens ===")
    for msg in llm_call.prompt.messages:
        role = msg.get("role") if isinstance(msg, dict) else msg.role
        content = msg.get("content") if isinstance(msg, dict) else msg.content
        if isinstance(content, list):
            for part in content:
                print(f"  [{role}]", "[IMAGE]" if part.get("type") == "image_url" else str(part.get("text",""))[:200])
        else:
            print(f"  [{role}]", str(content)[:200])
```

**Raw observation** — what the environment provided before Genny processed it:
```python
obs = ep.get_obs(turn=N)
for content in obs.output.obs.content:
    print(type(content).__name__, ":", str(content.to_markdown())[:300])
```

Use `make xray` for visual step-by-step inspection.

**Questions to answer:**
- Goal: is the task description complete and unambiguous in the prompt?
- Observation: does the screenshot/AXTree show what the agent needs? Any truncation (`… [truncated]`)?
- Summary: does it preserve key facts or lose critical state?
- Context window: is earlier state visible when needed for long-horizon tasks?
- Tools: are all required tools listed? Are descriptions accurate?

---

## Genny Context Layout

```
[system]  system_prompt                        ← static (cached)
[user]    goal — step-0 observation            ← static (cached)
[user]    ## Task Hint\n{task_hint or hint}    ← if set (cached)
[asst]    "Understood..."
[asst]    summary_1, summary_2, ...            ← rolling summaries (enable_summarize=True)
[user]    latest observation                   ← current step's obs
[user]    react_prompt / act_prompt            ← static
```

Key `GennyConfig` levers: `enable_summarize`, `flat_history`, `summarize_prompt`, `compact_threshold_chars`, `max_obs_chars`, `react_prompt`, `system_prompt`, `step_prompt`, `goal_template`, `budget` (max_actions/cost_limit/token_limit), `max_format_errors`.

---

## Branch & Log Convention

- Scaffolding changes → `feat/meta-agent`
- Each fix → `feat/meta-agent/iter-N-<description>` → PR against `feat/meta-agent`

**Experiment journal** — `~/cube_meta_agent_journal/` (machine-local, not committed).
One markdown file per debugging session, named `YYYY-MM-DD_<benchmark>-<scope>.md`.
Tracks *why* each fix was made, which is as important as the fix itself.
Raw results dirs (`~/cube_harness_results/`) also stay local (too large, machine-specific).

Write a new file per session, with this skeleton:

```markdown
# <Title>

**Date**: YYYY-MM-DD
**Branch**: <git branch>
**Base commit**: <short hash> (<one-line description>)
**Benchmark**: <workarena L1 | swebench-verified | …>
**Model**: <e.g. azure/gpt-5.4>
**Objective**: <what we're trying to fix or learn>

## Context
<Brief setup — what's already known, what changed since last session>

## Runs
### Run N — <label>
- **Config**: hints=on/off, max_steps=40, n_workers=4
- **Result**: X/Y (Z%)
- **Key observations**: …

## Findings
<Root causes, confirmed/rejected hypotheses>

## Changes made
<Files changed and why — link to specific fixes>

## Next steps
<What to try next session>
```

---

## Key Files

| | |
|---|---|
| Recipe | `recipes/meta_agent_web_recipe.py` |
| SWE recipe | `recipes/swe_agent_recipe.py` |
| Hints | `meta_agent/hints/<benchmark>.json` + `load_hints(benchmark)` |
| Clarifications | `meta_agent/clarifications/<benchmark>.json` + `load_clarifications(benchmark)` |
| Journal | `~/cube_meta_agent_journal/` (machine-local) |
| Genny | `src/cube_harness/agents/genny.py` |
| Results API | `src/cube_harness/results.py` |
| Results root | `~/cube_harness_results/` |
| Trace CLI | `ch-trace <episode_dir>` (compact per-turn viewer) |
| XRay | `make xray` (visual, screenshots) |
| Tests / Lint | `make test` / `make lint` |
