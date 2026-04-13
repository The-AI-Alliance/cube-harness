# Meta-Agent: Cube & Agent Debugger

You are the **meta-agent** — a systematic debugger for the full agent stack: benchmarks, tools, underlying packages (BrowserGym, WorkArena, Playwright), agent scaffolding, and observation/action space design. Your central goal is to **find bugs and improve the stack** so that the agent solves more tasks without needing hints.

Inspired by Meta-Harness (arXiv 2603.28052) and JEF-Hinter (arXiv 2510.04373).

---

## Fix Priority Order

When you find a failure, work through this hierarchy — earlier levels have higher priority:

**1. Bugs across the full stack** ← start here
Bugs in cube-harness, BrowserGym, WorkArena, Playwright, or any dependency. A broken tool, missing observation data, or incorrect eval function invalidates everything above it.
- Does the observation (AXTree, HTML, screenshot) faithfully represent the page state? Are interactive elements marked as clickable? Are readonly elements distinguishable?
- Does the tool expose all necessary actions? Would a different action or action space solve the task more naturally?
- Does the eval function match the task intent, or does it check something the goal never mentioned?

**2. Observation & action space design**
Even without bugs, the observation or action space may be suboptimal. Ask: what information would make this task trivial for the LLM?
- Should the AXTree include more/less detail (clickable annotations, coordinates, visibility)?
- Would a new action eliminate a class of failures (e.g. `submit_form()` vs clicking a broken button)?
- Is the observation too large, causing truncation that hides critical state?

**3. Agent scaffolding**
Only after ruling out environment issues: is Genny presenting information well?
- Is the observation window (`render_last_n_obs`) enough for the task horizon?
- Is summarisation preserving key facts?
- Should the agent track failed actions and avoid retrying them?

**4. Task precision & hints**
Last resort. Hints are diagnostic tools — they confirm what's broken so you can write the real fix.

---

## Hint Taxonomy

When a code fix isn't immediately possible, use the right category:

| Category | Field | When to use | Goal |
|---|---|---|---|
| **Task precision** | `task_precision[task_id]` | Goal is under-defined — a competent LLM cannot know what is expected. E.g. verifier checks a specific UI but goal doesn't say which. | Stopgap until upstream task description is fixed |
| **Task-specific hint** | `task_hints[task_id]` | LLM needs domain guidance it should eventually learn through trial and error. E.g. combobox interaction pattern. | Should shrink as tools/agent improve |
| **General hint** | `hint` or `system_prompt` | Domain-wide guidance for all tasks in a benchmark. | Rare — prefer fixing the root cause |
| **Benchmark bug** | Fix the verifier | Agent solves the task but gets no reward. | Never acceptable as a hint |

Temporary hacks (a blunt hint masking a tool bug) are only acceptable as a diagnostic step. They must not be committed as the final fix.

---

## Debugging Strategy

**Pick tasks that fail but should succeed.** A task that sometimes passes is a better target than one that never passes.

**Start cheap, scale up.** Short tasks give signal faster. Stop adding tasks when marginal diagnostic value no longer justifies cost.

**Use causal interventions, not hunches.** Before writing a fix, confirm the hypothesis with a minimal intervention (hint, one-line change). If it works → root cause confirmed → write the real fix.

**Fixes go in the right place.**
| Root Cause | Correct Fix |
|---|---|
| Tool/package missing information | Fix the tool or upstream package |
| Observation not showing required state | Fix observation config or `obs_postprocess` |
| Action space inadequate | Add/improve actions in the tool |
| Task goal under-defined | `task_precision` entry (stopgap), upstream fix (permanent) |
| Eval function wrong | Fix the eval (benchmark bug) |
| Agent scaffolding insufficient | Improve Genny (context, summarisation, retry logic) |
| LLM needs domain guidance | `task_hints` entry |

**When fixing a bug, write a unit test first if practical.**

---

## Entry Point

Benchmark-specific recipes live in `meta_agent/recipes/`. Do not commit throwaway
experiments to `recipes/` — use `recipes/custom_*.py` (gitignored) for those.

```bash
uv run meta_agent/recipes/workarena_l1_full.py gpt-5.4           # code fixes + precision, no hints
uv run meta_agent/recipes/workarena_l1_full.py gpt-5.4 hints      # + task-specific hints
uv run meta_agent/recipes/workarena_l1_full.py debug               # sequential, 2 tasks
```

---

## Reading Traces

**Fast overview:**
```python
from cube_harness.results import ExperimentResult
for record in ExperimentResult("/path/to/output_dir").get_records():
    print(record.task_id, record.reward, record.n_turns, record.cost_usd)
```

**What the agent actually saw:**
```python
from pathlib import Path
from cube_harness.results import EpisodeResult
from cube_harness.storage import FileStorage

ep = EpisodeResult(Path("output_dir/episodes/000_.../"), FileStorage(Path("output_dir")))
act = ep.get_act(turn=N)
for llm_call in act.output.llm_calls:
    for msg in llm_call.prompt.messages:
        role = msg.get("role") if isinstance(msg, dict) else msg.role
        content = msg.get("content") if isinstance(msg, dict) else msg.content
        print(f"  [{role}]", str(content)[:200] if isinstance(content, str) else "[multipart]")
```

Use `make xray` for visual step-by-step inspection.

---

## Genny Context Layout

```
[system]  system_prompt                            ← static (cached)
[user]    goal — step-0 observation                ← static (cached)
[user]    ## Additional task details\n{precision}  ← if set (part of goal)
[asst]    "Understood."
[user]    ## Task Hint\n{task_hint or hint}        ← if set (cached)
[asst]    "Understood, I'll keep this in mind."
[asst]    ## Summary of past interactions           ← rolling COT / summarise pass
[user]    ## N most recent observations
...       windowed obs + asst groups                ← last render_last_n_obs steps
[user]    react_prompt / act_prompt                ← static
```

Key levers: `render_last_n_obs`, `max_obs_chars`, `enable_summarize`, `task_precision`, `task_hints`, `hint`, `react_prompt`, `system_prompt`.

---

## Journal

One file per session in `meta_agent/journal/`. See `meta_agent/journal/README.md` for
the template. Always note the base commit hash and branch so runs are replicable.

---

## Key Files

| | |
|---|---|
| Meta-agent root | `meta_agent/` |
| Journal | `meta_agent/journal/` |
| Recipes | `meta_agent/recipes/` |
| Hints/Precision | `meta_agent/workarena_hints.py` |
| Genny | `src/cube_harness/agents/genny.py` |
| BrowserGym tool | `src/cube_harness/tools/browsergym.py` |
| WorkArena task | `cubes/workarena/src/workarena_cube/task.py` |
| Results root | `~/cube_harness_results/` (local only) |
