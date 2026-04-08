# Meta-Agent: Iterative Agent Improvement Loop

You are the **meta-agent**. Your job is to make the cube-harness Genny agent better at specific tasks by running evaluations, analysing failures, implementing targeted fixes, validating them, and repeating.

You draw inspiration from Meta-Harness (arXiv 2603.28052) and JEF-Hinter (arXiv 2510.04373): systematic, hypothesis-driven debugging with minimal cost.

---

## Working Contract

- All scaffolding changes (genny.py, episode.py, storage) go on branch `feat/meta-agent`.
- Each meta-agent discovery (hint, tool fix, prompt change, scaffold fix) goes on its own branch off `feat/meta-agent`: `feat/meta-agent/iter-N-<short-description>`, then a PR against `feat/meta-agent`.
- Keep an iteration log at `meta_agent_log.md` in the repo root. Append one entry per iteration.
- Be cost-conscious: prefer targeted re-runs over full benchmark sweeps.
- Every change must be validated before opening a PR.

---

## Entry Point

The starting recipe is `recipes/meta_agent_recipe.py`. Always start from there:

```bash
uv run recipes/meta_agent_recipe.py debug   # sequential, quick sanity check
uv run recipes/meta_agent_recipe.py         # full run on selected task subset
```

The recipe contains two things the meta-agent edits each iteration:
1. `task_ids` — the subset of tasks to focus on
2. `agent_config` — the `GennyConfig` with `hint` and `task_hints`

---

## Task Subset Selection

`MiniWobSubset` (defined in the recipe) wraps `MiniWobBenchmark` and filters `get_task_configs()` to the specified `task_ids`. To change the subset:

```python
task_ids: list[str] = ["click-button", "login-user", "search-engine"]
benchmark = MiniWobSubset(default_tool_config=tool_config, task_ids=task_ids)
```

An empty `task_ids` list runs all 125 MiniWob tasks (full benchmark sweep — avoid unless needed).

Available task IDs come from `miniwob_tasks.json` in `cubes/miniwob/`. To list them:

```python
from miniwob_cube.benchmark import MiniWobBenchmark
for tm in MiniWobBenchmark.task_metadata.values():
    print(tm.id)
```

---

## Iteration Loop

### 0. Orient

Before the first iteration, read:
- `meta_agent_log.md` (if it exists) for prior context
- Recent results in `~/cube_harness_results/` — pick the most recent output dir
- Understand which tasks are currently failing and at what rate

If no prior results exist, go to **Step 1** directly.

### 1. Select Tasks to Evaluate

Choose a small, informative task set (2–5 tasks). Prioritise:
- Tasks that **recently regressed** or have low reward
- Tasks with **high diagnostic value** (diverse failure modes)
- Tasks that are **cheap to run** (few steps, no heavy setup)

Avoid: tasks already passing reliably, tasks that take >60s each without strong reason.

Write your selection rationale in the iteration log.

### 2. Run Evaluation

Update `task_ids` in the recipe, then run:

```bash
uv run recipes/meta_agent_recipe.py debug
```

Note the output directory printed at startup.

### 3. Analyse Traces

Storage is V2 format. Directory layout:

```
output_dir/
  episodes/
    000_GennyConfig_on_miniwob-click-button/
      episode.metadata.json      ← task_id, agent_name, reward
      episode_config.json        ← full GennyConfig for re-launching
      episode_summary.jsonl      ← one StepSummary per turn (fast, no decompression)
      steps/
        000_obs.msgpack.zst      ← EnvironmentOutput (screenshots, axtree, html, reward)
        001_act.msgpack.zst      ← AgentOutput (actions + inline LLM calls with full prompts)
        ...
  experiment_summary.json
```

**Fast overview** — no step loading:

```python
from cube_harness.results import ExperimentResult

exp = ExperimentResult("/path/to/output_dir")
for record in exp.get_records():
    print(record.task_id, record.reward, record.n_turns, record.cost_usd)
```

**Per-episode step access:**

```python
from pathlib import Path
from cube_harness.results import EpisodeResult
from cube_harness.storage import FileStorage

storage = FileStorage(Path("/path/to/output_dir"))
ep = EpisodeResult(Path("/path/to/output_dir/episodes/000_.../"), storage)

for step in ep.summary():   # StepSummary: turn, status, tokens, cost, reward — no decompression
    print(step)

obs_step = ep.get_obs(turn=2)   # TrajectoryStep → .output: EnvironmentOutput
act_step = ep.get_act(turn=2)   # TrajectoryStep → .output: AgentOutput
```

Use `make xray` for visual inspection when helpful.

---

### 3b. Inspect What the Agent Actually Sees

**This is the most important diagnostic step.** Before concluding anything about reasoning quality, verify what the LLM actually received. Every `act` step stores the full prompt (all messages + tools) inside `AgentOutput.llm_calls`.

```python
from cube.core import ImageContent

act_step = ep.get_act(turn=N)
agent_output = act_step.output  # AgentOutput

for llm_call in agent_output.llm_calls:
    print(f"\n=== LLM call: {llm_call.tag} ===")
    print(f"Tokens: prompt={llm_call.usage.prompt_tokens}, completion={llm_call.usage.completion_tokens}")
    for i, msg in enumerate(llm_call.prompt.messages):
        role = msg.get("role") if isinstance(msg, dict) else msg.role
        content = msg.get("content") if isinstance(msg, dict) else msg.content
        if isinstance(content, list):
            # Multimodal: images + text parts
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    print(f"  [{i}] {role}: [IMAGE]")
                elif isinstance(part, dict):
                    print(f"  [{i}] {role}: {str(part.get('text',''))[:300]}")
        else:
            print(f"  [{i}] {role}: {str(content)[:300]}")
    print(f"  → Response: {str(llm_call.output.content)[:300]}")
```

Also inspect the observation directly to understand what the environment provided:

```python
obs_step = ep.get_obs(turn=N)
env_output = obs_step.output  # EnvironmentOutput
for content in env_output.obs.content:
    print(type(content).__name__, ":", str(content.to_markdown())[:300])
```

**Questions to answer while reading the prompt:**

1. **Goal clarity** — Is the task goal clearly stated in step 0? Is it complete?
2. **Observation quality** — Screenshot: is the relevant UI area visible? Is text legible? AXTree: does it list the elements the agent needs to interact with? HTML: is it truncated in a way that hides key elements?
3. **Summary quality** — If `enable_summarize=True`: does the summary accurately capture what happened and what matters? If `enable_summarize=False` (COT mode): does the extracted reasoning in `summaries` correctly reflect progress?
4. **Context window** — With `render_last_n_obs`, how many past observations does the agent see? Is it enough for the task's horizon? Is critical earlier state visible?
5. **Observation truncation** — Is `max_obs_chars` cutting off useful content? Check the `… [truncated]` marker.
6. **Tool visibility** — Are all necessary tools listed in the prompt? Are their descriptions accurate?

---

### 4. Identify Root Cause & Pick Fix Type

| Root Cause | Fix | Blast Radius |
|---|---|---|
| LLM lacked task-specific knowledge | Add `task_hints[task_id]` in recipe | Task only ✅ |
| Missing context applies to whole subset | Set `GennyConfig.hint` | Subset only ✅ |
| Tool implementation bug | Fix tool code | All tasks using that tool ⚠️ |
| Task eval function wrong/ambiguous | Update task code in cube | That task only ✅ |
| Observation not showing relevant UI elements | Fix observation postprocessing in cube task (`obs_postprocess`) | That task only ✅ |
| Screenshot crop misses important area | Fix `obs_postprocess` or viewport config in task | That task only ✅ |
| Summary loses key facts (systematic across tasks) | Improve `summarize_verbose_prompt` or switch `summarize_cot_only` | **All tasks** ❗ |
| COT extraction misses progress (systematic) | Tune `react_prompt` to ask for explicit progress tracking | **All tasks** ❗ |
| Context window too short for task horizon | Increase `render_last_n_obs` | **All tasks** ❗ |
| `max_obs_chars` truncates critical content | Increase limit or switch to AXTree-only obs | **All tasks** ❗ |
| Agent context management logic bug | Fix `genny.py` | **All tasks** ❗ |
| Insufficient trace data to diagnose | Add logging/telemetry fields | All tasks (additive) |

**Priority rule**: always pick the fix with the smallest blast radius. Reach for task hints before touching the generalist scaffold. Only change scaffold (prompts, context window, summarization) when the issue is clearly systematic across multiple task types.

**Generalist & long-horizon warning**: Genny is designed as a generalist agent. Changes to `system_prompt`, `react_prompt`, `render_last_n_obs`, `max_obs_chars`, or `summarize_*_prompt` affect every task and every horizon length. Always validate scaffold changes on a **diverse** task set (not just the failing ones) before merging.

Only pick **one fix per iteration**. Smallest effective change wins.

### 5. Implement the Fix on a Branch

Create a branch for this specific discovery:

```bash
git checkout -b feat/meta-agent/iter-N-<description> feat/meta-agent
```

**For task hints** (most common, zero risk):

```python
agent_config = GennyConfig(
    llm_config=llm_config,
    hint="General guidance for this task subset.",
    task_hints={
        "click-button": "The submit button always has id='subbtn'. Click it directly.",
        "login-user": "Tab between fields. Username field comes first.",
    },
)
```

**For scaffold changes** (prompts, context window, summarization) — read `genny.py` first, make a minimal targeted edit, then validate on a diverse task set:

Key levers in `GennyConfig`:
- `system_prompt` — static system instruction (affects reasoning style globally)
- `react_prompt` — inline COT instruction (used when `enable_summarize=False`)
- `act_prompt` — action-only instruction (used when `enable_summarize=True`)
- `summarize_verbose_prompt` / `summarize_cot_prompt` — what to extract in the summarize pass
- `enable_summarize` — separate summarize LLM call vs. COT extraction from act pass
- `summarize_cot_only` — concise vs. verbose + Key Facts summaries
- `render_last_n_obs` — how many past observations the agent sees (None = all)
- `max_obs_chars` — truncation limit per observation message

**For code fixes**: read the file first, make a minimal edit, run `make lint`.

Always run `make lint` and `make test` after any code change.

### 6. Validate

1. `make test` — must pass
2. Re-run the **same failing tasks** with the fix applied — reward must improve
3. For scaffold changes: also re-run a **diverse control set** (tasks that were passing) — reward must not regress

If improved with no regression: commit, open a PR against `feat/meta-agent`, log the iteration.
If no improvement or regression: revert, update hypothesis, go back to Step 4.

### 7. Log the Iteration

Append to `meta_agent_log.md`:

```markdown
## Iteration N — YYYY-MM-DD

**Branch**: feat/meta-agent/iter-N-<description>
**Tasks evaluated**: task_a, task_b
**What the agent saw**: [key observation about prompt content — truncation, missing elements, bad summary, etc.]
**Failure pattern**: [one line]
**Root cause hypothesis**: [one line]
**Fix applied**: [type + brief description, blast radius]
**Result**: reward before → after (or: no improvement, reverted)
**Control set check**: [pass/fail/skipped — note any regressions]
**Next hypothesis**: [what to try next]
```

### 8. Decide: Continue or Stop

Continue if there are still failing tasks with diagnosable root causes.
Stop if all selected tasks pass, or stuck on the same failure 3 iterations in a row (escalate to user).

---

## Key Files

| What | Where |
|---|---|
| **Start here** | `recipes/meta_agent_recipe.py` |
| Genny agent | `src/cube_harness/agents/genny.py` |
| Episode runner | `src/cube_harness/episode.py` |
| Results API | `src/cube_harness/results.py` |
| Summary models | `src/cube_harness/summary.py` |
| Results root | `~/cube_harness_results/` |
| XRay viewer | `make xray` |
| Tests | `make test` |
| Lint | `make lint` |

## Hint Priority

`task_hints[task_id]` → `hint` → no hint (in that order). Both are set on `GennyConfig` and serialised into the episode config for full reproducibility.

## Genny Context Layout (per act call)

Understanding this is essential for diagnosing what the agent sees:

```
[system]  system_prompt                    ← static
[user]    goal (step-0 observation)        ← static after step 0
[user]    ## Task Hint\n{hint/task_hint}   ← if set; static
[asst]    "Understood..."                  ← hint ack; static
[asst]    ## Summary of past interactions  ← collapsed history (rolling COT or summarize pass)
[user]    ## N most recent observations    ← window header
...       windowed raw obs+asst groups     ← last render_last_n_obs observations
[user]    react_prompt / act_prompt        ← static instruction
```

The hint + goal are in the static prefix → prompt cache hits. Summaries grow with each step. The observation window is the main knob for long-horizon tasks: wider window = more context but higher cost and token pressure.
