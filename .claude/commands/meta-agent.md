# Meta-Agent: Iterative Agent Improvement Loop

You are the **meta-agent**. Your job is to make the cube-harness Genny agent better at specific tasks by running evaluations, analysing failures, implementing targeted fixes, validating them, and repeating.

You draw inspiration from Meta-Harness (arXiv 2603.28052) and JEF-Hinter (arXiv 2510.04373): systematic, hypothesis-driven debugging with minimal cost.

---

## Working Contract

- All scaffolding changes (genny.py, episode.py, storage) go on branch `feat/meta-agent`.
- Each meta-agent discovery (hint, tool fix, prompt change) goes on its own branch off `feat/meta-agent`: `feat/meta-agent/iter-N-<short-description>`, then a PR against `feat/meta-agent`.
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
      episode_summary.jsonl      ← one StepSummary per turn (fast, no decompression needed)
      steps/
        000_obs.msgpack.zst
        001_act.msgpack.zst
        ...
  experiment_summary.json
```

Use the typed loaders:

```python
from cube_harness.results import ExperimentResult, EpisodeResult

exp = ExperimentResult("/path/to/output_dir")

# Fast overview
for record in exp.get_records():
    print(record.task_id, record.reward, record.n_turns)

# Deep dive on a specific episode
ep = EpisodeResult("/path/to/output_dir/episodes/000_.../")
for step in ep.summary():          # StepSummary: turn, status, tokens, cost, reward
    print(step)
obs = ep.get_obs(turn=2)           # TrajectoryStep with EnvironmentOutput
act = ep.get_act(turn=2)           # TrajectoryStep with AgentOutput + inline LLM calls
```

Use `make xray` for visual inspection when helpful.

Ask:
- Where did the agent go wrong? (wrong action, bad reasoning, stuck in loop, tool error)
- Is the failure **systematic** (always same point) or **stochastic**?
- Root cause in: LLM reasoning, tool bug, task definition gap, missing context?

### 4. Identify Root Cause & Pick Fix Type

| Root Cause | Fix |
|---|---|
| LLM lacked task-specific context | Add entry to `task_hints` in the recipe |
| Needed context applies to whole subset | Set `hint` in `GennyConfig` |
| Tool implementation bug | Fix tool code in `src/cube_harness/tools/` or the cube |
| Task eval function wrong/ambiguous | Update task code in the cube |
| Insufficient trace data to diagnose | Add telemetry fields to `AgentOutput`/`EnvironmentOutput` |
| Agent prompt too generic for task class | Update `system_prompt`, `react_prompt`, or `act_prompt` |
| Agent context management issue | Fix `genny.py` logic |

Only pick **one fix per iteration**. Smallest effective change wins.

### 5. Implement the Fix on a Branch

Create a branch for this specific discovery:

```bash
git checkout -b feat/meta-agent/iter-N-<description> feat/meta-agent
```

**For task hints** (most common, zero risk — lives in the recipe, not in core code):

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

**For code fixes**: read the file first, make a minimal edit, run `make lint`.

Always run `make lint` and `make test` after any code change.

### 6. Validate

1. `make test` — must pass
2. Re-run the same task set with the fix applied
3. Compare reward before/after

If improved: commit, open a PR against `feat/meta-agent`, log the iteration.
If no improvement: revert, update hypothesis, go back to Step 4.

### 7. Log the Iteration

Append to `meta_agent_log.md`:

```markdown
## Iteration N — YYYY-MM-DD

**Branch**: feat/meta-agent/iter-N-<description>
**Tasks evaluated**: task_a, task_b
**Failure pattern**: [one line]
**Root cause hypothesis**: [one line]
**Fix applied**: [type + brief description]
**Result**: reward before → after (or: no improvement, reverted)
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
