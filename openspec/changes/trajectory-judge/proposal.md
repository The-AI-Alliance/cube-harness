# RFC: Trajectory Judge — Post-Hoc Failure Analysis for Agent Episodes

**Status:** DRAFT  
**Author:** Alexandre Lacoste  
**Reviewer:** @NicolasAG  
**Date:** 2026-05-04

---

## Problem

cube-harness generates trajectories and measures reward, but produces no structured answer
to the most important question in agent research: *why* did a given episode fail?

Today, diagnosing a failure requires opening the trajectory in XRay, reading the full
conversation, and manually forming a hypothesis. At scale — dozens of runs, hundreds of
episodes — this is not tractable. Worse, human judgements are inconsistent across
reviewers and over time; the same trajectory reviewed twice may yield different root-cause
attributions.

Three concrete gaps this RFC addresses:

1. **No blame attribution.** Summary statistics (pass rate, cost, steps) tell you *how
   much* a system fails, but not where the budget to fix it should go — agent prompt,
   scaffolding design, environment brittleness, task ambiguity, or evaluation harness.

2. **No structured evidence.** Free-form researcher notes in journals or PRs rot. A
   trajectory-linked evidence record stays coherent as the codebase evolves.

3. **No hypothesis tracking.** Improvements are shipped without a testable hypothesis.
   If the hypothesis is recorded alongside the evidence, subsequent experiments can
   confirm or refute it quantitatively.

---

## Scope

- New module `cube_harness/analyze/judge.py` with a single public entry point.
- New Pydantic model `JudgeOutput` (V1 schema defined below).
- CLI runner and batch runner for post-hoc analysis of experiment output directories.
- Populate `judge_output` in `EpisodeRecord` (atlas-eval-log RFC, PR #297) when a judge
  has been run.
- No changes to the episode loop, trajectory format, storage protocol, or existing
  agent/benchmark contracts.

---

## Design

### Approach: LLM-as-judge via Claude Code Python API

The judge is a Claude Code agent invoked programmatically using the `claude` Python
SDK (or subprocess). It receives a structured prompt containing:

- The full trajectory (serialized as a flat conversation log with step indices).
- A concise codebase map (relevant source files: agent prompt, tool definitions, benchmark
  task description) — enough context to distinguish a scaffolding failure from a model
  capability failure.
- Optionally, a small cohort of related trajectories on the same task (to enable
  contrastive analysis: "this agent solved it, this one didn't — why?").

The judge outputs a JSON object conforming to `JudgeOutput`. The structured fields enable
aggregation and statistical analysis; the free-form `analysis` field captures reasoning
that doesn't fit in a taxonomy.

**Why LLM-as-judge, not heuristics?**

Failure mode classification requires understanding the agent's intent (did it *try* the
right approach and fail, or never try?), the task's ambiguity (is the description
underspecified?), and whether the evaluation harness is fair. No pattern-matching
heuristic can do this reliably. A strong LLM can read the conversation and make the
judgment — the same way a human researcher would, but consistently and at scale.

**Why Claude Code specifically?**

Claude Code's agentic loop with file-reading tools means the judge can inspect the
actual source files it's told are relevant, rather than trusting summaries embedded in
the prompt. This reduces hallucination risk when the judge attributes blame to a specific
prompt or scaffolding decision.

**Hallucination resistance**

The judge prompt is structured to minimize confabulation:
- Evidence is required: structured `evidence` fields must quote specific steps and
  transcript excerpts.
- Confidence signals are explicit: `primary_blame_confidence` and `hypothesis_confidence`
  are required enum fields, forcing the model to express uncertainty rather than hide it.
- Blame categories are closed-world: the judge picks from a fixed taxonomy and must
  assign `none` if no clear cause is found, rather than inventing a plausible-sounding one.

---

## V1 Schema: `JudgeOutput`

```python
class BlameCategory(str, Enum):
    task_unclear     = "task_unclear"      # Task description is ambiguous or underspecified
    model_capability = "model_capability"  # Agent lacks the reasoning/knowledge for this task
    tool_or_env      = "tool_or_env"       # Tool crash, env error, or infra failure
    scaffolding      = "scaffolding"       # Agent loop, prompt format, or budget design
    eval_brittle     = "eval_brittle"      # Evaluator rejects a correct solution
    submission_format = "submission_format" # Agent didn't call the right termination action
    none             = "none"              # Success, or cause is genuinely unclear

class Confidence(str, Enum):
    high   = "high"
    medium = "medium"
    low    = "low"

class Outcome(str, Enum):
    success        = "success"        # Agent solved the task
    success_lucky  = "success_lucky"  # Solved but by accident / wrong approach
    failure        = "failure"        # Task not solved
    indeterminate  = "indeterminate"  # Outcome is ambiguous or evaluator is unreliable

class EvidenceItem(TypedBaseModel):
    step: int                # Step index in the trajectory
    quote: str               # Verbatim excerpt from the agent or environment output

class JudgeOutput(TypedBaseModel):
    outcome: Outcome
    primary_blame: BlameCategory
    primary_blame_confidence: Confidence
    other_blames: list[BlameCategory] = []
    summary: str             # 1–3 sentences: what happened
    evidence: list[EvidenceItem]
    hypothesis: str          # 1–2 sentences: what change would most likely help
    hypothesis_confidence: Confidence
    analysis: str            # Free-form multi-paragraph narrative
```

### Blame taxonomy

| Category | Use when |
|---|---|
| `task_unclear` | The task description is ambiguous, contradictory, or missing necessary context. |
| `model_capability` | The agent understands the task but lacks the reasoning ability, domain knowledge, or multi-step planning to solve it. |
| `tool_or_env` | A tool raised an exception, an environment reset failed, a container crashed, or infra was unavailable — outside the agent's control. |
| `scaffolding` | The agent loop, system prompt, budget limits, context window management, or submission protocol caused the failure — not the underlying LLM. |
| `eval_brittle` | The agent produced a correct or acceptable solution but the evaluator rejected it (e.g. wrong whitespace, order-sensitive string match, stale ground truth). |
| `submission_format` | The agent reached a correct solution but failed to submit it through the required channel (e.g. never called `final_step`, submitted to the wrong tool). |
| `none` | Assign on success, or when the episode is too ambiguous to assign a blame without speculation. |

**Multi-blame:** `primary_blame` is the dominant cause. `other_blames` captures secondary
contributing factors. For example, a submission error (`submission_format`) may co-occur
with a `scaffolding` issue if the agent prompt doesn't mention the submission tool.

---

## Implementation

### Module layout

```
src/cube_harness/analyze/
├── xray.py              (existing)
├── inspect_results.py   (existing)
├── xray_utils.py        (existing)
└── judge.py             ← NEW
```

### Public API

```python
def judge_episode(
    trajectory: Trajectory,
    *,
    agent_config: AgentConfig,
    task_description: str,
    related_trajectories: list[Trajectory] | None = None,
    model: str = "claude-opus-4-7",
    codebase_files: list[Path] | None = None,
) -> JudgeOutput:
    """Run a post-hoc judge on a single episode trajectory."""
    ...

def judge_experiment(
    output_dir: Path,
    *,
    model: str = "claude-opus-4-7",
    n_parallel: int = 4,
    overwrite: bool = False,
) -> dict[str, JudgeOutput]:
    """Batch judge all episodes in an experiment output directory.

    Writes judge_output.json alongside each episode_record.json.
    Returns a mapping from trajectory_id to JudgeOutput.
    """
    ...
```

### Prompt structure

The judge prompt is rendered from a Jinja2 template (checked in at
`src/cube_harness/analyze/judge_prompt.md.j2`) with three sections:

1. **Context** — agent config summary, task description, codebase files (if provided).
2. **Trajectory** — flat conversation log with step indices, tool calls, and env responses.
3. **Instructions** — taxonomy definitions, output schema, evidence requirements,
   confidence calibration guidelines.

The judge is asked to return a JSON block only; the response is parsed with
`json.loads` with fallback to a regex extractor for common LLM wrapping.

### Integration with `EpisodeRecord`

When `judge_experiment()` is run on an output directory that already contains an
`experiment_record.json`, it updates each `episode_record.json` in-place by populating
the `judge_output` field (currently `null` in the atlas-eval-log schema):

```jsonc
{
  "judge_output": {
    "outcome": "failure",
    "primary_blame": "scaffolding",
    "primary_blame_confidence": "high",
    "other_blames": [],
    "summary": "Agent never called final_step despite reaching a correct patch.",
    "evidence": [{"step": 42, "quote": "diff --git a/..."}],
    "hypothesis": "Adding an explicit reminder to call final_step when a patch is ready would fix this class of failure.",
    "hypothesis_confidence": "high",
    "analysis": "..."
  }
}
```

### CLI

```bash
# Judge a single experiment
uv run python -m cube_harness.analyze.judge path/to/output_dir

# With model override
uv run python -m cube_harness.analyze.judge path/to/output_dir --model claude-sonnet-4-6

# Print aggregated blame counts
uv run python -m cube_harness.analyze.judge path/to/output_dir --summary
```

---

## Downstream goals

1. **Automated improvement loop.** The meta-agent (`meta_agent/`) can consume
   `JudgeOutput.hypothesis` from a batch run as structured input to generate targeted
   agent config changes — closing the eval → analyse → fix loop without human
   transcription.

2. **Benchmark-level diagnostics.** Aggregating `primary_blame` distributions across a
   run identifies systematic weaknesses: "60% of failures on this benchmark are
   `tool_or_env` — fix the cube before tuning the agent."

3. **Cross-run hypothesis tracking.** If `hypothesis` and `hypothesis_confidence` are
   persisted and linked to the experiment that tested the fix, the improvement is
   quantifiable: `H: adding anti-loop instruction → +6pp` becomes a stored,
   reproducible record.

---

## Alternatives considered

**Human annotation pipeline.** Accurate but doesn't scale; two annotators disagree ~20%
of the time on failure attribution. LLM judgement at scale is less accurate per-instance
but provides consistent, reproducible signals across runs.

**Rule-based classifiers.** Too brittle: failure mode boundaries are semantic, not
syntactic. "Agent looped" can be `model_capability` (couldn't find the fix) or
`scaffolding` (no anti-loop instruction) depending on context.

**Separate judge per benchmark.** Benchmark-specific prompts would be more accurate for
known tasks but require ongoing maintenance and don't generalize. The taxonomy is designed
to be benchmark-agnostic; benchmark-specific context is injected via `codebase_files` and
`task_description` rather than hardcoded.

**Embedding-based clustering.** Useful as a complement for large-scale pattern discovery,
but doesn't produce the structured blame + hypothesis needed for the improvement loop.

---

## Open questions

1. How many related trajectories to include in the contrastive prompt — and at what token
   cost — before the judge starts hallucinating?
2. Should `judge_experiment()` write a single aggregated `experiment_judge_summary.json`,
   or only per-episode files? (Proposed: both.)
3. Judge calibration: should `Confidence.low` be treated as `none` for aggregation
   purposes?
