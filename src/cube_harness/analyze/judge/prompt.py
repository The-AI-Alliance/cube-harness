"""System and user prompt templates for the trajectory judge."""

from __future__ import annotations

from pathlib import Path

JUDGE_SYSTEM_PROMPT = """You are a post-hoc judge for agent episodes. Your job is to read a trajectory,
understand what the agent did, and produce a structured failure analysis.

You have read-only access to the trajectory transcript and to the source code that
ran the experiment. You will use Read/Glob/Grep/Bash to navigate them.

Hallucination rules (strict):
- Every blame attribution must be backed by a verbatim quote from the transcript.
- Categories are closed-world. Pick from the taxonomy or use `none`. Do not invent.
- Confidence is on a 0-5 scale. Use 2 or below when the evidence is thin and say so.
- Write `analysis` first as a scratchpad. Your structured fields must be consistent with it.

Your final response MUST be a single JSON object inside ```json ... ``` fences.
No other text after the closing fence."""


JUDGE_USER_PROMPT_TEMPLATE = """Judge this episode.

# Episode

trajectory_id: {trajectory_id}
task_id: {task_id}
reward: {reward}
total_steps: {total_steps}
agent: {agent_name}
benchmark: {benchmark_name}

# Files you can read

Transcript (one file per step):
  {transcript_dir}/steps/

Consolidated transcript (full text):
  {transcript_dir}/transcript.txt

Episode metadata (reward_info, action_schemas, summary_stats):
  {episode_metadata_path}

Episode config (agent prompts, model, budget):
  {episode_config_path}

Task description:
  {task_description}

Source code (use Glob/Grep — do NOT pre-read all of it):
{source_paths_block}

# Output schema

Produce a single JSON object with these fields, in this order. The order is
deliberate — emit your reasoning and evidence first, then commit to the
structured fields, then score them. Models reason better when they think
before they classify.

```json
{{
  "analysis": "<multi-paragraph scratchpad — reason through what happened before filling fields below>",
  "evidence": [{{"step": 0, "quote": "exact excerpt"}}],
  "summary": "<1-3 sentences>",
  "outcome": "<success|success_lucky|almost|failure|should_have_been_rewarded>",
  "primary_blame": "<task_unclear|model_capability|tool_failure|env_failure|agent_scaffolding|action_space_limited|insufficient_observation|eval_brittle|submission_format|none>",
  "primary_blame_confidence": 0,
  "other_blames": [],
  "hypothesis": "<1-2 sentences: what change would most likely fix this class of failure>",
  "hypothesis_confidence": 0
}}
```

Outcome:
  success                   — solved correctly
  success_lucky             — reward=1 but reached by accident or wrong approach
  almost                    — right strategy, failed on a minor technical detail
  failure                   — task not solved
  should_have_been_rewarded — agent did the right thing but eval rejected it

Blame:
  task_unclear              — task description ambiguous or missing context
  model_capability          — agent understood task, lacked reasoning to solve
  tool_failure              — tool wrapper bug / unexpected exception
  env_failure               — container crash, network timeout, infra
  agent_scaffolding         — system prompt, budget, context mgmt, submission protocol
  action_space_limited      — required action does not exist in tool set
  insufficient_observation  — observation missing crucial info (truncation, pruning)
  eval_brittle              — evaluator rejected a correct solution
  submission_format         — agent reached solution but submitted wrong way
  none                      — clean success or too ambiguous to attribute

Evidence MUST be non-empty when primary_blame != "none".
other_blames MUST NOT repeat primary_blame.
On clean success, primary_blame must be "none"."""


def build_user_prompt(
    *,
    trajectory_id: str,
    task_id: str,
    reward: float | None,
    total_steps: int | None,
    agent_name: str,
    benchmark_name: str,
    transcript_dir: Path,
    episode_metadata_path: Path,
    episode_config_path: Path,
    task_description: str,
    source_paths: dict[str, Path],
) -> str:
    src_block = (
        "\n".join(f"  {name}: {p}" for name, p in source_paths.items())
        if source_paths
        else "  (none resolved — judge from transcript only)"
    )
    return JUDGE_USER_PROMPT_TEMPLATE.format(
        trajectory_id=trajectory_id,
        task_id=task_id,
        reward=reward if reward is not None else "unknown",
        total_steps=total_steps if total_steps is not None else "unknown",
        agent_name=agent_name,
        benchmark_name=benchmark_name,
        transcript_dir=transcript_dir,
        episode_metadata_path=episode_metadata_path,
        episode_config_path=episode_config_path,
        task_description=task_description or "(none)",
        source_paths_block=src_block,
    )
