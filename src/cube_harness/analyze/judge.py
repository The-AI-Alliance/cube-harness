"""Trajectory judge — post-hoc LLM analysis of cube-harness episodes.

Reads a completed experiment directory, decompresses the `*.msgpack.zst` step files
into a readable transcript, and invokes a Claude Code agent (via the `claude-agent-sdk`
Python API) to produce a structured `JudgeOutput` for each selected episode.

The judge agent is given:
  - The decoded transcript directory (one .txt per step).
  - Episode metadata: reward, task_id, agent config, total steps.
  - Read access to the cube package source (the benchmark) and the cube-harness source
    (the agent scaffolding) — resolved via `importlib.util.find_spec`.

Results are written into each episode's `episode_record.json` (sibling fields
`judge_output` and `judge_metadata`) and aggregated into
`<experiment_dir>/experiment_judge_summary.json`.

CLI: `python -m cube_harness.analyze.judge <experiment_dir> [options]` or `ch-judge`.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import difflib
import importlib.util
import json
import logging
import random
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import msgpack
import zstandard
from pydantic import ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from cube_harness.core import Trajectory
from cube_harness.eval_log import (
    EPISODE_RECORD_FILENAME,
    JUDGE_SCHEMA_VERSION,
    EpisodeRecord,
    InterventionCategory,
    JudgeConsensus,
    JudgeMetadata,
    JudgeOutput,
)
from cube_harness.experiment import Experiment

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_SAMPLE_FRACTION = 0.10
DEFAULT_JUDGE_MAX_TURNS = 40
DEFAULT_PRE_JUDGE_MAX_TURNS = 80
DEFAULT_JUDGE_MAX_BUDGET_USD = 1.0
PRE_JUDGE_METADATA_FILENAME = "pre_judge_metadata.json"
EXPERIMENT_JUDGE_SUMMARY_FILENAME = "experiment_judge_summary.json"
EXPERIMENT_JUDGE_REPORT_FILENAME = "experiment_judge_report.csv"

# Tools the judge needs: read transcript files, grep through cube/agent source,
# inspect screenshots if any. No write/edit — the judge only produces a JSON answer.
JUDGE_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Bash"]

# Pre-judge also needs Write to save judge_context.md.
PRE_JUDGE_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Bash", "Write"]
PRE_JUDGE_CONTEXT_FILENAME = "judge_context.md"
DEFAULT_PRE_JUDGE_MODEL = "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Step decoding (msgpack.zst → readable transcript)
# ---------------------------------------------------------------------------


def _decompress(path: Path) -> dict[str, Any]:
    """Read and decompress a single step file (.msgpack.zst) into a plain dict."""
    with open(path, "rb") as f:
        data = f.read()
    dctx = zstandard.ZstdDecompressor()
    return msgpack.unpackb(dctx.decompress(data), raw=False)


def _format_obs(step_idx: int, raw: dict[str, Any]) -> str:
    """Render one observation step as a readable text block."""
    output = raw.get("output", raw)
    obs = output.get("obs", output) if isinstance(output, dict) else {}
    contents = obs.get("contents", []) if isinstance(obs, dict) else []
    reward = obs.get("reward") if isinstance(obs, dict) else None
    done = obs.get("done") if isinstance(obs, dict) else None

    lines = [f"### Step {step_idx:03d} OBS"]
    if reward is not None:
        lines.append(f"reward={reward}  done={done}")
    for c in contents:
        if not isinstance(c, dict):
            lines.append(str(c))
            continue
        tool_call_id = c.get("tool_call_id")
        data = c.get("data", "")
        if isinstance(data, bytes):
            data = f"<binary {len(data)} bytes>"
        if tool_call_id:
            lines.append(f"[tool_call_id={tool_call_id}]")
        lines.append(str(data))
    return "\n".join(lines).rstrip() + "\n"


def _format_act(step_idx: int, raw: dict[str, Any]) -> str:
    """Render one action step as a readable text block."""
    output = raw.get("output", raw)
    actions = output.get("actions", []) if isinstance(output, dict) else []
    llm_calls = output.get("llm_calls", []) if isinstance(output, dict) else []
    error = output.get("error") if isinstance(output, dict) else None

    lines = [f"### Step {step_idx:03d} ACT"]
    for call in llm_calls:
        thinking = call.get("thinking") if isinstance(call, dict) else None
        if thinking:
            lines.append("THINKING:")
            lines.append(str(thinking))
    for action in actions:
        if not isinstance(action, dict):
            lines.append(f"ACTION: {action}")
            continue
        name = action.get("name", "?")
        args = action.get("arguments", {})
        try:
            args_repr = json.dumps(args, default=str, indent=2)
        except Exception:
            args_repr = str(args)
        lines.append(f"ACTION {name}:\n{args_repr}")
    if error:
        lines.append(f"ERROR: {error}")
    return "\n".join(lines).rstrip() + "\n"


def extract_transcript(episode_dir: Path, out_dir: Path) -> Path:
    """Decompress every step file in `<episode_dir>/steps/` into readable .txt files.

    Writes one file per step into `<out_dir>/steps/NNN_(obs|act).txt` and a
    lightweight `episode_summary.txt` with the first obs + last 5 steps.
    Returns `out_dir`.
    """
    steps_dir = episode_dir / "steps"
    if not steps_dir.exists():
        raise FileNotFoundError(f"No steps/ directory in {episode_dir}")

    out_steps = out_dir / "steps"
    out_steps.mkdir(parents=True, exist_ok=True)

    all_steps: list[tuple[int, str, str]] = []  # (idx, kind, text)
    for step_file in sorted(steps_dir.iterdir()):
        if not step_file.name.endswith(".msgpack.zst"):
            continue
        try:
            step_idx = int(step_file.name[:3])
        except ValueError:
            continue
        try:
            raw = _decompress(step_file)
        except Exception as e:
            logger.warning("Failed to decompress %s: %s", step_file, e)
            continue
        if "_obs" in step_file.name:
            text = _format_obs(step_idx, raw)
            kind = "obs"
        elif "_act" in step_file.name:
            text = _format_act(step_idx, raw)
            kind = "act"
        else:
            continue
        (out_steps / f"{step_idx:03d}_{kind}.txt").write_text(text)
        all_steps.append((step_idx, kind, text))

    _write_episode_summary(out_dir, all_steps)
    return out_dir


def _write_episode_summary(out_dir: Path, all_steps: list[tuple[int, str, str]]) -> None:
    """Write a content-aware summary: task + error steps + last K agent steps."""
    if not all_steps:
        (out_dir / "episode_summary.txt").write_text("# Episode summary\n\n(empty episode)\n")
        return

    # Sort by step idx (defensive — they should already be sorted)
    by_idx = sorted(all_steps, key=lambda t: t[0])

    # First obs
    first_obs_text = next((t for idx, k, t in by_idx if k == "obs"), None)

    # Error steps (heuristic: contains the literal "ERROR:" line we emit)
    error_steps = [(idx, k, t) for idx, k, t in by_idx if "ERROR:" in t]

    # Last K agent steps (act step + the obs that follows it).
    K = 3
    act_steps = [t for t in by_idx if t[1] == "act"]
    last_k_acts = act_steps[-K:] if act_steps else []
    last_k_idxs: set[int] = set()
    for idx, _, _ in last_k_acts:
        last_k_idxs.add(idx)
        last_k_idxs.add(idx + 1)  # the obs that follows

    # Collect, dedupe by step_idx, ordered
    selected_idxs: set[int] = set(last_k_idxs) | {idx for idx, _, _ in error_steps}
    selected = [(idx, k, t) for idx, k, t in by_idx if idx in selected_idxs]

    parts: list[str] = ["# Episode summary\n"]
    if first_obs_text:
        parts.append("## Task description (step 0 obs, truncated)\n")
        parts.append(first_obs_text[:2000])
        parts.append("\n...\n")
    if error_steps:
        parts.append(f"\n## Error steps ({len(error_steps)} total)\n")
    if last_k_acts:
        parts.append(f"\n## Last {K} agent steps (act + following obs)\n")

    body = "\n".join(t for _, _, t in selected)
    # Cap total at 30 KB
    if len(body) > 30_000:
        body = body[:30_000] + "\n[... summary truncated at 30 KB ...]\n"
    parts.append(body)

    (out_dir / "episode_summary.txt").write_text("\n".join(parts))


# ---------------------------------------------------------------------------
# Episode discovery and selection
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRef:
    trajectory_id: str
    episode_dir: Path
    record_path: Path
    record: EpisodeRecord | None  # None if record file missing


def _load_episode_record(record_path: Path) -> EpisodeRecord | None:
    if not record_path.exists():
        return None
    try:
        return EpisodeRecord.model_validate_json(record_path.read_text())
    except (ValidationError, json.JSONDecodeError) as e:
        logger.warning("Could not parse %s: %s", record_path, e)
        return None


def discover_episodes(experiment_dir: Path) -> list[EpisodeRef]:
    """Return all episode directories under `<experiment_dir>/episodes/`."""
    episodes_dir = experiment_dir / "episodes"
    if not episodes_dir.exists():
        # Maybe the user passed a single episode directory.
        if (experiment_dir / "steps").exists():
            return [
                EpisodeRef(
                    trajectory_id=experiment_dir.name,
                    episode_dir=experiment_dir,
                    record_path=experiment_dir / EPISODE_RECORD_FILENAME,
                    record=_load_episode_record(experiment_dir / EPISODE_RECORD_FILENAME),
                )
            ]
        raise FileNotFoundError(f"No 'episodes/' under {experiment_dir} and no 'steps/' inside it")

    refs: list[EpisodeRef] = []
    for ep_dir in sorted(episodes_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        record_path = ep_dir / EPISODE_RECORD_FILENAME
        refs.append(
            EpisodeRef(
                trajectory_id=ep_dir.name,
                episode_dir=ep_dir,
                record_path=record_path,
                record=_load_episode_record(record_path),
            )
        )
    return refs


_IN_FLIGHT_STATUSES: frozenset[str] = frozenset({"QUEUED", "RUNNING"})


def _is_in_flight(episode_dir: Path) -> bool:
    """Return True if the episode's status.json shows QUEUED or RUNNING."""
    status_path = episode_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        return json.loads(status_path.read_text()).get("status") in _IN_FLIGHT_STATUSES
    except (json.JSONDecodeError, OSError):
        return False


def select_episodes(
    refs: list[EpisodeRef],
    *,
    ids: list[str] | None = None,
    sample: float | None = None,
    n: int | None = None,
    failures_only: bool = False,
    overwrite: bool = False,
    seed: int | None = None,
) -> list[EpisodeRef]:
    """Filter and sample episode refs.

    `ids` is an explicit override: when set, the named episodes are returned
    verbatim regardless of `failures_only` / already-judged / sampling — the user
    typed exactly what they want.

    Otherwise: in-flight (QUEUED/RUNNING) → failures-only → already-judged
    (unless `overwrite`) → sample/n. In-flight episodes are always excluded to
    avoid judging incomplete trajectories.
    """
    if ids:
        wanted = set(ids)
        return [r for r in refs if r.trajectory_id in wanted or r.trajectory_id.split("_ep")[0] in wanted]

    # Always exclude episodes that haven't finished yet.
    in_flight = [r for r in refs if _is_in_flight(r.episode_dir)]
    if in_flight:
        logger.info("Skipping %d in-flight episode(s) (QUEUED/RUNNING).", len(in_flight))
    pool = [r for r in refs if not _is_in_flight(r.episode_dir)]

    if failures_only:
        pool = [r for r in pool if (r.record is not None and not r.record.is_correct)]

    if not overwrite:
        pool = [r for r in pool if not (r.record is not None and r.record.judge_output is not None)]

    rng = random.Random(seed)
    if n is not None:
        if n >= len(pool):
            return pool
        return rng.sample(pool, n)
    if sample is not None:
        if sample >= 1.0:
            return pool
        k = max(1, int(round(len(pool) * sample))) if pool else 0
        return rng.sample(pool, k)
    return pool


# ---------------------------------------------------------------------------
# Stratified pre-judge sampling helpers
# ---------------------------------------------------------------------------


def _task_family(ref: "EpisodeRef") -> str:
    """Best-effort prefix from a trajectory_id: 'django__django-14500_ep157' → 'django'."""
    base = ref.trajectory_id.split("_ep")[0]
    if "__" in base:
        return base.split("__", 1)[0]
    if "_" in base:
        return base.split("_", 1)[0]
    return base


def _stratified_sample(refs: list["EpisodeRef"], correct: bool, max_n: int) -> list["EpisodeRef"]:
    """Up to `max_n` refs, each from a different task family."""
    pool = [r for r in refs if r.record is not None and r.record.is_correct == correct]
    seen_families: set[str] = set()
    out: list[EpisodeRef] = []
    for r in pool:
        fam = _task_family(r)
        if fam in seen_families:
            continue
        seen_families.add(fam)
        out.append(r)
        if len(out) >= max_n:
            break
    return out


# ---------------------------------------------------------------------------
# Source-path resolution
# ---------------------------------------------------------------------------


def _resolve_module_root(dotted_name: str) -> Path | None:
    """Map a `_type` value (e.g. 'swebench_verified_cube.benchmark.X') to a directory."""
    top = dotted_name.split(".")[0]
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    origin = Path(spec.origin)
    return origin.parent if origin.name == "__init__.py" else origin.parent


def collect_source_paths(view: "_ExperimentView") -> dict[str, Path]:
    """Resolve the on-disk source paths the judge should be able to grep.

    Returns a name→path map (only paths that exist). Includes the cube package
    referenced by the benchmark, the agent package, and the cube-standard /
    cube-harness installs.
    """
    paths: dict[str, Path] = {}

    def _add(name: str, dotted: str | None) -> None:
        if not dotted:
            return
        root = _resolve_module_root(dotted)
        if root is not None and root.exists():
            paths[name] = root

    _add("cube_package", view.benchmark_dotted)
    _add("agent_package", view.agent_dotted)
    _add("infra_package", view.infra_dotted)
    _add("cube_harness", "cube_harness.eval_log")
    _add("cube_standard", "cube.core")
    return paths


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


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
{context_section}
# Episode

trajectory_id: {trajectory_id}
task_id: {task_id}
reward: {reward}
total_steps: {total_steps}
agent: {agent_name}
benchmark: {benchmark_name}

# Files you can read

Episode summary (task description + last 5 steps — start here):
  {transcript_dir}/episode_summary.txt

Step files (one per step, targeted access):
  {transcript_dir}/steps/NNN_obs.txt  — environment observation / tool result
  {transcript_dir}/steps/NNN_act.txt  — agent action (tool call + thinking)
  List with: ls {transcript_dir}/steps/ | tail -20

Episode metadata (reward and evaluation output, action schemas, usage stats):
  {episode_metadata_path}

Episode config (agent prompts, model, budget, task_config):
  {episode_config_path}

Task description:
  {task_description}

Source code (use Glob/Grep, consult experiment context for specific files):
{source_paths_block}

# Output schema

Produce a single JSON object with these fields, in this order:

```json
{{
  "analysis": "<multi-paragraph scratchpad — reason through what happened before filling fields below>",
  "outcome": "<success|success_lucky|almost|failure|should_have_been_rewarded>",
  "summary": "<1-3 sentences>",
  "primary_blame": "<task_unclear|model_capability|tool_failure|env_failure|agent_scaffolding|action_space_limited|insufficient_observation|eval_brittle|submission_format|none>",
  "primary_blame_confidence": 0,
  "other_blames": [],
  "evidence": [{{"step": 0, "quote": "exact excerpt"}}],
  "hypothesis": "<1-2 sentences: what change would most likely fix this class of failure>",
  "hypothesis_confidence": 0,
  "intervention": "<scaffold_change|prompt_change|eval_fix|infra_fix|model_upgrade|tool_change|task_clarification|none>",
  "intervention_confidence": 0
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

Intervention vs blame:
  primary_blame describes WHAT went wrong (cause).
  intervention describes WHERE the fix would land.
  These often differ. Example: agent identifies the bug but loops without
  editing → primary_blame=model_capability, intervention=scaffold_change
  (add loop detection).

Intervention categories:
  scaffold_change     — loop detection, budget logic, context management, prompt
  prompt_change       — system prompt or instruction text
  eval_fix            — benchmark evaluator is wrong or brittle
  infra_fix           — container, network, dependency issue
  model_upgrade       — stronger model would solve it
  tool_change         — add/remove/fix a tool
  task_clarification  — task description ambiguous
  none                — no intervention needed (clean success)

Evidence MUST be non-empty when primary_blame != "none".
other_blames MUST NOT repeat primary_blame.
On clean success, primary_blame must be "none" and intervention must be "none"."""


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
    context_path: Path | None = None,
) -> str:
    src_block = (
        "\n".join(f"  {name}: {p}" for name, p in source_paths.items())
        if source_paths
        else "  (none resolved — judge from transcript only)"
    )
    if context_path is not None and context_path.exists():
        context_text = context_path.read_text()
        context_section = f"\n# Experiment context\n\n{context_text}\n---\n"
    else:
        context_section = ""
    return JUDGE_USER_PROMPT_TEMPLATE.format(
        context_section=context_section,
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


# ---------------------------------------------------------------------------
# Claude Code SDK invocation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pre-judge: experiment-level context document
# ---------------------------------------------------------------------------


def _validate_context(context_path: Path) -> bool:
    """Return True if cited paths in judge_context.md still exist.

    Scans for absolute paths (lines containing '/' patterns that look like file
    references) and checks that a reasonable fraction still resolve. Returns
    True (= valid) when ≥80% of cited absolute paths exist.
    """
    text = context_path.read_text()
    # Heuristic: extract paths after backticks or in `path: /...` patterns.
    paths = re.findall(r"`(/[^\s`]+)`", text)
    paths += re.findall(
        r"^\s+([^/\s][^\s:]*\.(?:py|json|md|txt|yaml))",
        text,
        flags=re.MULTILINE,
    )
    if not paths:
        return True  # no cited paths to validate
    existing = sum(1 for p in paths if Path(p).exists())
    ratio = existing / len(paths)
    if ratio < 0.8:
        logger.warning(
            "judge_context.md cites %d paths, only %d (%.0f%%) exist on disk; will regenerate.",
            len(paths),
            existing,
            ratio * 100,
        )
        return False
    return True


PRE_JUDGE_SYSTEM_PROMPT = """You are an experiment analyst. Your job is to read an agent \
experiment's configuration and source code, then write a concise reference document that \
downstream episode judges will read before analyzing individual trajectory failures.

You have access to Read/Glob/Grep/Bash/Write tools. Use them to explore, then call Write \
to save the finished document.

Guidelines:
- Be benchmark-agnostic: the document must help judges reason about any failure category.
- Be concise: target 150–250 lines of Markdown.
- Cite exact file paths for every claim so judges can verify or dig deeper.
- Flag suspicious patterns proactively (hardcoded timeouts, fragile evaluators, unusual configs).
- Do NOT assess individual episode quality — that is the episode judge's job.
- When done exploring, call Write to save the document, then stop."""

PRE_JUDGE_USER_PROMPT_TEMPLATE = """Build a reference context document for the downstream \
episode judges in this experiment.

Save the finished document to:
  {output_path}

# Experiment

Name:       {exp_name}
Agent:      {agent_name}
Benchmark:  {benchmark_name}
Infra:      {infra_name}
Episodes:   {n_episodes} total  |  pass rate: {pass_rate}

Experiment config:
  {experiment_config_path}

# Source packages (explore — do NOT read everything)

{source_paths_block}

# Sample episode directories (for format reference)

{sample_dirs_block}

# Exploration checklist

Work through these in order:

1. **Experiment config** — read `experiment_config.json` to confirm agent/benchmark/infra \
types and key parameters (model used, max_steps, tool config class).

2. **Agent** — find the agent class. Summarise: LLM prompt structure, context management \
strategy, how the agent decides to stop / submit, any hard budget limits.

3. **Benchmark & evaluation** — find the `evaluate()` or reward function in the benchmark \
source. Provide the exact file path and relevant line numbers. Summarise: what the task \
asks the agent to do, the precise condition for reward=1 vs reward=0, and what the \
golden target is (ground-truth patch, test suite, expected output, etc.). \
Then read `episode.metadata.json` from one of the sample episodes to discover what fields \
`reward_info` actually contains — document those field names and what they mean so \
episode judges know exactly where to find evaluation output without having to guess.

4. **Tools & action space** — find the tool config class and its implementation. List \
available tool names; note output truncation, missing capabilities, or known failure modes.

5. **Infrastructure** — note container reset behaviour, timeouts, retry caps, resource limits.

6. **Episode artifact format** — read one sample episode's \
`_judge_transcript/episode_summary.txt` (task description + last 5 steps) so judges \
know what obs/act blocks look like. Note: individual steps are at \
`_judge_transcript/steps/NNN_obs.txt` / `NNN_act.txt`; `ls steps/ | tail -20` shows \
the final steps. Document any fields in `reward_info` that contain evaluation output \
(e.g. test run logs, error messages) that judges should read to understand why the agent \
failed or succeeded.

7. **Suspicious patterns** — flag anything that looks fragile, hardcoded, or likely to \
cause systematic failures across many episodes.

Write the document to {output_path} once done."""


def _build_pre_judge_user_prompt(
    output_path: Path,
    experiment_dir: Path,
    view: "_ExperimentView",
    source_paths: dict[str, Path],
    sample_refs: list["EpisodeRef"],
) -> str:
    src_block = "\n".join(f"  {name}: {p}" for name, p in source_paths.items()) if source_paths else "  (none resolved)"
    sample_block = "\n".join(f"  {ref.episode_dir}" for ref in sample_refs) if sample_refs else "  (none available)"
    summary_path = experiment_dir / "experiment_summary.json"
    pass_rate = "unknown"
    n_episodes = "unknown"
    if summary_path.exists():
        try:
            s = json.loads(summary_path.read_text())
            n_done = s.get("n_completed", 0) + s.get("n_failed", 0) or s.get("n_episodes", 0)
            n_correct = s.get("n_correct", s.get("n_success", -1))
            if n_correct < 0 and n_done:
                # Fallback: derive from avg_reward or total_reward
                avg = s.get("avg_reward", 0.0)
                total = s.get("total_reward")
                n_correct = int(round(total)) if total is not None else int(round(avg * n_done))
            n_episodes = str(n_done) if n_done else "unknown"
            if n_done:
                pass_rate = f"{n_correct / n_done:.1%} ({n_correct}/{n_done})"
        except Exception:
            pass
    return PRE_JUDGE_USER_PROMPT_TEMPLATE.format(
        output_path=output_path,
        exp_name=view.raw.get("name", experiment_dir.name),
        agent_name=view.agent_dotted,
        benchmark_name=view.benchmark_dotted,
        infra_name=view.infra_dotted or "unknown",
        n_episodes=n_episodes,
        pass_rate=pass_rate,
        experiment_config_path=experiment_dir / "experiment_config.json",
        source_paths_block=src_block,
        sample_dirs_block=sample_block,
    )


async def _run_pre_judge_impl(
    experiment_dir: Path,
    view: "_ExperimentView",
    source_paths: dict[str, Path],
    sample_refs: list["EpisodeRef"],
    model: str,
    verbose: bool,
    max_turns: int | None = DEFAULT_PRE_JUDGE_MAX_TURNS,
) -> tuple[Path, _SDKResult]:
    """Run the pre-judge agent and return (context_path, sdk_result)."""
    output_path = experiment_dir / PRE_JUDGE_CONTEXT_FILENAME
    user_prompt = _build_pre_judge_user_prompt(
        output_path=output_path,
        experiment_dir=experiment_dir,
        view=view,
        source_paths=source_paths,
        sample_refs=sample_refs,
    )
    logger.info("Running pre-judge for %s → %s", experiment_dir.name, output_path.name)
    result = await _run_claude_code(
        system_prompt=PRE_JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        cwd=experiment_dir,
        additional_dirs=list(source_paths.values()),
        model=model,
        verbose=verbose,
        trace_mode="off",
        allowed_tools=PRE_JUDGE_ALLOWED_TOOLS,
        max_turns=max_turns,
    )
    return output_path, result


def run_pre_judge(
    experiment_dir: Path,
    *,
    model: str = DEFAULT_PRE_JUDGE_MODEL,
    verbose: bool = False,
    overwrite: bool = False,
    max_turns: int | None = DEFAULT_PRE_JUDGE_MAX_TURNS,
) -> Path | None:
    """Build (or reuse) the experiment-level judge_context.md.

    Returns the path if the file exists after the call, or None if the run failed
    to produce it.  Skips the agent call if the file already exists and `overwrite`
    is False.

    Writes `pre_judge_metadata.json` alongside `judge_context.md` so cost and
    token usage are captured for the Opus pre-judge call.
    """
    experiment_dir = Path(experiment_dir).resolve()
    output_path = experiment_dir / PRE_JUDGE_CONTEXT_FILENAME
    if output_path.exists() and not overwrite:
        if _validate_context(output_path):
            logger.info("Pre-judge context already exists at %s — reusing.", output_path)
            return output_path
        logger.info("Pre-judge context exists but cited paths are stale; regenerating.")

    config_path = experiment_dir / "experiment_config.json"
    view = _load_experiment_view(config_path)
    source_paths = collect_source_paths(view)

    episodes_dir = experiment_dir / "episodes"
    refs: list[EpisodeRef] = []
    if episodes_dir.exists():
        all_refs = discover_episodes(experiment_dir)
        # Stratified sample: up to 2 from different families, failed + succeeded.
        sample = _stratified_sample(all_refs, correct=False, max_n=2) + _stratified_sample(
            all_refs, correct=True, max_n=2
        )
        if not sample:
            sample = all_refs[:2]
        refs = sample

    _, result = asyncio.run(_run_pre_judge_impl(experiment_dir, view, source_paths, refs, model, verbose, max_turns))
    if output_path.exists():
        meta = JudgeMetadata(
            model=model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
            duration_s=result.duration_s,
            timestamp=time.time(),
            judge_schema_version=JUDGE_SCHEMA_VERSION,
        )
        (experiment_dir / PRE_JUDGE_METADATA_FILENAME).write_text(meta.model_dump_json(indent=2))
        logger.info(
            "Pre-judge cost: $%.4f  tokens: %dp + %dc  duration: %.0fs",
            result.cost_usd,
            result.prompt_tokens,
            result.completion_tokens,
            result.duration_s,
        )
    return output_path if output_path.exists() else None


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_block(text: str) -> dict[str, Any]:
    """Extract a JSON object from the judge's final assistant message."""
    m = _JSON_FENCE_RE.search(text)
    candidate = m.group(1) if m else None
    if candidate is None:
        # Fall back to first {...} block at top level.
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in judge output")
        candidate = text[start:]
    # Try strict parse first, then scan backwards through all } positions.
    # A single rfind isn't enough when the JSON is truncated mid-string and
    # a stray } appears inside a string value before the real closing brace.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    end = len(candidate)
    while True:
        pos = candidate.rfind("}", 0, end)
        if pos == -1:
            raise ValueError("No valid JSON object found in judge output")
        try:
            return json.loads(candidate[: pos + 1])
        except json.JSONDecodeError:
            end = pos


TraceMode = Literal["actions", "full", "off"]


@dataclass
class _SDKResult:
    output_text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    duration_s: float
    actions: list[dict[str, Any]] = field(default_factory=list)


def _summarise_tool_input(name: str, raw_input: dict[str, Any]) -> str:
    """One-line summary of a Claude Code tool call argument set."""
    if not isinstance(raw_input, dict):
        return str(raw_input)[:80]
    if name == "Bash":
        cmd = str(raw_input.get("command", ""))
        return cmd if len(cmd) <= 100 else cmd[:97] + "..."
    if name == "Read":
        target = raw_input.get("file_path") or raw_input.get("path") or ""
        offset = raw_input.get("offset")
        limit = raw_input.get("limit")
        suffix = f" (offset={offset}, limit={limit})" if offset or limit else ""
        return f"{target}{suffix}"
    if name == "Grep":
        pattern = raw_input.get("pattern", "")
        path = raw_input.get("path") or raw_input.get("glob") or ""
        return f"{pattern!r} in {path}" if path else repr(pattern)
    if name == "Glob":
        return str(raw_input.get("pattern", ""))
    # Fallback: compact JSON, truncated.
    try:
        s = json.dumps(raw_input, default=str)
    except Exception:
        s = str(raw_input)
    return s if len(s) <= 100 else s[:97] + "..."


def _is_retryable_sdk_error(exc: BaseException) -> bool:
    """True for transient API errors (overload / rate-limit) that are worth retrying."""
    msg = str(exc).lower()
    return any(tok in msg for tok in ("529", "overload", "rate limit", "too many requests", "502", "503"))


@retry(
    retry=retry_if_exception(_is_retryable_sdk_error),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(4),
    reraise=True,
    before_sleep=lambda rs: logger.warning(
        "Transient SDK error (attempt %d/4); retrying in %.0fs: %s",
        rs.attempt_number,
        rs.next_action.sleep,  # type: ignore[union-attr]
        rs.outcome.exception(),
    ),
)
async def _run_claude_code(
    *,
    system_prompt: str,
    user_prompt: str,
    cwd: Path,
    additional_dirs: list[Path],
    model: str,
    verbose: bool = False,
    trace_mode: TraceMode = "actions",
    allowed_tools: list[str] | None = None,
    max_turns: int | None = None,
    max_budget_usd: float | None = None,
) -> _SDKResult:
    """Invoke Claude Code via the SDK and return the assistant text + usage.

    Retries up to 4 attempts with exponential backoff (10s → 120s) on transient
    API errors (529 Overloaded, 503, rate-limit). Non-retryable errors propagate
    immediately.

    When `verbose=True`, stream a one-line summary of each tool call and assistant
    text chunk to stderr as they arrive — useful to see what the judge is doing
    without waiting for the final JSON.
    """
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            query,
        )
    except ImportError as e:
        raise RuntimeError("claude-agent-sdk not installed. Run: pip install 'cube-harness[judge]'") from e

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=allowed_tools if allowed_tools is not None else JUDGE_ALLOWED_TOOLS,
        permission_mode="bypassPermissions",
        cwd=str(cwd),
        add_dirs=[str(p) for p in additional_dirs],
        model=model,
        include_partial_messages=False,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
    )
    if verbose:
        logger.info("Running judge with model %s and options: %r", model, options)

    final_text: list[str] = []
    collected_actions: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    cost_usd = 0.0
    duration_ms = 0
    start = time.time()

    def _emit(line: str) -> None:
        # Verbose progress goes to stderr so stdout stays parseable for `--summary`.
        print(line, file=sys.stderr, flush=True)

    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    final_text.append(block.text)
                    if verbose and block.text.strip():
                        first_line = block.text.strip().splitlines()[0][:140]
                        _emit(f"  · {first_line}")
                elif isinstance(block, ToolUseBlock):
                    if verbose:
                        _emit(f"  > {block.name}({_summarise_tool_input(block.name, block.input)})")
                    if trace_mode == "actions":
                        collected_actions.append(
                            {"tool": block.name, "input": _summarise_tool_input(block.name, block.input)}
                        )
                    elif trace_mode == "full":
                        collected_actions.append(
                            {
                                "tool": block.name,
                                "input": _summarise_tool_input(block.name, block.input),
                                "raw_input": block.input
                                if isinstance(block.input, dict)
                                else {"value": str(block.input)},
                            }
                        )
        elif isinstance(message, ResultMessage):
            usage = getattr(message, "usage", None) or {}
            prompt_tokens = (
                int(usage.get("input_tokens", 0) or 0)
                + int(usage.get("cache_read_input_tokens", 0) or 0)
                + int(usage.get("cache_creation_input_tokens", 0) or 0)
            )
            completion_tokens = int(usage.get("output_tokens", 0) or 0)
            cost_usd = float(getattr(message, "total_cost_usd", 0.0) or 0.0)
            duration_ms = int(getattr(message, "duration_ms", 0) or 0)

    duration_s = duration_ms / 1000.0 if duration_ms else (time.time() - start)
    return _SDKResult(
        output_text="\n".join(final_text),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        duration_s=duration_s,
        actions=collected_actions if trace_mode != "off" else [],
    )


# ---------------------------------------------------------------------------
# Public API: judge_episode / judge_experiment
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


@dataclass
class _ExperimentView:
    """A view of `experiment_config.json` that prefers the typed `Experiment` object
    but falls back to a raw dict when `_type` references can't be imported (e.g.
    the experiment was run with an agent class that no longer exists locally)."""

    experiment: Experiment | None
    raw: dict[str, Any]

    @property
    def agent_dotted(self) -> str:
        if self.experiment is not None:
            t = type(self.experiment.agent_config)
            return f"{t.__module__}.{t.__name__}"
        return self.raw.get("agent_config", {}).get("_type", "unknown")

    @property
    def benchmark_dotted(self) -> str:
        if self.experiment is not None:
            t = type(self.experiment.benchmark_config)
            return f"{t.__module__}.{t.__name__}"
        return self.raw.get("benchmark_config", {}).get("_type", "unknown")

    @property
    def infra_dotted(self) -> str | None:
        if self.experiment is not None and self.experiment.infra is not None:
            t = type(self.experiment.infra)
            return f"{t.__module__}.{t.__name__}"
        return self.raw.get("infra", {}).get("_type") if isinstance(self.raw.get("infra"), dict) else None


def _load_experiment_view(path: Path) -> _ExperimentView:
    """Load experiment_config.json. Try typed Experiment first; fall back to dict.

    Typed load can fail in many ways: a referenced class was renamed
    (`ImportError`), removed (`AttributeError`, when `_type` is `__main__.X` from
    an ad-hoc script), or changed shape (`ValidationError`). Any failure falls
    back to the dict view, which is always good enough for the judge — it only
    needs the `_type` strings and a few well-known fields.
    """
    if not path.exists():
        return _ExperimentView(experiment=None, raw={})
    raw = _read_json(path)
    try:
        return _ExperimentView(experiment=Experiment.load_config(str(path)), raw=raw)
    except Exception as e:
        logger.info(
            "experiment_config.json could not be loaded as Experiment (%s: %s) — "
            "falling back to dict view. Source paths will still be resolved from `_type` strings.",
            type(e).__name__,
            e,
        )
        return _ExperimentView(experiment=None, raw=raw)


def _load_trajectory_meta(path: Path) -> Trajectory | None:
    """Load episode.metadata.json as a Trajectory. The `steps` field will be empty
    (steps live in `steps/*.msgpack.zst`); reward_info/metadata/summary_stats are populated."""
    if not path.exists():
        return None
    try:
        return Trajectory.model_validate_json(path.read_text())
    except (ValidationError, json.JSONDecodeError) as e:
        logger.warning("Could not parse %s as Trajectory: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Evidence verification (R2.1)
# ---------------------------------------------------------------------------

EVIDENCE_FUZZY_THRESHOLD = 0.85


def _normalize_for_match(text: str) -> str:
    """Whitespace-collapsed, case-folded for matching."""
    return " ".join(text.split()).casefold()


def _find_quote_in_text(quote: str, step_text: str) -> tuple[bool, float | None]:
    """Try to locate `quote` in `step_text`. Returns (verified, match_ratio).

    Strategy: exact case-insensitive substring first; if miss, slide a window
    of len(quote) across step_text and take max difflib ratio. Accept if
    ratio >= EVIDENCE_FUZZY_THRESHOLD.
    """
    q_norm = _normalize_for_match(quote)
    t_norm = _normalize_for_match(step_text)
    if not q_norm:
        return False, None
    if q_norm in t_norm:
        return True, None  # exact (after normalization)

    # Sliding window fuzzy match.
    n = len(q_norm)
    best = 0.0
    # Sample windows at quarter-length stride to keep cost bounded.
    stride = max(1, n // 4)
    for start in range(0, max(1, len(t_norm) - n + 1), stride):
        window = t_norm[start : start + n]
        ratio = difflib.SequenceMatcher(None, q_norm, window).ratio()
        if ratio > best:
            best = ratio
        if best >= 0.99:
            break
    if best >= EVIDENCE_FUZZY_THRESHOLD:
        return True, round(best, 3)
    return False, round(best, 3)


def _build_artifact_text(episode_dir: Path) -> str:
    """Concatenate episode artifact files that judges commonly quote from.

    Includes episode.metadata.json (contains reward_info.fail_to_pass_output)
    and episode_record.json. These are the source of evaluator-side quotes
    that cannot appear in agent step files.
    """
    text = ""
    for name in ("episode.metadata.json", "episode_record.json"):
        p = episode_dir / name
        if p.exists():
            text += p.read_text() + "\n"
    return text


def _verify_evidence(judge_output: JudgeOutput, transcript_dir: Path) -> None:
    """Mutate JudgeOutput.evidence in place: set `verified` and `match_ratio`.

    Search order for each quoted item:
    1. The cited step files (NNN_obs.txt / NNN_act.txt) — agent-side content.
    2. Episode artifact files (episode.metadata.json, episode_record.json) —
       evaluator-side content such as fail_to_pass_output that the judge reads
       but the agent never sees in its steps.

    If any evidence item fails both searches, downgrade primary_blame_confidence
    by 1 (clamped at 0). This is a soft signal — it doesn't refuse the judgment.
    """
    if not judge_output.evidence:
        return
    steps_dir = transcript_dir / "steps"
    episode_dir = transcript_dir.parent
    artifact_text = _build_artifact_text(episode_dir)
    n_unverified = 0
    for item in judge_output.evidence:
        # 1. Check agent step files (obs + act for the cited step index).
        step_text = ""
        for suffix in ("obs", "act"):
            f = steps_dir / f"{item.step:03d}_{suffix}.txt"
            if f.exists():
                step_text += f.read_text() + "\n"

        if step_text:
            verified, ratio = _find_quote_in_text(item.quote, step_text)
            if verified:
                item.verified = True
                item.match_ratio = ratio
                continue

        # 2. Fall back to episode artifact files (evaluator output, reward_info, etc.).
        if artifact_text:
            verified, ratio = _find_quote_in_text(item.quote, artifact_text)
            if verified:
                item.verified = True
                item.match_ratio = ratio
                continue

        # Not found in either source.
        item.verified = False
        item.match_ratio = (
            ratio
            if (not step_text and artifact_text)
            else (
                _find_quote_in_text(item.quote, step_text + artifact_text)[1] if (step_text or artifact_text) else None
            )
        )
        n_unverified += 1

    if n_unverified > 0:
        logger.warning(
            "%d/%d evidence items failed verification; downgrading primary_blame_confidence.",
            n_unverified,
            len(judge_output.evidence),
        )
        judge_output.primary_blame_confidence = max(0, judge_output.primary_blame_confidence - 1)


def _validate_invariants(obj: JudgeOutput) -> None:
    """Enforce the invariants from the spec (post-parse, pre-write)."""
    if obj.primary_blame.value != "none" and not obj.evidence:
        raise ValueError("evidence must be non-empty when primary_blame != 'none'")
    if obj.primary_blame in obj.other_blames:
        raise ValueError("other_blames must not repeat primary_blame")
    if obj.outcome.value in ("success", "success_lucky") and obj.primary_blame.value != "none":
        # Soft-correct: tighten to spec rather than fail.
        logger.warning(
            "Judge returned outcome=%s with primary_blame=%s; spec requires 'none'. Coercing.",
            obj.outcome.value,
            obj.primary_blame.value,
        )
        obj.primary_blame = obj.primary_blame.__class__("none")
    # When outcome is success/success_lucky, intervention should be 'none'.
    if obj.outcome.value in ("success", "success_lucky") and obj.intervention.value != "none":
        logger.warning(
            "Judge returned outcome=%s with intervention=%s; coercing to 'none'.",
            obj.outcome.value,
            obj.intervention.value,
        )
        obj.intervention = InterventionCategory.none


async def _judge_episode_impl(
    episode_dir: Path,
    experiment_dir: Path,
    model: str,
    verbose: bool,
    trace_mode: TraceMode = "actions",
    context_path: Path | None = None,
    max_turns: int | None = DEFAULT_JUDGE_MAX_TURNS,
    transcript_dir: Path | None = None,
    max_budget_usd: float | None = DEFAULT_JUDGE_MAX_BUDGET_USD,
) -> tuple[JudgeOutput, JudgeMetadata, list[dict[str, Any]]]:
    """Async core shared by judge_episode (single) and judge_experiment (parallel).

    When `transcript_dir` is provided the caller owns extract/cleanup — this is
    used by `_judge_episode_k_times` to share a single extracted transcript
    across K concurrent judge calls without races.
    """
    owns_transcript = transcript_dir is None
    if transcript_dir is None:
        transcript_dir = episode_dir / "_judge_transcript"
        extract_transcript(episode_dir, transcript_dir)

    metadata_path = episode_dir / "episode.metadata.json"
    config_path = episode_dir / "episode_config.json"
    experiment_config_path = experiment_dir / "experiment_config.json"

    trajectory = _load_trajectory_meta(metadata_path)
    view = _load_experiment_view(experiment_config_path)

    if trajectory is not None:
        task_id = trajectory.metadata.get("task_id") or trajectory.id
        reward = trajectory.reward_info.get("reward")
        total_steps = (trajectory.summary_stats or {}).get("n_agent_steps")
        task_description = trajectory.metadata.get("task_description", "")
    else:
        task_id, reward, total_steps, task_description = "unknown", None, None, ""

    source_paths = collect_source_paths(view)
    user_prompt = build_user_prompt(
        trajectory_id=episode_dir.name,
        task_id=task_id,
        reward=reward,
        total_steps=total_steps,
        agent_name=view.agent_dotted,
        benchmark_name=view.benchmark_dotted,
        transcript_dir=transcript_dir,
        episode_metadata_path=metadata_path,
        episode_config_path=config_path,
        task_description=task_description,
        source_paths=source_paths,
        context_path=context_path,
    )

    logger.info("Judging %s (reward=%s, steps=%s) with %s", episode_dir.name, reward, total_steps, model)

    result = await _run_claude_code(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        cwd=episode_dir,
        additional_dirs=list(source_paths.values()) + [transcript_dir],
        model=model,
        verbose=verbose,
        trace_mode=trace_mode,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
    )

    obj = _extract_json_block(result.output_text)
    judge_output = JudgeOutput.model_validate(obj)
    _validate_invariants(judge_output)
    _verify_evidence(judge_output, transcript_dir)
    if owns_transcript:
        _cleanup_transcript(transcript_dir)

    judge_metadata = JudgeMetadata(
        model=model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        duration_s=result.duration_s,
        timestamp=time.time(),
        judge_schema_version=JUDGE_SCHEMA_VERSION,
    )
    return judge_output, judge_metadata, result.actions


# ---------------------------------------------------------------------------
# Inter-judge consensus (R3.2)
# ---------------------------------------------------------------------------


def _build_consensus(outputs: list[JudgeOutput]) -> JudgeConsensus:
    """Aggregate K JudgeOutputs into a JudgeConsensus via modal voting."""
    n = len(outputs)
    outcome_mode, outcome_count = Counter(o.outcome for o in outputs).most_common(1)[0]
    blame_mode, blame_count = Counter(o.primary_blame for o in outputs).most_common(1)[0]
    intervention_mode = Counter(o.intervention for o in outputs).most_common(1)[0][0]
    return JudgeConsensus(
        n_judges=n,
        outcome=outcome_mode,
        primary_blame=blame_mode,
        intervention=intervention_mode,
        outcome_agreement=outcome_count / n,
        blame_agreement=blame_count / n,
        avg_primary_blame_confidence=sum(o.primary_blame_confidence for o in outputs) / n,
    )


async def _judge_episode_k_times(
    episode_dir: Path,
    experiment_dir: Path,
    model: str,
    verbose: bool,
    trace_mode: TraceMode,
    context_path: Path | None,
    max_turns: int | None,
    n_judges: int,
    max_budget_usd: float | None = DEFAULT_JUDGE_MAX_BUDGET_USD,
) -> tuple[JudgeConsensus, list[JudgeOutput], list[JudgeMetadata], list[list[dict[str, Any]]]]:
    """Run K judge calls on the same episode, return consensus + raw outputs.

    Extracts the transcript once and shares it across all K concurrent judges.
    Each judge reads the same step files; cleanup runs once after all finish.
    """
    shared_transcript = episode_dir / "_judge_transcript"
    extract_transcript(episode_dir, shared_transcript)
    try:
        tasks = [
            _judge_episode_impl(
                episode_dir,
                experiment_dir,
                model,
                verbose,
                trace_mode,
                context_path,
                max_turns,
                transcript_dir=shared_transcript,
                max_budget_usd=max_budget_usd,
            )
            for _ in range(n_judges)
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        _cleanup_transcript(shared_transcript)
    outputs: list[JudgeOutput] = []
    metas: list[JudgeMetadata] = []
    actions: list[list[dict[str, Any]]] = []
    for r in raw_results:
        if isinstance(r, Exception):
            logger.warning("One of K=%d judges failed: %s", n_judges, r)
            continue
        o, m, a = r  # type: ignore[misc]
        outputs.append(o)
        metas.append(m)
        actions.append(a)
    if not outputs:
        raise RuntimeError(f"All {n_judges} judges failed for {episode_dir.name}")
    consensus = _build_consensus(outputs)
    return consensus, outputs, metas, actions


def judge_episode(
    episode_dir: Path,
    *,
    experiment_dir: Path | None = None,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    trace_mode: TraceMode = "actions",
    context_path: Path | None = None,
    max_turns: int | None = None,
    max_budget_usd: float | None = DEFAULT_JUDGE_MAX_BUDGET_USD,
) -> tuple[JudgeOutput, JudgeMetadata]:
    """Run a post-hoc judge on a single episode trajectory directory.

    `experiment_dir` is needed only to locate `experiment_config.json`; if omitted,
    we look one level up from `episode_dir`.
    """
    episode_dir = Path(episode_dir).resolve()
    if experiment_dir is None:
        experiment_dir = episode_dir.parent.parent
    judge_output, judge_metadata, _ = asyncio.run(
        _judge_episode_impl(
            episode_dir,
            Path(experiment_dir).resolve(),
            model,
            verbose,
            trace_mode,
            context_path,
            max_turns,
            max_budget_usd=max_budget_usd,
        )
    )
    return judge_output, judge_metadata


def _cleanup_transcript(transcript_dir: Path) -> None:
    """Delete the unpacked step files after judging to avoid disk accumulation.

    Removes steps/, episode_summary.txt, and the legacy transcript.txt if present.
    Then attempts to remove the directory itself (succeeds only when empty).
    """
    for name in ("steps",):
        d = transcript_dir / name
        if d.exists():
            shutil.rmtree(d)
    for name in ("episode_summary.txt", "transcript.txt"):
        f = transcript_dir / name
        if f.exists():
            f.unlink()
    try:
        transcript_dir.rmdir()  # only removes if now empty
    except OSError:
        pass  # non-empty (e.g. other files present) — leave it


def _archive_versioned(path: Path) -> None:
    """Rename `path` to `<stem>_old_v<N><ext>` if it exists, incrementing N until free."""
    if not path.exists():
        return
    n = 1
    while True:
        candidate = path.parent / f"{path.stem}_old_v{n}{path.suffix}"
        if not candidate.exists():
            path.rename(candidate)
            return
        n += 1


def _persist_judgment(
    ref: EpisodeRef,
    judge_output: JudgeOutput,
    judge_metadata: JudgeMetadata,
    actions: list[dict[str, Any]] | None = None,
    trace_mode: TraceMode = "actions",
    consensus: JudgeConsensus | None = None,
    all_outputs: list[tuple[JudgeOutput, JudgeMetadata]] | None = None,
) -> None:
    """Write `judge_output` and `judge_metadata` into the episode_record.json.

    If the record file does not exist yet (older runs without atlas-eval-log enabled),
    write a sidecar `judge_output.json` so the result is not lost.

    When `actions` is non-empty, writes a `judge_trace.json` sidecar alongside the
    episode record with the judge's tool-call sequence.

    When `consensus` is set (n_judges > 1), also writes `judge_outputs.jsonl` sidecar
    (one JSON line per judge output) and stores `judge_consensus` on the episode record.
    The primary `judge_output` on the record is the first output from the modal-outcome
    group (i.e., the first output matching consensus.outcome).

    Existing `judge_output.json` and `judge_trace.json` sidecars are never clobbered;
    they are renamed to `<stem>_old_v<N><ext>` before the new file is written.
    """
    record_updates: dict[str, Any] = {"judge_output": judge_output, "judge_metadata": judge_metadata}
    if consensus is not None:
        record_updates["judge_consensus"] = consensus

    if ref.record is not None:
        updated = ref.record.model_copy(update=record_updates)
        ref.record_path.write_text(updated.model_dump_json(indent=2))
    else:
        sidecar = ref.episode_dir / "judge_output.json"
        _archive_versioned(sidecar)
        sidecar.write_text(
            json.dumps(
                {
                    "judge_output": judge_output.model_dump(mode="json"),
                    "judge_metadata": judge_metadata.model_dump(mode="json"),
                },
                indent=2,
            )
        )

    if actions:
        trace_path = ref.episode_dir / "judge_trace.json"
        _archive_versioned(trace_path)
        trace_path.write_text(json.dumps({"trace_mode": trace_mode, "actions": actions}, indent=2))

    if all_outputs is not None:
        jsonl_path = ref.episode_dir / "judge_outputs.jsonl"
        lines = [
            json.dumps({"judge_output": o.model_dump(mode="json"), "judge_metadata": m.model_dump(mode="json")})
            for o, m in all_outputs
        ]
        jsonl_path.write_text("\n".join(lines) + "\n")


def judge_experiment(
    experiment_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    ids: list[str] | None = None,
    sample: float | None = None,
    n: int | None = None,
    failures_only: bool = False,
    overwrite: bool = False,
    seed: int | None = None,
    verbose: bool = False,
    n_parallel: int = 1,
    trace_mode: TraceMode = "actions",
    pre_judge_model: str = DEFAULT_PRE_JUDGE_MODEL,
    skip_pre_judge: bool = False,
    max_turns: int | None = None,
    pre_judge_max_turns: int | None = None,
    n_judges: int = 1,
    judges_parallel: int | None = None,
    max_budget_usd: float | None = DEFAULT_JUDGE_MAX_BUDGET_USD,
) -> dict[str, tuple[JudgeOutput, JudgeMetadata]]:
    """Batch judge selected episodes in an experiment output directory.

    Selection (default): all episodes that don't already have a judge_output.
    With `--sample 0.1`: 10% of those, randomly. With `--ids`: exactly those.
    With `n_parallel > 1`: run that many judge sub-processes concurrently.
    With `n_judges > 1`: run K independent judges per episode and aggregate into
    a JudgeConsensus. Cost scales linearly with K.

    Writes per-episode results into `episode_record.json` (or a sidecar if missing)
    and aggregate stats into `experiment_judge_summary.json`.
    """
    experiment_dir = Path(experiment_dir).resolve()
    refs = discover_episodes(experiment_dir)
    selected = select_episodes(
        refs,
        ids=ids,
        sample=sample,
        n=n,
        failures_only=failures_only,
        overwrite=overwrite,
        seed=seed,
    )

    if not selected:
        logger.info("No episodes selected to judge in %s", experiment_dir)
        return {}

    logger.info(
        "Judging %d / %d episodes in %s (n_parallel=%d, n_judges=%d)",
        len(selected),
        len(refs),
        experiment_dir.name,
        n_parallel,
        n_judges,
    )

    context_path: Path | None = None
    if not skip_pre_judge:
        context_path = run_pre_judge(
            experiment_dir,
            model=pre_judge_model,
            verbose=verbose,
            max_turns=pre_judge_max_turns,
        )

    results: dict[str, tuple[JudgeOutput, JudgeMetadata]] = {}
    if n_parallel > 1:
        results = asyncio.run(
            _judge_experiment_parallel(
                selected,
                experiment_dir,
                model,
                verbose,
                n_parallel,
                trace_mode,
                context_path,
                max_turns,
                n_judges,
                max_budget_usd,
            )
        )
    else:
        for ref in selected:
            try:
                if n_judges > 1:
                    _j_parallel = judges_parallel if judges_parallel is not None else n_judges
                    _ = _j_parallel  # reserved for future semaphore limiting per episode
                    consensus, all_judge_outputs, all_metas, all_actions = asyncio.run(
                        _judge_episode_k_times(
                            ref.episode_dir,
                            experiment_dir,
                            model,
                            verbose,
                            trace_mode,
                            context_path,
                            max_turns,
                            n_judges,
                            max_budget_usd=max_budget_usd,
                        )
                    )
                    # Primary output: first from the modal-outcome group.
                    modal_output = next(
                        (o for o in all_judge_outputs if o.outcome == consensus.outcome),
                        all_judge_outputs[0],
                    )
                    modal_meta = all_metas[all_judge_outputs.index(modal_output)]
                    primary_actions = all_actions[all_judge_outputs.index(modal_output)]
                    judge_output, judge_metadata, actions = modal_output, modal_meta, primary_actions
                else:
                    consensus, all_judge_outputs, all_metas = None, None, None
                    judge_output, judge_metadata, actions = asyncio.run(
                        _judge_episode_impl(
                            ref.episode_dir,
                            experiment_dir,
                            model,
                            verbose,
                            trace_mode,
                            context_path,
                            max_turns,
                            max_budget_usd=max_budget_usd,
                        )
                    )
            except Exception as e:
                logger.exception("Judge failed on %s: %s", ref.trajectory_id, e)
                continue
            ref.record = _load_episode_record(ref.record_path)
            pairs = list(zip(all_judge_outputs, all_metas)) if all_judge_outputs and all_metas else None
            _persist_judgment(
                ref, judge_output, judge_metadata, actions, trace_mode, consensus=consensus, all_outputs=pairs
            )
            results[ref.trajectory_id] = (judge_output, judge_metadata)

    _write_summary(experiment_dir, selected, results, model=model)
    return results


async def _judge_experiment_parallel(
    selected: list[EpisodeRef],
    experiment_dir: Path,
    model: str,
    verbose: bool,
    n_parallel: int,
    trace_mode: TraceMode = "actions",
    context_path: Path | None = None,
    max_turns: int | None = None,
    n_judges: int = 1,
    max_budget_usd: float | None = DEFAULT_JUDGE_MAX_BUDGET_USD,
) -> dict[str, tuple[JudgeOutput, JudgeMetadata]]:
    semaphore = asyncio.Semaphore(n_parallel)
    results: dict[str, tuple[JudgeOutput, JudgeMetadata]] = {}

    async def _one(ref: EpisodeRef) -> None:
        async with semaphore:
            try:
                if n_judges > 1:
                    consensus, all_judge_outputs, all_metas, all_actions = await _judge_episode_k_times(
                        ref.episode_dir,
                        experiment_dir,
                        model,
                        verbose,
                        trace_mode,
                        context_path,
                        max_turns,
                        n_judges,
                        max_budget_usd=max_budget_usd,
                    )
                    modal_output = next(
                        (o for o in all_judge_outputs if o.outcome == consensus.outcome),
                        all_judge_outputs[0],
                    )
                    idx = all_judge_outputs.index(modal_output)
                    judge_output = modal_output
                    judge_metadata = all_metas[idx]
                    actions = all_actions[idx]
                    pairs: list[tuple[JudgeOutput, JudgeMetadata]] | None = list(zip(all_judge_outputs, all_metas))
                else:
                    consensus = None
                    pairs = None
                    judge_output, judge_metadata, actions = await _judge_episode_impl(
                        ref.episode_dir,
                        experiment_dir,
                        model,
                        verbose,
                        trace_mode,
                        context_path,
                        max_turns,
                        max_budget_usd=max_budget_usd,
                    )
            except Exception as e:
                logger.exception("Judge failed on %s: %s", ref.trajectory_id, e)
                return
            ref.record = _load_episode_record(ref.record_path)
            _persist_judgment(
                ref, judge_output, judge_metadata, actions, trace_mode, consensus=consensus, all_outputs=pairs
            )
            results[ref.trajectory_id] = (judge_output, judge_metadata)

    await asyncio.gather(*[_one(ref) for ref in selected])
    return results


def _write_summary(
    experiment_dir: Path,
    selected: list[EpisodeRef],
    results: dict[str, tuple[JudgeOutput, JudgeMetadata]],
    *,
    model: str,
) -> None:
    if not results:
        return
    outcomes = Counter(o.outcome.value for o, _ in results.values())
    blames = Counter(o.primary_blame.value for o, _ in results.values())
    interventions = Counter(o.intervention.value for o, _ in results.values())
    total_cost = sum(m.cost_usd for _, m in results.values())
    total_prompt = sum(m.prompt_tokens for _, m in results.values())
    total_completion = sum(m.completion_tokens for _, m in results.values())
    n = len(results)

    judged_episodes = [
        {
            "trajectory_id": ref.trajectory_id,
            "episode_record": str(ref.record_path.relative_to(experiment_dir)),
        }
        for ref in selected
        if ref.trajectory_id in results
    ]

    summary = {
        "n_judged": n,
        "model": model,
        "judge_schema_version": JUDGE_SCHEMA_VERSION,
        "timestamp": time.time(),
        "total_judge_cost_usd": round(total_cost, 4),
        "avg_judge_cost_usd": round(total_cost / n, 4) if n else 0.0,
        "total_judge_prompt_tokens": total_prompt,
        "total_judge_completion_tokens": total_completion,
        "outcomes": dict(outcomes),
        "primary_blame": dict(blames),
        "interventions": dict(interventions),
        "report_csv": EXPERIMENT_JUDGE_REPORT_FILENAME,
        "judged_episodes": judged_episodes,
    }
    (experiment_dir / EXPERIMENT_JUDGE_SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))
    _write_csv_report(experiment_dir, selected, results)


def _write_csv_report(
    experiment_dir: Path,
    selected: list[EpisodeRef],
    results: dict[str, tuple[JudgeOutput, JudgeMetadata]],
) -> None:
    """Write one row per judged episode for spreadsheet-friendly inspection.

    Excludes `analysis` and `evidence` — they're too verbose for LLM consumption.
    Read them from the per-episode `episode_record.json` when needed.
    """
    fields = [
        "trajectory_id",
        "episode_record",
        "reward",
        "n_steps",
        "outcome",
        "primary_blame",
        "primary_blame_confidence",
        "other_blames",
        "hypothesis_confidence",
        "summary",
        "hypothesis",
        "intervention",
        "intervention_confidence",
        "cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "duration_s",
    ]
    path = experiment_dir / EXPERIMENT_JUDGE_REPORT_FILENAME
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ref in selected:
            if ref.trajectory_id not in results:
                continue
            o, m = results[ref.trajectory_id]
            reward = ref.record.score if ref.record is not None else None
            n_steps = ref.record.n_agent_steps if ref.record is not None else None
            w.writerow(
                {
                    "trajectory_id": ref.trajectory_id,
                    "episode_record": str(ref.record_path.relative_to(experiment_dir)),
                    "reward": reward,
                    "n_steps": n_steps,
                    "outcome": o.outcome.value,
                    "primary_blame": o.primary_blame.value,
                    "primary_blame_confidence": o.primary_blame_confidence,
                    "other_blames": ";".join(b.value for b in o.other_blames),
                    "hypothesis_confidence": o.hypothesis_confidence,
                    "summary": o.summary,
                    "hypothesis": o.hypothesis,
                    "intervention": o.intervention.value,
                    "intervention_confidence": o.intervention_confidence,
                    "cost_usd": round(m.cost_usd, 4),
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "duration_s": round(m.duration_s, 2),
                }
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_summary_table(results: dict[str, tuple[JudgeOutput, JudgeMetadata]]) -> None:
    if not results:
        print("(no episodes judged)")
        return
    print(f"\n{'trajectory_id':50s}  {'outcome':25s}  {'blame':24s}  {'conf':4s}  {'h_conf':6s}")
    print(f"{'-' * 50}  {'-' * 25}  {'-' * 24}  {'-' * 4}  {'-' * 6}")
    for tid, (o, _) in results.items():
        print(
            f"{tid[:50]:50s}  {o.outcome.value:25s}  {o.primary_blame.value:24s}  "
            f"{o.primary_blame_confidence:4d}  {o.hypothesis_confidence:6d}"
        )

    outcomes = Counter(o.outcome.value for o, _ in results.values())
    blames = Counter(o.primary_blame.value for o, _ in results.values())
    total_cost = sum(m.cost_usd for _, m in results.values())
    n = len(results)
    print(f"\nJudged {n} episodes  |  total cost: ${total_cost:.2f}  |  avg: ${total_cost / n:.3f}/ep")
    print(f"Outcomes:       {dict(outcomes)}")
    print(f"Primary blame:  {dict(blames)}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ch-judge",
        description="Post-hoc trajectory judge for cube-harness experiments.",
    )
    p.add_argument("path", type=Path, help="Experiment directory or single episode directory.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Judge model (default: {DEFAULT_MODEL}).")

    sel = p.add_argument_group("episode selection (mutually exclusive)")
    g = sel.add_mutually_exclusive_group()
    g.add_argument("--ids", default=None, help="Comma-separated trajectory IDs (or task IDs) to judge exactly.")
    g.add_argument(
        "--sample",
        type=float,
        default=None,
        metavar="FRACTION",
        help=f"Random fraction of eligible episodes (default: {DEFAULT_SAMPLE_FRACTION} when no other selector given).",
    )
    g.add_argument("--n", type=int, default=None, help="Random N eligible episodes.")
    g.add_argument("--all", action="store_true", help="Judge every eligible episode.")

    p.add_argument("--failures-only", action="store_true", help="Restrict to episodes with is_correct=False.")
    p.add_argument("--overwrite", action="store_true", help="Re-judge episodes that already have judge_output.")
    p.add_argument("--seed", type=int, default=None, help="Seed for sampling reproducibility.")
    p.add_argument("--summary", action="store_true", help="Print aggregate blame/outcome distribution.")
    p.add_argument("--n-parallel", type=int, default=1, help="Number of episodes to judge concurrently (default: 1).")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Stream the judge's tool calls and assistant text to stderr while it runs.",
    )
    p.add_argument(
        "--trace",
        choices=["actions", "full", "off"],
        default="actions",
        dest="trace_mode",
        help=(
            "Judge action trace level stored in judge_metadata.judge_actions. "
            "'actions' (default): compact list of (tool, summarised_input). "
            "'full': also includes raw_input dict. "
            "'off': nothing stored."
        ),
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_JUDGE_MAX_TURNS,
        dest="max_turns",
        help=f"Max turns for each episode judge call (default: {DEFAULT_JUDGE_MAX_TURNS}).",
    )
    p.add_argument(
        "--max-budget",
        type=float,
        default=DEFAULT_JUDGE_MAX_BUDGET_USD,
        dest="max_budget_usd",
        help=(
            f"Max USD budget per episode judge call (default: ${DEFAULT_JUDGE_MAX_BUDGET_USD:.2f}). "
            "With --n-judges K the total per episode is K × this value."
        ),
    )
    pj = p.add_argument_group("pre-judge (experiment-level context)")
    pj.add_argument(
        "--pre-judge-model",
        default=DEFAULT_PRE_JUDGE_MODEL,
        help=f"Model for the pre-judge context pass (default: {DEFAULT_PRE_JUDGE_MODEL}).",
    )
    pj.add_argument(
        "--no-pre-judge",
        action="store_true",
        help="Skip the pre-judge context pass (use existing judge_context.md if present).",
    )
    pj.add_argument(
        "--overwrite-context",
        action="store_true",
        help="Re-run the pre-judge even if judge_context.md already exists.",
    )
    pj.add_argument(
        "--pre-judge-max-turns",
        type=int,
        default=DEFAULT_PRE_JUDGE_MAX_TURNS,
        dest="pre_judge_max_turns",
        help=f"Max turns for the pre-judge context call (default: {DEFAULT_PRE_JUDGE_MAX_TURNS}).",
    )
    kj = p.add_argument_group("inter-judge agreement (--n-judges K)")
    kj.add_argument(
        "--n-judges",
        type=int,
        default=1,
        dest="n_judges",
        help="Number of independent judges per episode (default: 1). Cost scales linearly with K.",
    )
    kj.add_argument(
        "--judges-parallel",
        type=int,
        default=None,
        dest="judges_parallel",
        help="Concurrency for K judges within a single episode (default: same as --n-judges).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ids = [s.strip() for s in args.ids.split(",")] if args.ids else None
    sample: float | None
    if args.all:
        sample = 1.0
    elif args.sample is not None:
        sample = args.sample
    elif args.n is not None or ids is not None:
        sample = None
    else:
        sample = DEFAULT_SAMPLE_FRACTION

    # When --no-pre-judge is set, still pass the existing context file if present.
    if args.no_pre_judge:
        existing = args.path / PRE_JUDGE_CONTEXT_FILENAME
        if existing.exists():
            logger.info("--no-pre-judge set; reusing existing %s", existing)
    elif args.overwrite_context:
        run_pre_judge(args.path, model=args.pre_judge_model, verbose=args.verbose, overwrite=True)

    results = judge_experiment(
        args.path,
        model=args.model,
        ids=ids,
        sample=sample,
        n=args.n,
        failures_only=args.failures_only,
        overwrite=args.overwrite,
        seed=args.seed,
        verbose=args.verbose,
        n_parallel=args.n_parallel,
        trace_mode=args.trace_mode,
        pre_judge_model=args.pre_judge_model,
        skip_pre_judge=args.no_pre_judge,
        max_turns=args.max_turns,
        pre_judge_max_turns=args.pre_judge_max_turns,
        n_judges=args.n_judges,
        judges_parallel=args.judges_parallel,
        max_budget_usd=args.max_budget_usd,
    )

    if args.summary:
        _print_summary_table(results)
    else:
        for tid, (o, m) in results.items():
            print(f"{tid}: {o.outcome.value} / {o.primary_blame.value} (conf={o.primary_blame_confidence})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
