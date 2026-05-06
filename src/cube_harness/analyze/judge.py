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
import importlib.util
import json
import logging
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import zstandard
from pydantic import ValidationError

from cube_harness.core import Trajectory
from cube_harness.eval_log import (
    EPISODE_RECORD_FILENAME,
    JUDGE_SCHEMA_VERSION,
    EpisodeRecord,
    JudgeMetadata,
    JudgeOutput,
)
from cube_harness.experiment import Experiment

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_SAMPLE_FRACTION = 0.10
EXPERIMENT_JUDGE_SUMMARY_FILENAME = "experiment_judge_summary.json"
EXPERIMENT_JUDGE_REPORT_FILENAME = "experiment_judge_report.csv"

# Tools the judge needs: read transcript files, grep through cube/agent source,
# inspect screenshots if any. No write/edit — the judge only produces a JSON answer.
JUDGE_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Bash"]


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

    Writes one file per step into `<out_dir>/steps/NNN_(obs|act).txt` plus a
    consolidated `transcript.txt`. Returns `out_dir`.
    """
    steps_dir = episode_dir / "steps"
    if not steps_dir.exists():
        raise FileNotFoundError(f"No steps/ directory in {episode_dir}")

    out_steps = out_dir / "steps"
    out_steps.mkdir(parents=True, exist_ok=True)

    consolidated: list[str] = []
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
            (out_steps / f"{step_idx:03d}_obs.txt").write_text(text)
        elif "_act" in step_file.name:
            text = _format_act(step_idx, raw)
            (out_steps / f"{step_idx:03d}_act.txt").write_text(text)
        else:
            continue
        consolidated.append(text)

    (out_dir / "transcript.txt").write_text("\n".join(consolidated))
    return out_dir


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

    Otherwise: failures-only → already-judged (unless `overwrite`) → sample/n.
    """
    if ids:
        wanted = set(ids)
        return [r for r in refs if r.trajectory_id in wanted or r.trajectory_id.split("_ep")[0] in wanted]

    pool = refs
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


# ---------------------------------------------------------------------------
# Claude Code SDK invocation
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
    # Try strict parse first, then trim trailing chatter.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    end = candidate.rfind("}")
    if end == -1:
        raise ValueError("No closing brace in judge output")
    return json.loads(candidate[: end + 1])


@dataclass
class _SDKResult:
    output_text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    duration_s: float


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


async def _run_claude_code(
    *,
    system_prompt: str,
    user_prompt: str,
    cwd: Path,
    additional_dirs: list[Path],
    model: str,
    verbose: bool = False,
) -> _SDKResult:
    """Invoke Claude Code via the SDK and return the assistant text + usage.

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
        allowed_tools=JUDGE_ALLOWED_TOOLS,
        permission_mode="bypassPermissions",
        cwd=str(cwd),
        add_dirs=[str(p) for p in additional_dirs],
        model=model,
        include_partial_messages=False,
    )
    if verbose:
        logger.info("Running judge with model %s and options: %r", model, options)

    final_text: list[str] = []
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
                elif isinstance(block, ToolUseBlock) and verbose:
                    _emit(f"  > {block.name}({_summarise_tool_input(block.name, block.input)})")
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


def _validate_invariants(obj: JudgeOutput) -> None:
    """Enforce the V1 invariants from the spec (post-parse, pre-write)."""
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


async def _judge_episode_impl(
    episode_dir: Path,
    experiment_dir: Path,
    model: str,
    verbose: bool,
) -> tuple[JudgeOutput, JudgeMetadata]:
    """Async core shared by judge_episode (single) and judge_experiment (parallel)."""
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
    )

    logger.info("Judging %s (reward=%s, steps=%s) with %s", episode_dir.name, reward, total_steps, model)

    result = await _run_claude_code(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        cwd=episode_dir,
        additional_dirs=list(source_paths.values()) + [transcript_dir],
        model=model,
        verbose=verbose,
    )

    obj = _extract_json_block(result.output_text)
    judge_output = JudgeOutput.model_validate(obj)
    _validate_invariants(judge_output)

    judge_metadata = JudgeMetadata(
        model=model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        duration_s=result.duration_s,
        timestamp=time.time(),
        judge_schema_version=JUDGE_SCHEMA_VERSION,
    )
    return judge_output, judge_metadata


def judge_episode(
    episode_dir: Path,
    *,
    experiment_dir: Path | None = None,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> tuple[JudgeOutput, JudgeMetadata]:
    """Run a post-hoc judge on a single episode trajectory directory.

    `experiment_dir` is needed only to locate `experiment_config.json`; if omitted,
    we look one level up from `episode_dir`.
    """
    episode_dir = Path(episode_dir).resolve()
    if experiment_dir is None:
        experiment_dir = episode_dir.parent.parent
    return asyncio.run(_judge_episode_impl(episode_dir, Path(experiment_dir).resolve(), model, verbose))


def _persist_judgment(
    ref: EpisodeRef,
    judge_output: JudgeOutput,
    judge_metadata: JudgeMetadata,
) -> None:
    """Write `judge_output` and `judge_metadata` into the episode_record.json.

    If the record file does not exist yet (older runs without atlas-eval-log enabled),
    write a sidecar `judge_output.json` so the result is not lost.
    """
    if ref.record is not None:
        updated = ref.record.model_copy(update={"judge_output": judge_output, "judge_metadata": judge_metadata})
        ref.record_path.write_text(updated.model_dump_json(indent=2))
    else:
        sidecar = ref.episode_dir / "judge_output.json"
        sidecar.write_text(
            json.dumps(
                {
                    "judge_output": judge_output.model_dump(mode="json"),
                    "judge_metadata": judge_metadata.model_dump(mode="json"),
                },
                indent=2,
            )
        )


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
) -> dict[str, tuple[JudgeOutput, JudgeMetadata]]:
    """Batch judge selected episodes in an experiment output directory.

    Selection (default): all episodes that don't already have a judge_output.
    With `--sample 0.1`: 10% of those, randomly. With `--ids`: exactly those.
    With `n_parallel > 1`: run that many judge sub-processes concurrently.

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
        "Judging %d / %d episodes in %s (n_parallel=%d)",
        len(selected),
        len(refs),
        experiment_dir.name,
        n_parallel,
    )

    results: dict[str, tuple[JudgeOutput, JudgeMetadata]] = {}
    if n_parallel > 1:
        results = asyncio.run(_judge_experiment_parallel(selected, experiment_dir, model, verbose, n_parallel))
    else:
        for ref in selected:
            try:
                judge_output, judge_metadata = judge_episode(
                    ref.episode_dir, experiment_dir=experiment_dir, model=model, verbose=verbose
                )
            except Exception as e:
                logger.exception("Judge failed on %s: %s", ref.trajectory_id, e)
                continue
            ref.record = _load_episode_record(ref.record_path)
            _persist_judgment(ref, judge_output, judge_metadata)
            results[ref.trajectory_id] = (judge_output, judge_metadata)

    _write_summary(experiment_dir, selected, results, model=model)
    return results


async def _judge_experiment_parallel(
    selected: list[EpisodeRef],
    experiment_dir: Path,
    model: str,
    verbose: bool,
    n_parallel: int,
) -> dict[str, tuple[JudgeOutput, JudgeMetadata]]:
    semaphore = asyncio.Semaphore(n_parallel)
    results: dict[str, tuple[JudgeOutput, JudgeMetadata]] = {}

    async def _one(ref: EpisodeRef) -> None:
        async with semaphore:
            try:
                judge_output, judge_metadata = await _judge_episode_impl(
                    ref.episode_dir, experiment_dir, model, verbose
                )
            except Exception as e:
                logger.exception("Judge failed on %s: %s", ref.trajectory_id, e)
                return
            ref.record = _load_episode_record(ref.record_path)
            _persist_judgment(ref, judge_output, judge_metadata)
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
    )

    if args.summary:
        _print_summary_table(results)
    else:
        for tid, (o, m) in results.items():
            print(f"{tid}: {o.outcome.value} / {o.primary_blame.value} (conf={o.primary_blame_confidence})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
