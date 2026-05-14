"""Post-batch synthesis: turn N per-episode judgments into one analysis.

Runs after `judge_experiment` finishes and `write_summary` has produced the
flat aggregates. The synthesis call asks an LLM-driven sub-agent (Opus by
default) to read the per-episode judge outputs and produce:

  - clustered failure patterns,
  - hypothesised root causes,
  - concrete intervention suggestions,
  - a human-readable markdown summary.

The structured payload is written as `<experiment_dir>/meta_analysis.json`;
the prose body is written as `<experiment_dir>/meta_analysis.md`.

When `journal_dir` is supplied, both files are also mirrored into
`<journal_dir>/<experiment_basename>/` so the meta-agent's outer loop can
collect a running history of what it observed across iterations.

Off-by-default for cheap repeat runs; the CLI's `--synthesize/--no-synthesize`
toggle controls it.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from cube.core import TypedBaseModel
from pydantic import Field, ValidationError

from cube_harness.analyze.judge.parse import extract_json_block
from cube_harness.analyze.judge.schema_prompt import model_to_json_example

if TYPE_CHECKING:
    from cube_harness.analyze.judge.driver import AgentDriver
    from cube_harness.eval_log import BaseJudgeOutput, JudgeMetadata

logger = logging.getLogger(__name__)

META_ANALYSIS_FILENAME = "meta_analysis.json"
META_ANALYSIS_MARKDOWN_FILENAME = "meta_analysis.md"
META_ANALYSIS_SCHEMA_VERSION = 1
DEFAULT_META_ANALYSIS_MODEL = "claude-opus-4-7"


class FailurePattern(TypedBaseModel):
    """A cluster of trajectories exhibiting the same failure mode.

    Field order is CoT-deliberate: describe what you see → cite the
    trajectories that show it → name the dominant blame → only then assign
    a short label. Models token-emit in this order, so naming-last forces
    the cluster to be understood before it is summarised.
    """

    description: str = Field(description="One or two sentences naming the pattern.")
    affected_trajectories: list[str] = Field(description="trajectory_ids the judge attributed to this pattern.")
    dominant_blame: str = Field(description="BlameCategory value most often cited for these trajectories.")
    name: str = Field(description="Short kebab-case label (e.g. 'inspects-but-never-edits').")


class RootCauseHypothesis(TypedBaseModel):
    """A guess at *why* one or more patterns are happening.

    Field order is CoT-deliberate: reason first, link to patterns second,
    score confidence last (after the reasoning has played out).
    """

    description: str = Field(description="One to three sentences. Cite patterns and per-episode evidence.")
    pattern_names: list[str] = Field(description="`FailurePattern.name` values this hypothesis covers.")
    confidence: int = Field(ge=0, le=5, description="0=guess, 5=load-bearing evidence across multiple patterns.")


class Intervention(TypedBaseModel):
    """A concrete, tryable change.

    Field order is CoT-deliberate: identify the target → justify what
    change should be made and why → then state the concrete change →
    finally score how confident we are it will help.
    """

    target: str = Field(
        description="What to change. e.g. 'agent system prompt', 'scaffold loop', 'evaluation criterion'.",
    )
    rationale: str = Field(description="Why this addresses one or more root causes.")
    change: str = Field(description="Specific edit, prompt addition, or scaffold tweak.")
    confidence: int = Field(ge=0, le=5, description="0=long shot, 5=highly likely to help.")


class MetaAnalysis(TypedBaseModel):
    """Cross-episode synthesis of an experiment's judgments.

    `markdown_summary` carries the LLM's human-readable prose; the rest is
    structured for programmatic aggregation across experiments by the
    meta-agent's outer loop.
    """

    schema_version: int = META_ANALYSIS_SCHEMA_VERSION
    experiment_id: str
    recipe: str
    driver: str
    model: str
    timestamp: float
    n_episodes_judged: int

    # Aggregates — duplicated from `experiment_judge_summary.json` for
    # self-contained reading.
    outcome_distribution: dict[str, int] = Field(default_factory=dict)
    primary_blame_distribution: dict[str, int] = Field(default_factory=dict)
    success_rate: float = 0.0

    # The LLM-authored synthesis.
    patterns: list[FailurePattern] = Field(default_factory=list)
    root_cause_hypotheses: list[RootCauseHypothesis] = Field(default_factory=list)
    suggested_interventions: list[Intervention] = Field(default_factory=list)
    markdown_summary: str = Field(description="Human-readable body; also written as meta_analysis.md.")

    notes: str | None = None
    cost_usd: float = 0.0
    duration_s: float = 0.0


# Fields the runtime supplies post-parse; the model must not emit these.
_RUNTIME_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "recipe",
        "driver",
        "model",
        "timestamp",
        "n_episodes_judged",
        "outcome_distribution",
        "primary_blame_distribution",
        "success_rate",
        "cost_usd",
        "duration_s",
    }
)


def _render_system_prompt() -> str:
    """Build the system prompt by deriving the JSON example from `MetaAnalysis`.

    The Pydantic class is the single spec; the prompt section is generated
    from `model_fields`, so renaming or reordering a field updates the prompt
    automatically. The runtime-provenance fields are stripped because the
    runtime fills them in after parsing.
    """
    example = model_to_json_example(MetaAnalysis, skip=_RUNTIME_PROVENANCE_FIELDS)
    return f"""You are a meta-analyst for a trajectory-judge sweep.

The trajectory judge has just produced one structured judgment per episode
in a single experiment. Your job is to read those judgments, synthesise
patterns across them, hypothesise root causes, and propose concrete
interventions a human engineer can act on next.

Stay tightly grounded in what the per-episode judgments support. Do not
invent failure modes the judgments don't already describe. Every
trajectory_id in `affected_trajectories` must come from the list of judged
trajectories in the user prompt.

You have read-only tools (Read / Glob / Grep / Bash). Use them to drill into
specific `judge_output.json` files when the summaries are too thin. Do not
modify any files. Your only output is the assistant message containing the
final JSON object.

The Pydantic-validated output must match this shape EXACTLY — do not rename,
omit, or add top-level fields. Each leaf is annotated `<type — description>`:

```json
{example}
```

Field order in your emitted JSON should follow the example above; the order
is deliberate (describe before naming, justify before changing, score after
reasoning).

The runtime fills in provenance and aggregate fields after parsing — you do
not need to emit `schema_version`, `experiment_id`, `recipe`, `driver`,
`model`, `timestamp`, `n_episodes_judged`, `outcome_distribution`,
`primary_blame_distribution`, `success_rate`, `cost_usd`, or `duration_s`."""


META_ANALYSIS_SYSTEM_PROMPT = _render_system_prompt()


def _digest_judgments(
    results: dict[str, tuple["BaseJudgeOutput", "JudgeMetadata"]],
) -> str:
    """Render the per-episode judgments as a compact text digest for the prompt.

    Each entry lists outcome, primary blame, confidence, summary, hypothesis,
    and the first two evidence quotes (enough signal without bloating the
    prompt for 100-episode sweeps).
    """
    lines: list[str] = []
    for tid, (judge_output, _meta) in sorted(results.items()):
        evidence_lines = []
        for ev in judge_output.evidence[:2]:
            evidence_lines.append(f"      step {ev.step}: {ev.quote[:140]!r}")
        evidence_block = "\n".join(evidence_lines) if evidence_lines else "      (no evidence)"
        lines.append(
            "\n".join(
                [
                    f"- {tid}: outcome={judge_output.outcome.value} "
                    f"primary_blame={judge_output.primary_blame.value} "
                    f"conf={judge_output.primary_blame_confidence}",
                    f"    summary: {judge_output.summary}",
                    f"    hypothesis: {judge_output.hypothesis}",
                    f"    evidence:\n{evidence_block}",
                ]
            )
        )
    return "\n".join(lines)


def _build_user_prompt(
    *,
    experiment_dir: Path,
    experiment_id: str,
    n_judged: int,
    outcome_distribution: dict[str, int],
    primary_blame_distribution: dict[str, int],
    success_rate: float,
    digest: str,
) -> str:
    return f"""Experiment: {experiment_id}
Experiment directory (drill-in via Read/Glob if needed): {experiment_dir}

Episodes judged: {n_judged}
Success rate: {success_rate:.2%}

Outcome distribution: {dict(outcome_distribution)}
Primary blame distribution: {dict(primary_blame_distribution)}

Per-episode digest:
{digest}

Produce a single JSON object matching the `MetaAnalysis` schema. Wrap in
a ```json fence. Include a non-empty `markdown_summary`. Reference only
trajectory_ids that appear in the digest above."""


def _build_retry_prompt(prior_raw_output: str, validation_error: str) -> str:
    """Compose a follow-up prompt that asks the agent to fix invalid JSON.

    Used after a `ValidationError` on the first attempt. The retry is cheap
    relative to the initial pass: no transcript digest, no tool use — just
    the prior output and the validator's complaint.
    """
    return f"""Your prior reply failed Pydantic validation. Re-emit the JSON
object with the schema corrections noted below — do NOT redo the analysis,
just fix the structural mismatch. Reply with a single ```json fence and
nothing else.

Prior reply:
{prior_raw_output}

Validation error from Pydantic:
{validation_error}

Re-emit a valid JSON object matching the schema in the system prompt."""


async def run_meta_analysis(
    *,
    experiment_dir: Path,
    experiment_id: str,
    recipe_name: str,
    driver: "AgentDriver",
    results: dict[str, tuple["BaseJudgeOutput", "JudgeMetadata"]],
    model: str = DEFAULT_META_ANALYSIS_MODEL,
    verbose: bool = False,
    max_retries: int = 1,
) -> MetaAnalysis:
    """Run the synthesis sub-agent and return a populated `MetaAnalysis`.

    On `ValidationError`, retries up to `max_retries` times with a cheap
    follow-up call that ships only the prior raw output + the validator's
    error message — no transcript digest, no tool use — and asks the model
    to fix the structural mismatch. Bounds retry cost while still self-
    correcting on the most common failure mode (field renames, missing
    fields, extra fields).

    Aggregates are computed here from `results` rather than re-reading the
    summary file — keeps the function pure(-ish) and lets tests pass a fake
    driver without on-disk dependencies.
    """
    from collections import Counter

    outcomes = Counter(o.outcome.value for o, _ in results.values())
    blames = Counter(o.primary_blame.value for o, _ in results.values())
    n_judged = len(results)
    n_success = outcomes.get("success", 0) + outcomes.get("success_lucky", 0)
    success_rate = n_success / n_judged if n_judged else 0.0

    initial_prompt = _build_user_prompt(
        experiment_dir=experiment_dir,
        experiment_id=experiment_id,
        n_judged=n_judged,
        outcome_distribution=dict(outcomes),
        primary_blame_distribution=dict(blames),
        success_rate=success_rate,
        digest=_digest_judgments(results),
    )

    total_cost = 0.0
    total_duration = 0.0
    prior_raw_output: str | None = None
    last_validation_error: str | None = None

    for attempt in range(max_retries + 1):
        user_prompt = (
            initial_prompt if attempt == 0 else _build_retry_prompt(prior_raw_output or "", last_validation_error or "")
        )
        result = await driver.run(
            system_prompt=META_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            cwd=experiment_dir,
            additional_dirs=[],
            model=model,
            verbose=verbose,
        )
        total_cost += result.cost_usd
        total_duration += result.duration_s
        prior_raw_output = result.output_text

        try:
            raw = extract_json_block(result.output_text)
        except ValueError as e:
            last_validation_error = f"could not find a JSON block: {e}"
            logger.warning("Meta-analysis attempt %d failed to extract JSON: %s", attempt + 1, e)
            continue

        # Provenance is ours, not the model's. Override (don't setdefault) so
        # any model-emitted values are corrected.
        raw["experiment_id"] = experiment_id
        raw["recipe"] = recipe_name
        raw["driver"] = driver.name
        raw["model"] = model
        raw["timestamp"] = time.time()
        raw["n_episodes_judged"] = n_judged
        raw["outcome_distribution"] = dict(outcomes)
        raw["primary_blame_distribution"] = dict(blames)
        raw["success_rate"] = success_rate
        raw["cost_usd"] = total_cost
        raw["duration_s"] = total_duration
        raw["schema_version"] = META_ANALYSIS_SCHEMA_VERSION

        try:
            return MetaAnalysis.model_validate(raw)
        except ValidationError as e:
            last_validation_error = str(e)
            logger.warning("Meta-analysis attempt %d failed validation: %s", attempt + 1, e)

    raise RuntimeError(
        f"Meta-analysis output failed schema validation after {max_retries + 1} attempt(s).\n"
        f"Last validation error: {last_validation_error}"
    )


def write_meta_analysis(experiment_dir: Path, analysis: MetaAnalysis) -> tuple[Path, Path]:
    """Write `meta_analysis.json` and `meta_analysis.md`. Returns the two paths."""
    json_path = experiment_dir / META_ANALYSIS_FILENAME
    md_path = experiment_dir / META_ANALYSIS_MARKDOWN_FILENAME
    json_path.write_text(json.dumps(analysis.model_dump(mode="json"), indent=2))
    md_path.write_text(analysis.markdown_summary.rstrip() + "\n")
    return json_path, md_path


def copy_to_journal(
    journal_dir: Path,
    experiment_id: str,
    json_path: Path,
    md_path: Path,
) -> tuple[Path, Path]:
    """Mirror the synthesis into `<journal_dir>/<experiment_id>/`.

    The meta-agent's slash command later reads this directory to accumulate
    a running narrative across iterations. The judge only deposits files;
    the meta-agent owns the narrative log.
    """
    out_dir = Path(journal_dir).expanduser() / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    j_dst = out_dir / json_path.name
    m_dst = out_dir / md_path.name
    j_dst.write_text(json_path.read_text())
    m_dst.write_text(md_path.read_text())
    logger.info("Mirrored meta-analysis to %s", out_dir)
    return j_dst, m_dst


__all__ = [
    "FailurePattern",
    "Intervention",
    "MetaAnalysis",
    "META_ANALYSIS_FILENAME",
    "META_ANALYSIS_MARKDOWN_FILENAME",
    "META_ANALYSIS_SCHEMA_VERSION",
    "META_ANALYSIS_SYSTEM_PROMPT",
    "DEFAULT_META_ANALYSIS_MODEL",
    "RootCauseHypothesis",
    "copy_to_journal",
    "run_meta_analysis",
    "write_meta_analysis",
]
