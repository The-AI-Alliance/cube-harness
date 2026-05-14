#!/usr/bin/env python3
"""Smoke test: end-to-end judge run against a programmatically built fixture.

Builds a tiny experiment dir on disk (1 episode, 2 step files, pre-seeded
judge_context.md so the benchmark-context sub-agent does not run — that
component has its own smoke at `judge_context_agent.py`), then drives
`judge_experiment(...)` against it with the default `general_blame` recipe
and the real `ClaudeCodeSDKDriver`.

What this exercises that unit tests cannot:
  - The real SDK call to Anthropic — proves `claude-agent-sdk` is wired up
    correctly on this host and our prompt actually elicits well-formed JSON.
  - The full `_judge_episode_impl` pipeline: transcript extraction →
    context-file validation → prompt assembly → driver.run → JSON parse →
    invariant check → `_persist_judgment` → batch CSV / JSON reports.
  - The `general_blame` use-case `OutputModel` against a real model response.
  - That `experiment_judge_report.{csv,json}` and `judge_trace.json` are
    actually written and parseable.

Cost: roughly 1-2 cents on `claude-haiku-4-5`, ~5x that on Sonnet. The
fixture trajectory is intentionally short (~3 steps).

Auto-skips when:
  - `claude-agent-sdk` is not importable, or
  - `ANTHROPIC_API_KEY` is not set.

Use `--driver claude-terminal` to run the same checks through the
subscription path; auto-skips when the `claude` CLI is not on PATH.

Usage:
    uv run scripts/smoke/judge_e2e.py
    uv run scripts/smoke/judge_e2e.py --model claude-haiku-4-5
    uv run scripts/smoke/judge_e2e.py --driver claude-terminal
    uv run scripts/smoke/judge_e2e.py --keep-temp     # leave fixture for inspection

Final line follows the cube-harness smoke contract:
    SMOKE OK: judge_e2e
    SMOKE FAIL: judge_e2e
    SMOKE SKIP: judge_e2e
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import msgpack
import typer
import zstandard

from cube_harness.analyze.judge import (
    DEFAULT_RECIPE,
    EXPERIMENT_JUDGE_REPORT_FILENAME,
    EXPERIMENT_JUDGE_SUMMARY_FILENAME,
    JudgeBatchConfig,
    judge_experiment,
)
from cube_harness.analyze.judge.core import EXPERIMENT_JUDGE_REPORT_JSON_FILENAME
from cube_harness.analyze.judge.driver import ClaudeCodeSDKDriver, TerminalClaudeDriver
from cube_harness.eval_log import EPISODE_RECORD_FILENAME, EpisodeRecord, UsageSummary

# A short, plausible failure trajectory. Three step files: initial obs (task
# description) → agent action (a Bash inspection) → terminal obs (env tells the
# agent it failed the verification). This is just enough for the judge to have
# something to chew on without being so trivial that the model answers in one
# token.
TRAJECTORY_ID = "smoke_task_ep0"
TASK_DESCRIPTION = (
    "Fix the off-by-one bug in `compute_total()` in src/billing.py so that "
    "the unit test `test_total_includes_tax` passes. The function currently "
    "returns `subtotal + tax - 1`; the test expects `subtotal + tax`."
)
AGENT_ACTION_CMD = "cat src/billing.py"
FAILURE_OBS = (
    "Verification failed: `test_total_includes_tax` still fails with "
    "`AssertionError: expected 110, got 109`. The agent inspected the file but "
    "did not modify it before declaring done."
)


def _write_step(steps_dir: Path, name: str, payload: dict) -> None:
    raw = msgpack.packb(payload, use_bin_type=True)
    cctx = zstandard.ZstdCompressor()
    (steps_dir / name).write_bytes(cctx.compress(raw))


def _build_fixture(root: Path) -> Path:
    """Create `<root>/exp/` containing one episode + a pre-seeded judge_context.md.

    Returns the experiment dir.
    """
    exp = root / "exp"
    ep = exp / "episodes" / TRAJECTORY_ID
    (ep / "steps").mkdir(parents=True)

    # Step 0: initial observation (task description).
    _write_step(
        ep / "steps",
        "000_obs.msgpack.zst",
        {
            "output": {
                "obs": {
                    "contents": [{"data": TASK_DESCRIPTION, "tool_call_id": None}],
                    "reward": None,
                    "done": False,
                }
            }
        },
    )
    # Step 1: agent action — read the file, but no edit follows.
    _write_step(
        ep / "steps",
        "001_act.msgpack.zst",
        {
            "output": {
                "actions": [{"name": "Bash", "arguments": {"command": AGENT_ACTION_CMD}}],
                "llm_calls": [],
                "error": None,
            }
        },
    )
    # Step 2: terminal observation — the env reports a failure.
    _write_step(
        ep / "steps",
        "002_obs.msgpack.zst",
        {
            "output": {
                "obs": {
                    "contents": [{"data": FAILURE_OBS, "tool_call_id": None}],
                    "reward": 0.0,
                    "done": True,
                }
            }
        },
    )

    # Episode record — what `_persist_judgment` updates.
    record = EpisodeRecord(
        evaluation_id="smoke-eval",
        sample_id="smoke_task",
        is_correct=False,
        score=0.0,
        num_turns=2,
        n_agent_steps=1,
        n_env_steps=2,
        usage=UsageSummary(),
        trajectory_id=TRAJECTORY_ID,
        timestamp=0.0,
    )
    (ep / EPISODE_RECORD_FILENAME).write_text(record.model_dump_json(indent=2))

    # Minimal experiment_config.json. The judge falls back to a dict view when
    # the typed Experiment cannot be loaded, so unimportable `_type` strings
    # are fine. Auto-skips collecting source paths from these (which is what we
    # want — the context file below is the source of truth).
    (exp / "experiment_config.json").write_text(
        json.dumps(
            {
                "name": "judge_e2e_smoke",
                "agent_config": {"_type": "smoke.SmokeAgentConfig"},
                "benchmark_config": {"_type": "smoke.SmokeBenchmarkConfig"},
            }
        )
    )

    # Pre-seed judge_context.md to skip the benchmark-context sub-agent — that
    # path has its own smoke. We point the judge at the cube-harness src dir
    # so it has something real to grep, plus the experiment dir itself.
    cube_harness_src = Path(__file__).resolve().parents[2] / "src" / "cube_harness"
    (exp / "judge_context.md").write_text(
        "# Judge context (smoke fixture)\n\n"
        "Pre-seeded by judge_e2e.py — bypasses the benchmark-context sub-agent.\n\n"
        "```paths\n"
        f"experiment: {exp}\n"
        f"cube_harness: {cube_harness_src}\n"
        "```\n"
    )
    return exp


def _check_sdk_driver() -> tuple[ClaudeCodeSDKDriver | None, str | None]:
    if importlib.util.find_spec("claude_agent_sdk") is None:
        return None, "claude-agent-sdk package not importable"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, "ANTHROPIC_API_KEY not set"
    return ClaudeCodeSDKDriver(), None


def _check_terminal_driver() -> tuple[TerminalClaudeDriver | None, str | None]:
    if shutil.which("claude") is None:
        return None, "`claude` CLI not on PATH"
    return TerminalClaudeDriver(), None


def _verify_outputs(exp_dir: Path, judge_out, judge_meta) -> list[tuple[bool, str]]:
    """Return a list of (passed, description) checks against the on-disk artefacts."""
    checks: list[tuple[bool, str]] = []

    # 1. The transcript was actually extracted.
    ep = exp_dir / "episodes" / TRAJECTORY_ID
    transcript = ep / "_judge_transcript" / "transcript.txt"
    checks.append((transcript.exists() and transcript.stat().st_size > 0, f"transcript written ({transcript})"))

    # 2. The judge produced a non-empty analysis grounded in the trajectory.
    has_analysis = bool(judge_out.analysis and len(judge_out.analysis) > 50)
    checks.append((has_analysis, f"analysis is non-trivial (len={len(judge_out.analysis or '')})"))

    # 3. The outcome / blame are within the closed taxonomy. Pydantic enforces
    # this at parse time, so reaching here means the values were valid; we
    # additionally assert they look plausible for a known-failure trajectory.
    plausible_outcome = judge_out.outcome.value in ("failure", "almost", "should_have_been_rewarded")
    checks.append((plausible_outcome, f"outcome={judge_out.outcome.value} matches a known-failure shape"))

    # 4. Evidence is non-empty when blame is non-`none`.
    if judge_out.primary_blame.value != "none":
        checks.append((len(judge_out.evidence) >= 1, f"evidence non-empty (n={len(judge_out.evidence)})"))

    # 5. Metadata reflects a real billable call.
    checks.append((judge_meta.prompt_tokens > 0, f"prompt_tokens > 0 ({judge_meta.prompt_tokens})"))
    checks.append((judge_meta.duration_s > 0, f"duration_s > 0 ({judge_meta.duration_s:.2f}s)"))

    # 6. Persistence side-effects.
    record_path = ep / EPISODE_RECORD_FILENAME
    record_after = json.loads(record_path.read_text())
    checks.append((record_after.get("judge_output") is not None, "judge_output landed in episode_record.json"))
    checks.append((record_after.get("judge_metadata") is not None, "judge_metadata landed in episode_record.json"))

    # 7. judge_trace.json exists with at least one tool action recorded.
    trace_path = ep / "judge_trace.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        checks.append(
            (
                isinstance(trace.get("actions"), list),
                f"judge_trace.json has actions list ({len(trace.get('actions', []))} entries)",
            )
        )
    else:
        checks.append((False, "judge_trace.json was not written"))

    # 8. Batch reports written (the experiment-level CSV + JSON).
    csv_path = exp_dir / EXPERIMENT_JUDGE_REPORT_FILENAME
    json_path = exp_dir / EXPERIMENT_JUDGE_REPORT_JSON_FILENAME
    summary_path = exp_dir / EXPERIMENT_JUDGE_SUMMARY_FILENAME
    checks.append((csv_path.exists(), f"{csv_path.name} written"))
    checks.append((json_path.exists(), f"{json_path.name} written"))
    checks.append((summary_path.exists(), f"{summary_path.name} written"))

    return checks


def main(
    driver: Annotated[str, typer.Option(help="Driver to use.", click_type=None)] = "claude-code-sdk",
    model: Annotated[
        str | None,
        typer.Option(help="Override the recipe's model. Defaults to the recipe's setting."),
    ] = None,
    keep_temp: Annotated[bool, typer.Option(help="Leave the fixture experiment dir for inspection.")] = False,
) -> None:
    """End-to-end judge smoke against a self-contained fixture experiment."""
    typer.echo(f"judge_e2e — driver={driver} model={model or DEFAULT_RECIPE.model}")

    # Driver selection mirrors the smoke contract: skip cleanly when the
    # selected driver's deps are missing.
    if driver == "claude-code-sdk":
        chosen, skip_reason = _check_sdk_driver()
    elif driver == "claude-terminal":
        chosen, skip_reason = _check_terminal_driver()
    else:
        typer.echo(f"Unknown driver: {driver!r} (expected 'claude-code-sdk' or 'claude-terminal')")
        typer.echo("SMOKE FAIL: judge_e2e")
        raise typer.Exit(1)

    if chosen is None:
        typer.echo(f"Driver {driver} unavailable: {skip_reason}")
        typer.echo("SMOKE SKIP: judge_e2e")
        raise typer.Exit(2)

    tmp_root = Path(tempfile.mkdtemp(prefix="judge_e2e_smoke_"))
    typer.echo(f"Fixture root: {tmp_root}")

    try:
        exp_dir = _build_fixture(tmp_root)
        typer.echo(f"Built fixture experiment at {exp_dir}")

        # Build the recipe with the optional model override applied.
        recipe = DEFAULT_RECIPE if model is None else DEFAULT_RECIPE.model_copy(update={"model": model})

        typer.echo(f"Calling judge_experiment(...) with recipe={recipe.name} ...")
        config = JudgeBatchConfig(
            recipe=recipe,
            driver=chosen,
            ids=[TRAJECTORY_ID],
            verbose=False,
        )
        results = judge_experiment(exp_dir, config)
    except Exception as e:
        typer.echo(f"Driver call raised: {type(e).__name__}: {e}")
        if not keep_temp:
            shutil.rmtree(tmp_root, ignore_errors=True)
        typer.echo("SMOKE FAIL: judge_e2e")
        raise typer.Exit(1) from e

    if TRAJECTORY_ID not in results:
        typer.echo(f"Judge returned no result for {TRAJECTORY_ID!r} — got keys {list(results)}")
        if not keep_temp:
            shutil.rmtree(tmp_root, ignore_errors=True)
        typer.echo("SMOKE FAIL: judge_e2e")
        raise typer.Exit(1)

    judge_out, judge_meta = results[TRAJECTORY_ID]
    typer.echo()
    typer.echo("Judge produced:")
    typer.echo(
        f"  outcome={judge_out.outcome.value} primary_blame={judge_out.primary_blame.value} confidence={judge_out.primary_blame_confidence}"
    )
    typer.echo(f"  summary: {judge_out.summary[:140]}")
    typer.echo(
        f"  metadata: {judge_meta.prompt_tokens}p+{judge_meta.completion_tokens}c tokens, ${judge_meta.cost_usd:.4f}, {judge_meta.duration_s:.1f}s"
    )
    typer.echo()

    checks = _verify_outputs(exp_dir, judge_out, judge_meta)
    passed = sum(1 for ok, _ in checks if ok)
    total = len(checks)
    for ok, label in checks:
        marker = "✓" if ok else "✗"
        typer.echo(f"  {marker} {label}")

    if not keep_temp:
        shutil.rmtree(tmp_root, ignore_errors=True)
        typer.echo()
        typer.echo("(fixture cleaned up — pass --keep-temp to inspect next time)")
    else:
        typer.echo()
        typer.echo(f"Fixture preserved at {tmp_root}")

    typer.echo()
    typer.echo(f"Summary: {passed}/{total} checks passed.")
    if passed == total:
        typer.echo("SMOKE OK: judge_e2e")
        raise typer.Exit(0)
    typer.echo("SMOKE FAIL: judge_e2e")
    raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
