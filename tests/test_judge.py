"""Tests for cube_harness.analyze.judge — schemas, transcript decoding, selection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import pytest
import zstandard

from cube_harness.analyze.judge import EpisodeRef as _EpisodeRef
from cube_harness.analyze.judge import (
    _build_consensus,
    _extract_json_block,
    _run_claude_code,
    _stratified_sample,
    _task_family,
    _validate_context,
    _validate_invariants,
    _write_episode_summary,
    discover_episodes,
    extract_transcript,
    select_episodes,
)
from cube_harness.eval_log import (
    EPISODE_RECORD_FILENAME,
    EpisodeRecord,
    JudgeMetadata,
    JudgeOutput,
    Outcome,
    UsageSummary,
)

# ---------------------------------------------------------------------------
# Schemas: round-trip and invariants
# ---------------------------------------------------------------------------


def _valid_judge_output(**overrides) -> JudgeOutput:
    base = dict(
        analysis="Agent looped on the same diagnostic command for 40 steps.",
        outcome="failure",
        summary="Agent diagnosed correctly but failed to write any fix to disk.",
        primary_blame="model_capability",
        primary_blame_confidence=4,
        other_blames=["agent_scaffolding"],
        evidence=[{"step": 28, "quote": "for i in range(...): print(...)"}],
        hypothesis="Add an explicit 'apply the fix to the source file' instruction.",
        hypothesis_confidence=3,
    )
    base.update(overrides)
    return JudgeOutput(**base)


def test_judge_output_roundtrip() -> None:
    obj = _valid_judge_output()
    payload = obj.model_dump_json()
    restored = JudgeOutput.model_validate_json(payload)
    assert restored == obj


def test_judge_output_rejects_out_of_range_confidence() -> None:
    with pytest.raises(Exception):
        _valid_judge_output(primary_blame_confidence=6)
    with pytest.raises(Exception):
        _valid_judge_output(hypothesis_confidence=-1)


def test_judge_output_rejects_unknown_blame() -> None:
    with pytest.raises(Exception):
        _valid_judge_output(primary_blame="my_made_up_category")


def test_judge_metadata_defaults() -> None:
    m = JudgeMetadata(model="claude-opus-4-7", timestamp=1234567890.0)
    assert m.cost_usd == 0.0
    assert m.judge_schema_version == "v2"


def test_episode_record_with_judge_metadata_roundtrip(tmp_path: Path) -> None:
    record = EpisodeRecord(
        evaluation_id="eval-1",
        sample_id="task-1",
        is_correct=False,
        score=0.0,
        num_turns=10,
        n_agent_steps=5,
        n_env_steps=5,
        usage=UsageSummary(),
        trajectory_id="task-1_ep0",
        timestamp=0.0,
        judge_output=_valid_judge_output(),
        judge_metadata=JudgeMetadata(model="claude-opus-4-7", timestamp=42.0, cost_usd=0.087),
    )
    payload = record.model_dump_json()
    restored = EpisodeRecord.model_validate_json(payload)
    assert restored.judge_metadata is not None
    assert restored.judge_metadata.cost_usd == 0.087
    assert restored.judge_output is not None
    assert restored.judge_output.outcome == Outcome.failure


# ---------------------------------------------------------------------------
# JSON extraction from judge response text
# ---------------------------------------------------------------------------


def test_extract_json_block_fenced() -> None:
    text = 'Some preamble.\n```json\n{"outcome": "failure", "x": 1}\n```\nTrailing chatter.'
    assert _extract_json_block(text) == {"outcome": "failure", "x": 1}


def test_extract_json_block_unfenced() -> None:
    text = 'final answer: {"outcome": "success", "primary_blame": "none"}'
    assert _extract_json_block(text) == {"outcome": "success", "primary_blame": "none"}


def test_extract_json_block_with_nested_braces() -> None:
    text = '```json\n{"evidence": [{"step": 1, "quote": "}"}]}\n```'
    assert _extract_json_block(text) == {"evidence": [{"step": 1, "quote": "}"}]}


def test_extract_json_block_raises_when_absent() -> None:
    with pytest.raises(ValueError):
        _extract_json_block("no json here")


# ---------------------------------------------------------------------------
# extract_transcript: msgpack.zst → readable .txt
# ---------------------------------------------------------------------------


def _write_step(steps_dir: Path, name: str, payload: dict) -> None:
    raw = msgpack.packb(payload, use_bin_type=True)
    cctx = zstandard.ZstdCompressor()
    (steps_dir / name).write_bytes(cctx.compress(raw))


def test_extract_transcript_decompresses_obs_and_act(tmp_path: Path) -> None:
    ep = tmp_path / "task_ep0"
    steps = ep / "steps"
    steps.mkdir(parents=True)

    _write_step(
        steps,
        "000_obs.msgpack.zst",
        {"output": {"obs": {"contents": [{"data": "Hello task description.", "tool_call_id": None}]}}},
    )
    _write_step(
        steps,
        "001_act.msgpack.zst",
        {"output": {"actions": [{"name": "bash", "arguments": {"command": "ls /testbed"}}], "llm_calls": []}},
    )

    out = tmp_path / "decoded"
    extract_transcript(ep, out)

    obs_text = (out / "steps" / "000_obs.txt").read_text()
    act_text = (out / "steps" / "001_act.txt").read_text()
    summary = (out / "episode_summary.txt").read_text()

    assert "Hello task description." in obs_text
    assert "ACTION bash" in act_text
    assert "ls /testbed" in act_text
    assert "Hello task description." in summary
    assert "ls /testbed" in summary


# ---------------------------------------------------------------------------
# Episode discovery and selection
# ---------------------------------------------------------------------------


def _make_experiment(tmp_path: Path, episodes: list[tuple[str, bool, bool]]) -> Path:
    """Build a fake experiment dir.

    episodes is a list of (trajectory_id, is_correct, has_judge).
    """
    exp = tmp_path / "exp"
    eps = exp / "episodes"
    eps.mkdir(parents=True)
    for tid, is_correct, has_judge in episodes:
        ep = eps / tid
        ep.mkdir()
        record = EpisodeRecord(
            evaluation_id="eval-1",
            sample_id=tid.split("_ep")[0],
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            num_turns=2,
            n_agent_steps=1,
            n_env_steps=1,
            usage=UsageSummary(),
            trajectory_id=tid,
            timestamp=0.0,
        )
        if has_judge:
            record = record.model_copy(update={"judge_output": _valid_judge_output()})
        (ep / EPISODE_RECORD_FILENAME).write_text(record.model_dump_json(indent=2))
    return exp


def test_discover_episodes_finds_each_subdir(tmp_path: Path) -> None:
    exp = _make_experiment(
        tmp_path,
        [("a_ep0", True, False), ("b_ep0", False, False)],
    )
    refs = discover_episodes(exp)
    assert {r.trajectory_id for r in refs} == {"a_ep0", "b_ep0"}
    assert all(r.record is not None for r in refs)


def test_select_episodes_failures_only_skips_judged(tmp_path: Path) -> None:
    exp = _make_experiment(
        tmp_path,
        [
            ("ok_ep0", True, False),
            ("fail1_ep0", False, False),
            ("fail2_ep0", False, True),  # already judged
        ],
    )
    refs = discover_episodes(exp)
    selected = select_episodes(refs, failures_only=True)
    assert {r.trajectory_id for r in selected} == {"fail1_ep0"}


def test_select_episodes_overwrite_includes_already_judged(tmp_path: Path) -> None:
    exp = _make_experiment(
        tmp_path,
        [("fail1_ep0", False, False), ("fail2_ep0", False, True)],
    )
    refs = discover_episodes(exp)
    selected = select_episodes(refs, failures_only=True, overwrite=True)
    assert {r.trajectory_id for r in selected} == {"fail1_ep0", "fail2_ep0"}


def test_select_episodes_ids_takes_priority(tmp_path: Path) -> None:
    exp = _make_experiment(
        tmp_path,
        [("a_ep0", True, False), ("b_ep0", False, False)],
    )
    refs = discover_episodes(exp)
    # IDs override sampling and failures_only filter
    selected = select_episodes(refs, ids=["a_ep0"], failures_only=True)
    assert {r.trajectory_id for r in selected} == {"a_ep0"}


def test_select_episodes_ids_match_task_id_prefix(tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path, [("django__django-11099_ep0", False, False)])
    refs = discover_episodes(exp)
    selected = select_episodes(refs, ids=["django__django-11099"])
    assert len(selected) == 1


def test_select_episodes_sample_is_seeded(tmp_path: Path) -> None:
    eps = [(f"t{i}_ep0", False, False) for i in range(20)]
    exp = _make_experiment(tmp_path, eps)
    refs = discover_episodes(exp)
    a = select_episodes(refs, sample=0.25, seed=42)
    b = select_episodes(refs, sample=0.25, seed=42)
    assert {r.trajectory_id for r in a} == {r.trajectory_id for r in b}
    assert len(a) == 5  # 20 * 0.25


def test_select_episodes_n_caps_at_pool_size(tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path, [("a_ep0", False, False), ("b_ep0", False, False)])
    refs = discover_episodes(exp)
    selected = select_episodes(refs, n=99)
    assert len(selected) == 2


# ---------------------------------------------------------------------------
# Helpers for new tests (Round 1+)
# ---------------------------------------------------------------------------


def _async_iter(items: list[Any]) -> Any:
    """Return an async generator factory that yields items from `items`."""

    async def _gen(*args: Any, **kwargs: Any) -> Any:
        for x in items:
            yield x

    return _gen


def _make_ref(trajectory_id: str, correct: bool = False) -> _EpisodeRef:
    """Build a minimal EpisodeRef with a stub EpisodeRecord."""
    record = EpisodeRecord(
        evaluation_id="eval-1",
        sample_id=trajectory_id.split("_ep")[0],
        is_correct=correct,
        score=1.0 if correct else 0.0,
        num_turns=1,
        n_agent_steps=1,
        n_env_steps=0,
        usage=UsageSummary(),
        trajectory_id=trajectory_id,
        timestamp=0.0,
    )
    return _EpisodeRef(
        trajectory_id=trajectory_id,
        episode_dir=Path("/fake") / trajectory_id,
        record_path=Path("/fake") / trajectory_id / EPISODE_RECORD_FILENAME,
        record=record,
    )


# ---------------------------------------------------------------------------
# R1.1 — Turn budget
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    usage: dict[str, int]
    total_cost_usd: float
    duration_ms: int


def test_run_claude_code_passes_max_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """When max_turns is set, it must reach ClaudeAgentOptions."""
    captured: dict[str, Any] = {}

    class _FakeOptions:
        def __init__(self, **kw: Any) -> None:
            captured.update(kw)

    monkeypatch.setattr("claude_agent_sdk.ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(
        "claude_agent_sdk.query",
        _async_iter([_FakeResult(usage={}, total_cost_usd=0.0, duration_ms=0)]),
    )
    asyncio.run(
        _run_claude_code(
            system_prompt="x",
            user_prompt="y",
            cwd=Path("."),
            additional_dirs=[],
            model="m",
            max_turns=42,
        )
    )
    assert captured.get("max_turns") == 42


def test_run_claude_code_default_max_turns_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default behavior unchanged — max_turns=None means no cap passed to SDK."""
    captured: dict[str, Any] = {}

    class _FakeOptions:
        def __init__(self, **kw: Any) -> None:
            captured.update(kw)

    monkeypatch.setattr("claude_agent_sdk.ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(
        "claude_agent_sdk.query",
        _async_iter([_FakeResult(usage={}, total_cost_usd=0.0, duration_ms=0)]),
    )
    asyncio.run(
        _run_claude_code(
            system_prompt="x",
            user_prompt="y",
            cwd=Path("."),
            additional_dirs=[],
            model="m",
        )
    )
    assert captured.get("max_turns") is None


# ---------------------------------------------------------------------------
# R1.2 — Smarter episode_summary.txt
# ---------------------------------------------------------------------------


def test_episode_summary_includes_first_obs(tmp_path: Path) -> None:
    """First obs (truncated to 2KB) appears at the top."""
    out = tmp_path / "out"
    out.mkdir()
    steps: list[tuple[int, str, str]] = [(0, "obs", "### Step 000 OBS\nSolve task X" + "y" * 5000)]
    _write_episode_summary(out, steps)
    text = (out / "episode_summary.txt").read_text()
    assert "Solve task X" in text
    assert len(text) < 10_000  # truncation enforced


def test_episode_summary_includes_error_steps(tmp_path: Path) -> None:
    """Steps containing 'ERROR:' must be in the summary."""
    out = tmp_path / "out"
    out.mkdir()
    steps: list[tuple[int, str, str]] = [
        (0, "obs", "### Step 000 OBS\nstart"),
        (15, "act", "### Step 015 ACT\nERROR: tool exception"),
        (50, "act", "### Step 050 ACT\nclean step"),
    ]
    _write_episode_summary(out, steps)
    text = (out / "episode_summary.txt").read_text()
    assert "ERROR: tool exception" in text


def test_episode_summary_includes_last_k_acts(tmp_path: Path) -> None:
    """Last 3 act + their following obs must appear."""
    out = tmp_path / "out"
    out.mkdir()
    steps: list[tuple[int, str, str]] = []
    for i in range(10):
        kind = "obs" if i % 2 == 0 else "act"
        steps.append((i, kind, f"### Step {i:03d} {kind.upper()}\nbody-{i}"))
    _write_episode_summary(out, steps)
    text = (out / "episode_summary.txt").read_text()
    # Last 3 acts are at idx 5, 7, 9 → with following obs idx 6, 8 (10 doesn't exist)
    for i in [5, 6, 7, 8, 9]:
        assert f"body-{i}" in text


def test_episode_summary_caps_at_30kb(tmp_path: Path) -> None:
    """When error steps balloon, total summary stays bounded."""
    out = tmp_path / "out"
    out.mkdir()
    big_body = "x" * 20_000
    steps: list[tuple[int, str, str]] = [(i, "act", f"### Step {i:03d} ACT\nERROR: " + big_body) for i in range(1, 11)]
    steps.insert(0, (0, "obs", "### Step 000 OBS\nstart"))
    _write_episode_summary(out, steps)
    text = (out / "episode_summary.txt").read_text()
    assert len(text) < 35_000  # 30KB body + headers


def test_episode_summary_handles_empty(tmp_path: Path) -> None:
    """Empty steps list → file still written with placeholder."""
    out = tmp_path / "out"
    out.mkdir()
    _write_episode_summary(out, [])
    assert (out / "episode_summary.txt").exists()


# ---------------------------------------------------------------------------
# R1.3 — Stratified pre-judge sampling
# ---------------------------------------------------------------------------


def test_task_family_extraction() -> None:
    assert _task_family(_make_ref("django__django-14500_ep1")) == "django"
    assert _task_family(_make_ref("matplotlib__matplotlib-24149_ep5")) == "matplotlib"
    assert _task_family(_make_ref("simple_task_ep2")) == "simple"


def test_stratified_sample_picks_distinct_families() -> None:
    """Two failed episodes from same family + two from different family →
    pick one from each, not two from the same."""
    refs = [
        _make_ref("django__django-1_ep1", correct=False),
        _make_ref("django__django-2_ep2", correct=False),  # same family
        _make_ref("astropy__astropy-3_ep3", correct=False),
        _make_ref("matplotlib__plot-4_ep4", correct=False),
    ]
    sample = _stratified_sample(refs, correct=False, max_n=2)
    families = {_task_family(r) for r in sample}
    assert len(families) == 2
    assert "django" in families  # first django wins


def test_stratified_sample_falls_back_when_one_family() -> None:
    """When all refs share one family, only one episode is returned per call."""
    refs = [_make_ref(f"django__django-{i}_ep{i}", correct=False) for i in range(3)]
    sample = _stratified_sample(refs, correct=False, max_n=2)
    assert len(sample) == 1  # only one family available


# ---------------------------------------------------------------------------
# R1.4 — Context validation
# ---------------------------------------------------------------------------


def test_validate_context_returns_true_when_paths_exist(tmp_path: Path) -> None:
    real = tmp_path / "exists.py"
    real.write_text("x")
    ctx = tmp_path / "judge_context.md"
    ctx.write_text(f"See `{real}` for details.")
    assert _validate_context(ctx) is True


def test_validate_context_returns_false_when_paths_stale(tmp_path: Path) -> None:
    ctx = tmp_path / "judge_context.md"
    ctx.write_text("`/nonexistent/a.py` `/nonexistent/b.py` `/nonexistent/c.py`")
    assert _validate_context(ctx) is False


def test_validate_context_returns_true_when_no_paths_cited(tmp_path: Path) -> None:
    ctx = tmp_path / "judge_context.md"
    ctx.write_text("No file paths in here, just prose.")
    assert _validate_context(ctx) is True


# ---------------------------------------------------------------------------
# R2.2 — Intervention taxonomy
# ---------------------------------------------------------------------------


def test_judge_output_accepts_intervention() -> None:
    obj = _valid_judge_output(intervention="scaffold_change", intervention_confidence=4)
    assert obj.intervention.value == "scaffold_change"


def test_judge_output_rejects_unknown_intervention() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _valid_judge_output(intervention="invent_new_category", intervention_confidence=3)


def test_judge_output_intervention_defaults_to_none_for_v1_records() -> None:
    """Schema-v1 records (no intervention field) must still load."""
    v1_payload = {
        "analysis": "...",
        "outcome": "success",
        "summary": "Done.",
        "primary_blame": "none",
        "primary_blame_confidence": 5,
        "other_blames": [],
        "evidence": [],
        "hypothesis": "n/a",
        "hypothesis_confidence": 0,
    }
    obj = JudgeOutput.model_validate(v1_payload)
    assert obj.intervention.value == "none"
    assert obj.intervention_confidence == 0


def test_validate_invariants_coerces_intervention_on_clean_success() -> None:
    obj = _valid_judge_output(
        outcome="success",
        primary_blame="none",
        evidence=[],
        intervention="model_upgrade",
        intervention_confidence=2,
    )
    _validate_invariants(obj)
    assert obj.intervention.value == "none"


# ---------------------------------------------------------------------------
# R3.2 — Inter-judge consensus
# ---------------------------------------------------------------------------


def test_build_consensus_unanimous() -> None:
    outputs = [_valid_judge_output() for _ in range(3)]
    c = _build_consensus(outputs)
    assert c.n_judges == 3
    assert c.outcome_agreement == 1.0
    assert c.blame_agreement == 1.0


def test_build_consensus_two_one_split() -> None:
    outputs = [
        _valid_judge_output(outcome="failure"),
        _valid_judge_output(outcome="failure"),
        _valid_judge_output(outcome="almost"),
    ]
    c = _build_consensus(outputs)
    assert c.outcome.value == "failure"
    assert abs(c.outcome_agreement - 2 / 3) < 1e-9


def test_build_consensus_avg_confidence() -> None:
    outputs = [
        _valid_judge_output(primary_blame_confidence=2),
        _valid_judge_output(primary_blame_confidence=4),
    ]
    c = _build_consensus(outputs)
    assert c.avg_primary_blame_confidence == 3.0
