"""Tests for cube_harness.analyze.judge — schemas, transcript decoding, selection."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import msgpack
import pytest
import zstandard

from cube_harness.analyze.judge import (
    EXPERIMENT_JUDGE_REPORT_FILENAME,
    _extract_json_block,
    _persist_judgment,
    _write_csv_report,
    discover_episodes,
    extract_transcript,
    select_episodes,
)
from cube_harness.analyze.judge.selection import EpisodeRef
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
    assert m.judge_schema_version == "v1"


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
    transcript = (out / "transcript.txt").read_text()

    assert "Hello task description." in obs_text
    assert "ACTION bash" in act_text
    assert "ls /testbed" in act_text
    assert "Hello task description." in transcript
    assert "ls /testbed" in transcript


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
# _write_csv_report
# ---------------------------------------------------------------------------


def test_write_csv_report_columns_and_values(tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path, [("t1_ep0", False, False)])
    refs = discover_episodes(exp)
    ref = refs[0]

    judge_out = _valid_judge_output()
    judge_meta = JudgeMetadata(model="claude-sonnet-4-6", timestamp=1_000_000.0, cost_usd=0.05, duration_s=12.3)
    results: dict[str, tuple[JudgeOutput, JudgeMetadata]] = {ref.trajectory_id: (judge_out, judge_meta)}

    _write_csv_report(exp, refs, results)

    csv_path = exp / EXPERIMENT_JUDGE_REPORT_FILENAME
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["trajectory_id"] == "t1_ep0"
    assert row["outcome"] == judge_out.outcome.value
    assert row["primary_blame"] == judge_out.primary_blame.value
    assert row["primary_blame_confidence"] == str(judge_out.primary_blame_confidence)
    assert float(row["cost_usd"]) == pytest.approx(0.05)
    assert float(row["duration_s"]) == pytest.approx(12.3)
    # other_blames is semicolon-joined; single entry
    assert row["other_blames"] == "agent_scaffolding"


# ---------------------------------------------------------------------------
# _persist_judgment
# ---------------------------------------------------------------------------


def test_persist_judgment_updates_episode_record(tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path, [("t1_ep0", False, False)])
    refs = discover_episodes(exp)
    ref = refs[0]

    judge_out = _valid_judge_output()
    judge_meta = JudgeMetadata(model="claude-sonnet-4-6", timestamp=1_000_000.0)

    _persist_judgment(ref, judge_out, judge_meta)

    restored = EpisodeRecord.model_validate_json(ref.record_path.read_text())
    assert restored.judge_output is not None
    assert restored.judge_output.outcome == judge_out.outcome
    assert restored.judge_metadata is not None
    assert restored.judge_metadata.model == "claude-sonnet-4-6"
    # No trace file written when actions is empty/None
    assert not (ref.episode_dir / "judge_trace.json").exists()


def test_persist_judgment_writes_sidecar_when_no_record(tmp_path: Path) -> None:
    ep_dir = tmp_path / "ep_no_record"
    ep_dir.mkdir()
    record_path = ep_dir / EPISODE_RECORD_FILENAME

    ref = EpisodeRef(
        trajectory_id="ep_no_record",
        episode_dir=ep_dir,
        record_path=record_path,
        record=None,  # simulates an older run without episode_record.json
    )
    judge_out = _valid_judge_output()
    judge_meta = JudgeMetadata(model="claude-sonnet-4-6", timestamp=1_000_000.0)

    _persist_judgment(ref, judge_out, judge_meta, actions=[{"tool": "Read", "input": "transcript.txt"}])

    # No episode_record.json created — only the sidecar
    assert not record_path.exists()
    sidecar = ep_dir / "judge_output.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["judge_output"]["outcome"] == judge_out.outcome.value

    # Trace file written because actions is non-empty
    trace = ep_dir / "judge_trace.json"
    assert trace.exists()
    trace_payload = json.loads(trace.read_text())
    assert len(trace_payload["actions"]) == 1
