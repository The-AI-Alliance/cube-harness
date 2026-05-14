"""Tests for cube_harness.analyze.judge — schemas, transcript decoding,
selection, recipes, drivers, audit, cross-experiment writers."""

from __future__ import annotations

import asyncio
import csv
import json
import os
from pathlib import Path
from typing import Any

import msgpack
import pytest
import zstandard

from cube_harness.analyze.cross_experiment.cross_judge_agreement import (
    AGREEMENT_COLUMNS,
    AGREEMENT_REPORT_FILENAME,
    write_cross_judge_agreement,
)
from cube_harness.analyze.cross_experiment.joint_csv import (
    JOINT_REPORT_COLUMNS,
    JOINT_REPORT_FILENAME,
    write_joint_csv,
)
from cube_harness.analyze.judge import (
    EXPERIMENT_JUDGE_REPORT_FILENAME,
    EXPERIMENT_JUDGE_REPORT_JSON_FILENAME,
    EXPERIMENT_JUDGE_SUMMARY_FILENAME,
    AgentDriver,
    AuditOutput,
    DriverResult,
    JudgeBatchConfig,
    JudgeRecipe,
    SameAgentPreviousIteration,
    SameTaskDifferentAgent,
    ToolAction,
    TopKBySimilarityStub,
    discover_episodes,
    extract_json_block,
    extract_transcript,
    judge_experiment,
    persist_judgment,
    select_episodes,
    validate_context_file,
    write_csv_report,
)
from cube_harness.analyze.judge.context import JUDGE_CONTEXT_FILENAME
from cube_harness.analyze.judge.recipe import _deserialize_output_model
from cube_harness.analyze.judge.selection import EpisodeRef
from cube_harness.analyze.judge.use_cases import RECIPE_CATALOG
from cube_harness.analyze.judge.use_cases.general_blame import GeneralBlameOutput
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


def _valid_judge_output(**overrides: Any) -> JudgeOutput:
    base: dict[str, Any] = dict(
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
    assert extract_json_block(text) == {"outcome": "failure", "x": 1}


def test_extract_json_block_unfenced() -> None:
    text = 'final answer: {"outcome": "success", "primary_blame": "none"}'
    assert extract_json_block(text) == {"outcome": "success", "primary_blame": "none"}


def test_extract_json_block_with_nested_braces() -> None:
    text = '```json\n{"evidence": [{"step": 1, "quote": "}"}]}\n```'
    assert extract_json_block(text) == {"evidence": [{"step": 1, "quote": "}"}]}


def test_extract_json_block_raises_when_absent() -> None:
    with pytest.raises(ValueError):
        extract_json_block("no json here")


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
# Related-trajectory selectors
# ---------------------------------------------------------------------------


def test_selectors_drop_main_episode(tmp_path: Path) -> None:
    exp = _make_experiment(
        tmp_path,
        [("a_ep0", True, False), ("b_ep0", False, False), ("c_ep0", False, False)],
    )
    refs = discover_episodes(exp)
    main = refs[0]
    out = TopKBySimilarityStub(k=5).select(main_episode=main, experiment_dir=exp, all_refs=refs)
    assert main.episode_dir not in out
    assert len(out) <= 5


def test_same_task_different_agent(tmp_path: Path) -> None:
    # Two episodes with the same sample_id but distinct evaluation_id.
    exp = tmp_path / "exp"
    eps = exp / "episodes"
    eps.mkdir(parents=True)
    for tid, eval_id in [("task1_ep0", "eval-A"), ("task1_ep1", "eval-B"), ("task2_ep0", "eval-A")]:
        ep_dir = eps / tid
        ep_dir.mkdir()
        rec = EpisodeRecord(
            evaluation_id=eval_id,
            sample_id=tid.split("_ep")[0],
            is_correct=False,
            score=0.0,
            num_turns=1,
            n_agent_steps=1,
            n_env_steps=0,
            usage=UsageSummary(),
            trajectory_id=tid,
            timestamp=0.0,
        )
        (ep_dir / EPISODE_RECORD_FILENAME).write_text(rec.model_dump_json(indent=2))
    refs = discover_episodes(exp)
    main = next(r for r in refs if r.trajectory_id == "task1_ep0")
    out = SameTaskDifferentAgent(k=3).select(main_episode=main, experiment_dir=exp, all_refs=refs)
    assert len(out) == 1
    assert out[0].name == "task1_ep1"


def test_same_agent_previous_iteration(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    eps = exp / "episodes"
    eps.mkdir(parents=True)
    # Two earlier iterations and one later (must be excluded).
    for tid, ts in [
        ("task1_ep_old1", 10.0),
        ("task1_ep_old2", 20.0),
        ("task1_ep_main", 30.0),
        ("task1_ep_later", 40.0),
    ]:
        ep_dir = eps / tid
        ep_dir.mkdir()
        rec = EpisodeRecord(
            evaluation_id="eval-same",
            sample_id="task1",
            is_correct=False,
            score=0.0,
            num_turns=1,
            n_agent_steps=1,
            n_env_steps=0,
            usage=UsageSummary(),
            trajectory_id=tid,
            timestamp=ts,
        )
        (ep_dir / EPISODE_RECORD_FILENAME).write_text(rec.model_dump_json(indent=2))
    refs = discover_episodes(exp)
    main = next(r for r in refs if r.trajectory_id == "task1_ep_main")
    out = SameAgentPreviousIteration(k=5).select(main_episode=main, experiment_dir=exp, all_refs=refs)
    assert [p.name for p in out] == ["task1_ep_old1", "task1_ep_old2"]


# ---------------------------------------------------------------------------
# JudgeRecipe serialisation + catalog
# ---------------------------------------------------------------------------


def test_judge_recipe_serialization() -> None:
    """Recipe round-trips through model_dump_json — output_model survives as a dotted name."""
    recipe = RECIPE_CATALOG["general_blame"]
    payload = recipe.model_dump_json()
    parsed = json.loads(payload)
    assert "output_model" in parsed
    assert parsed["output_model"].endswith("GeneralBlameOutput")

    # Round-trip via Pydantic — output_model gets re-resolved to the class.
    restored = JudgeRecipe.model_validate_json(payload)
    assert restored.output_model is GeneralBlameOutput
    assert restored.name == "general_blame"
    assert restored.allowed_tools == ("Read", "Glob", "Grep", "Bash")


def test_deserialize_output_model_rejects_non_typed_base_model() -> None:
    with pytest.raises(TypeError):
        _deserialize_output_model("builtins.dict")


def test_recipe_catalog_assembly() -> None:
    """RECIPE_CATALOG contains the three initial recipes from the spec."""
    assert set(RECIPE_CATALOG.keys()) == {"general_blame", "profiling", "agent_scaffolding"}
    for name, recipe in RECIPE_CATALOG.items():
        assert recipe.name == name
        assert recipe.system_prompt.strip()
        assert recipe.user_prompt_template.strip()


# ---------------------------------------------------------------------------
# AuditOutput
# ---------------------------------------------------------------------------


def test_audit_output_schema_accepts_valid_payload() -> None:
    a = AuditOutput(
        recipe="general_blame",
        driver="claude-code-sdk",
        verdict="sound",
        reasoning_quality=4,
        ease_of_analysis=3,
        context_quality=5,
        tooling_gaps=["BashOutput on long-running commands"],
        missed_evidence=[],
        alternative_blames=[{"blame": "eval_brittle", "rationale": "graders rejected a valid patch"}],
        notes=None,
    )
    assert a.schema_version == 1
    assert a.verdict == "sound"
    assert a.alternative_blames[0].blame == "eval_brittle"


def test_audit_output_rejects_out_of_range_score() -> None:
    with pytest.raises(Exception):
        AuditOutput(
            recipe="general_blame",
            driver="claude-code-sdk",
            verdict="sound",
            reasoning_quality=6,  # out of range
            ease_of_analysis=3,
            context_quality=3,
        )


def test_audit_output_rejects_unknown_verdict() -> None:
    with pytest.raises(Exception):
        AuditOutput(
            recipe="general_blame",
            driver="claude-code-sdk",
            verdict="maybe",  # not in the Literal set
            reasoning_quality=3,
            ease_of_analysis=3,
            context_quality=3,
        )


# ---------------------------------------------------------------------------
# Context file validation
# ---------------------------------------------------------------------------


def test_validate_context_file_parses_paths_block(tmp_path: Path) -> None:
    p = tmp_path / JUDGE_CONTEXT_FILENAME
    other = tmp_path / "sub"
    other.mkdir()
    p.write_text(f"# context\n\n```paths\nsub: {other}\n# comment line\n\n{tmp_path}\n```\n")
    resolved = validate_context_file(p)
    assert resolved["sub"] == other
    assert "path_1" in resolved


def test_validate_context_file_raises_on_missing_path(tmp_path: Path) -> None:
    p = tmp_path / JUDGE_CONTEXT_FILENAME
    p.write_text("```paths\nfake: /no/such/path\n```\n")
    with pytest.raises(FileNotFoundError):
        validate_context_file(p)


def test_validate_context_file_raises_without_paths_fence(tmp_path: Path) -> None:
    p = tmp_path / JUDGE_CONTEXT_FILENAME
    p.write_text("# no fenced block here\n")
    with pytest.raises(ValueError):
        validate_context_file(p)


# ---------------------------------------------------------------------------
# Cross-experiment writers
# ---------------------------------------------------------------------------


def _judge_run_pair(blame: str, outcome: str, confidence: int) -> tuple[JudgeOutput, JudgeMetadata]:
    o = _valid_judge_output(primary_blame=blame, outcome=outcome, primary_blame_confidence=confidence)
    m = JudgeMetadata(model="claude-sonnet-4-6", timestamp=1_000_000.0 + confidence, cost_usd=0.01)
    return o, m


def test_cross_judge_agreement_writer_modal_and_columns(tmp_path: Path) -> None:
    judgments = {
        ("trajA", "general_blame"): [
            _judge_run_pair("model_capability", "failure", 4),
            _judge_run_pair("model_capability", "failure", 3),
            _judge_run_pair("agent_scaffolding", "failure", 5),
        ],
        ("trajB", "general_blame"): [
            _judge_run_pair("eval_brittle", "should_have_been_rewarded", 2),
        ],  # single judgment — skipped
    }
    out = write_cross_judge_agreement(tmp_path, judgments=judgments)
    assert out.name == AGREEMENT_REPORT_FILENAME
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    row = rows[0]
    assert tuple(row.keys()) == AGREEMENT_COLUMNS
    assert row["trajectory_id"] == "trajA"
    assert row["primary_blame_modal"] == "model_capability"
    assert float(row["primary_blame_agreement"]) == pytest.approx(2 / 3, abs=1e-3)
    assert int(row["n_judgments"]) == 3
    assert float(row["confidence_mean"]) == pytest.approx(4.0)


def _fake_experiment_with_csv(experiment_dir: Path, *, recipe: str, driver: str, rows: list[dict[str, str]]) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    # experiment_config.json with agent/benchmark _type strings
    (experiment_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "agent_config": {"_type": "my_agent.MyAgentConfig"},
                "benchmark_config": {"_type": "my_bench.MyBenchmarkConfig"},
            }
        )
    )
    (experiment_dir / "experiment_judge_summary.json").write_text(
        json.dumps({"recipe": recipe, "driver": driver, "litellm_proxy_url": ""})
    )
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
    csv_path = experiment_dir / "experiment_judge_report.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in fields})


def test_joint_csv_writer_walks_sweep(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    _fake_experiment_with_csv(
        sweep / "exp1",
        recipe="general_blame",
        driver="claude-code-sdk",
        rows=[{"trajectory_id": "tA", "outcome": "failure", "primary_blame": "model_capability"}],
    )
    _fake_experiment_with_csv(
        sweep / "exp2",
        recipe="profiling",
        driver="claude-code-sdk",
        rows=[{"trajectory_id": "tB", "outcome": "success", "primary_blame": "none"}],
    )

    out = write_joint_csv(sweep)
    assert out.name == JOINT_REPORT_FILENAME
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    assert set(rows[0].keys()) == set(JOINT_REPORT_COLUMNS)
    by_id = {r["trajectory_id"]: r for r in rows}
    assert by_id["tA"]["experiment_id"] == "exp1"
    assert by_id["tA"]["recipe"] == "general_blame"
    assert by_id["tA"]["agent_dotted"] == "my_agent.MyAgentConfig"
    assert by_id["tB"]["benchmark_dotted"] == "my_bench.MyBenchmarkConfig"


def test_joint_csv_writer_empty_sweep_writes_header(tmp_path: Path) -> None:
    sweep = tmp_path / "empty"
    sweep.mkdir()
    out = write_joint_csv(sweep)
    rows = list(csv.DictReader(out.open()))
    assert rows == []
    # Header still written
    assert (sweep / JOINT_REPORT_FILENAME).read_text().splitlines()[0].split(",")[0] == "experiment_id"


# ---------------------------------------------------------------------------
# write_csv_report — unchanged columns, used by old and new paths.
# ---------------------------------------------------------------------------


def test_write_csv_report_columns_and_values(tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path, [("t1_ep0", False, False)])
    refs = discover_episodes(exp)
    ref = refs[0]

    judge_out = _valid_judge_output()
    judge_meta = JudgeMetadata(model="claude-sonnet-4-6", timestamp=1_000_000.0, cost_usd=0.05, duration_s=12.3)
    results: dict[str, tuple[JudgeOutput, JudgeMetadata]] = {ref.trajectory_id: (judge_out, judge_meta)}

    write_csv_report(exp, refs, results)

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
# persist_judgment
# ---------------------------------------------------------------------------


def test_persist_judgment_updates_episode_record(tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path, [("t1_ep0", False, False)])
    refs = discover_episodes(exp)
    ref = refs[0]

    judge_out = _valid_judge_output()
    judge_meta = JudgeMetadata(model="claude-sonnet-4-6", timestamp=1_000_000.0)

    persist_judgment(ref, judge_out, judge_meta)

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

    persist_judgment(ref, judge_out, judge_meta, actions=[ToolAction(tool="Read", input_summary="transcript.txt")])

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


# ---------------------------------------------------------------------------
# Integration: full judge_episode pipeline with a fake driver
# ---------------------------------------------------------------------------

_VALID_JUDGE_JSON = json.dumps(
    {
        "analysis": "Agent ran `cat foo.py` but never attempted a fix.",
        "outcome": "failure",
        "summary": "Agent read the file but made no edit.",
        "primary_blame": "model_capability",
        "primary_blame_confidence": 4,
        "other_blames": ["agent_scaffolding"],
        "evidence": [{"step": 1, "quote": "cat foo.py"}],
        "hypothesis": "Provide an explicit edit instruction in the system prompt.",
        "hypothesis_confidence": 3,
    }
)


class _FakeDriver:
    """Implements the AgentDriver Protocol with a canned response.

    Tracks the kwargs of the last call so tests can assert on prompt content.
    """

    name = "fake-driver"
    max_parallelism = 4

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.last_call: dict[str, Any] | None = None

    async def run(self, **kwargs: Any) -> DriverResult:
        self.last_call = kwargs
        return DriverResult(
            output_text=self.output_text,
            prompt_tokens=1200,
            completion_tokens=300,
            cost_usd=0.042,
            duration_s=8.5,
            actions=[],
            litellm_proxy_url=None,
            session_id=None,
        )

    async def continue_session(self, **kwargs: Any) -> DriverResult:
        raise NotImplementedError("not needed for these tests")


def _make_episode_dir(tmp_path: Path, trajectory_id: str) -> tuple[Path, Path]:
    """Create a minimal experiment dir with one episode and a pre-seeded judge_context.md."""
    exp = tmp_path / "exp"
    ep = exp / "episodes" / trajectory_id
    (ep / "steps").mkdir(parents=True)

    _write_step(
        ep / "steps",
        "000_obs.msgpack.zst",
        {
            "output": {
                "obs": {
                    "contents": [{"data": "Fix the bug in foo.py.", "tool_call_id": None}],
                    "reward": None,
                    "done": False,
                }
            }
        },
    )
    _write_step(
        ep / "steps",
        "001_act.msgpack.zst",
        {
            "output": {
                "actions": [{"name": "Bash", "arguments": {"command": "cat foo.py"}}],
                "llm_calls": [],
                "error": None,
            }
        },
    )

    record = EpisodeRecord(
        evaluation_id="eval-1",
        sample_id=trajectory_id.split("_ep")[0],
        is_correct=False,
        score=0.0,
        num_turns=2,
        n_agent_steps=1,
        n_env_steps=1,
        usage=UsageSummary(),
        trajectory_id=trajectory_id,
        timestamp=0.0,
    )
    (ep / EPISODE_RECORD_FILENAME).write_text(record.model_dump_json(indent=2))

    # Pre-seed judge_context.md so the benchmark-context sub-agent does not run.
    # The file points at the experiment dir itself — any existing path works.
    (exp / JUDGE_CONTEXT_FILENAME).write_text(f"# auto-seeded for tests\n\n```paths\nexp: {exp}\n```\n")
    return exp, ep


def test_judge_episode_pipeline(tmp_path: Path) -> None:
    """Drive judge_experiment end-to-end with a fake driver.

    Exercises: transcript extraction → context file validation → prompt
    building → driver (fake) → JSON parsing → invariant validation →
    persist_judgment → write_summary → write_csv_report → write_json_report.
    """
    exp, ep = _make_episode_dir(tmp_path, "task1_ep0")
    driver = _FakeDriver(output_text=f"Here is my analysis:\n```json\n{_VALID_JUDGE_JSON}\n```")

    # synthesize=False here — the integration test is scoped to the judge
    # pipeline; meta-analysis has its own dedicated test below.
    results = judge_experiment(exp, JudgeBatchConfig(driver=driver, ids=["task1_ep0"], synthesize=False))

    assert "task1_ep0" in results
    judge_out, judge_meta = results["task1_ep0"]

    # -- Transcript was extracted into the episode dir --
    assert (ep / "_judge_transcript" / "transcript.txt").exists()
    assert (ep / "_judge_transcript" / "steps" / "000_obs.txt").read_text().startswith("### Step 000 OBS")
    assert "cat foo.py" in (ep / "_judge_transcript" / "steps" / "001_act.txt").read_text()

    # -- JudgeOutput fields round-tripped correctly --
    assert judge_out.outcome == Outcome.failure
    assert judge_out.primary_blame.value == "model_capability"
    assert judge_out.primary_blame_confidence == 4
    assert len(judge_out.evidence) == 1
    assert judge_out.evidence[0].quote == "cat foo.py"

    # -- JudgeMetadata reflects the driver's billing --
    assert judge_meta.prompt_tokens == 1200
    assert judge_meta.cost_usd == pytest.approx(0.042)
    assert judge_meta.model == "claude-sonnet-4-6"

    # -- episode_record.json updated in-place by persist_judgment --
    restored = EpisodeRecord.model_validate_json((ep / EPISODE_RECORD_FILENAME).read_text())
    assert restored.judge_output is not None
    assert restored.judge_output.outcome == Outcome.failure
    assert restored.judge_metadata is not None
    assert restored.judge_metadata.cost_usd == pytest.approx(0.042)

    # -- experiment_judge_summary.json written by write_summary --
    summary = json.loads((exp / EXPERIMENT_JUDGE_SUMMARY_FILENAME).read_text())
    assert summary["n_judged"] == 1
    assert summary["outcomes"] == {"failure": 1}
    assert summary["primary_blame"] == {"model_capability": 1}
    assert summary["total_judge_cost_usd"] == pytest.approx(0.042)
    assert summary["recipe"] == "general_blame"
    assert summary["driver"] == "fake-driver"

    # -- experiment_judge_report.csv written by write_csv_report --
    csv_path = exp / EXPERIMENT_JUDGE_REPORT_FILENAME
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    assert rows[0]["outcome"] == "failure"
    assert rows[0]["primary_blame"] == "model_capability"

    # -- experiment_judge_report.json written by write_json_report --
    json_path = exp / EXPERIMENT_JUDGE_REPORT_JSON_FILENAME
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["recipe"] == "general_blame"
    assert payload["driver"] == "fake-driver"
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["judge_output"]["primary_blame"] == "model_capability"

    # -- Driver received the right kwargs --
    assert driver.last_call is not None
    assert driver.last_call["model"] == "claude-sonnet-4-6"
    assert driver.last_call["cwd"] == ep
    assert "task1_ep0" in driver.last_call["user_prompt"]
    assert "_judge_transcript" in driver.last_call["user_prompt"]


# ---------------------------------------------------------------------------
# TerminalClaudeDriver — subprocess shape
# ---------------------------------------------------------------------------


def test_terminal_driver_builds_expected_argv(tmp_path: Path) -> None:
    """Verify the command-line construction without spawning a real subprocess."""
    from cube_harness.analyze.judge.driver import TerminalClaudeDriver

    drv = TerminalClaudeDriver(executable="claude")
    args = drv._build_args(
        user_prompt="hello",
        system_prompt="be a judge",
        cwd=tmp_path,
        additional_dirs=[tmp_path / "extra"],
        model="claude-sonnet-4-6",
        allowed_tools=("Read", "Glob"),
        permission_mode="bypassPermissions",
    )
    assert args[0] == "claude"
    assert "-p" in args and args[args.index("-p") + 1] == "hello"
    assert "--append-system-prompt" in args
    assert "--allowedTools" in args and "Read,Glob" in args
    assert "--output-format" in args and "json" in args
    assert "--model" in args and "claude-sonnet-4-6" in args
    assert "--add-dir" in args
    assert "--dangerously-skip-permissions" in args


def test_terminal_driver_parses_envelope() -> None:
    """JSON envelope from `claude -p` is parsed into a DriverResult."""
    from cube_harness.analyze.judge.driver import TerminalClaudeDriver

    drv = TerminalClaudeDriver()
    envelope = json.dumps(
        {
            "result": "```json\n" + _VALID_JUDGE_JSON + "\n```",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "session_id": "sess-1",
        }
    ).encode()
    result = drv._parse_envelope(envelope, duration_s=1.23, proxy_url=None, trace_mode="actions")
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 50
    assert result.session_id == "sess-1"
    assert "model_capability" in result.output_text


# ---------------------------------------------------------------------------
# Benchmark-context-agent
# ---------------------------------------------------------------------------


def test_benchmark_context_agent_writes_paths_block(tmp_path: Path) -> None:
    """Mock the driver; verify the agent writes a judge_context.md with a paths block."""
    from cube_harness.analyze.judge.benchmark_context_agent import generate_context_file

    other = tmp_path / "agent_src"
    other.mkdir()

    class _FakeContextDriver:
        name = "fake-context-driver"
        max_parallelism = 1

        async def run(self, **kwargs: Any) -> DriverResult:
            return DriverResult(
                output_text=(f"# auto-generated context\n\n```paths\nagent_src: {other}\n```\n"),
            )

        async def continue_session(self, **kwargs: Any) -> DriverResult:
            raise NotImplementedError

    import asyncio

    out = asyncio.run(generate_context_file(tmp_path, driver=_FakeContextDriver()))
    assert out.name == JUDGE_CONTEXT_FILENAME
    assert "```paths" in out.read_text()
    resolved = validate_context_file(out)
    assert resolved["agent_src"] == other


def test_benchmark_context_agent_fallback_when_no_driver(tmp_path: Path) -> None:
    """No driver → uses the collect_source_paths fallback, still produces a parseable file."""
    import asyncio

    from cube_harness.analyze.judge.benchmark_context_agent import generate_context_file

    out = asyncio.run(generate_context_file(tmp_path, driver=None))
    assert out.exists()
    # The fallback either lists at least one real path or a "no source paths" comment.
    assert "```paths" in out.read_text()


# ---------------------------------------------------------------------------
# CLI smoke (subcommand dispatch only — no real LLM calls)
# ---------------------------------------------------------------------------


def test_cli_main_with_help_returns_zero() -> None:
    from cube_harness.analyze.judge.cli import main

    # Capture the SystemExit; --help is the safest probe that exercises the typer app.
    rc = main(["--help"])
    assert rc == 0


# Skip the real `claude` CLI smoke test in normal runs — it requires the binary.
@pytest.mark.skipif(
    not (os.environ.get("CLAUDE_CLI_AVAILABLE") or False),
    reason="real `claude` CLI not available (set CLAUDE_CLI_AVAILABLE=1 to enable)",
)
@pytest.mark.slow
def test_terminal_driver_real_subprocess() -> None:  # pragma: no cover — env-gated
    """End-to-end check against the real terminal CLI. Slow + gated."""
    import asyncio

    from cube_harness.analyze.judge.driver import TerminalClaudeDriver

    drv = TerminalClaudeDriver()
    result = asyncio.run(
        drv.run(
            system_prompt='reply with a single ```json {"ok": true} ``` block.',
            user_prompt="just say ok",
            cwd=Path.cwd(),
            additional_dirs=[],
            model="claude-sonnet-4-6",
        )
    )
    assert "ok" in result.output_text.lower()


# AgentDriver Protocol check — make sure our fake satisfies the surface.
def test_fake_driver_satisfies_protocol() -> None:
    # Use a structural assert: name + max_parallelism + run + continue_session must exist.
    drv = _FakeDriver("")
    assert isinstance(drv.name, str)
    assert isinstance(drv.max_parallelism, int)
    assert callable(drv.run)
    assert callable(drv.continue_session)
    # And it is accepted as an AgentDriver at type-check time:
    _: AgentDriver = drv  # noqa: F841


# ---------------------------------------------------------------------------
# Meta-analysis: schema round-trip + write_meta_analysis + run_meta_analysis
# ---------------------------------------------------------------------------


_VALID_META_ANALYSIS_JSON = json.dumps(
    {
        "patterns": [
            {
                "name": "inspects-but-never-edits",
                "description": "Agent runs a read tool, sees the bug, but never writes a fix.",
                "affected_trajectories": ["taskA_ep0"],
                "dominant_blame": "model_capability",
            }
        ],
        "root_cause_hypotheses": [
            {
                "description": "The system prompt does not instruct the agent to apply edits before declaring done.",
                "pattern_names": ["inspects-but-never-edits"],
                "confidence": 4,
            }
        ],
        "suggested_interventions": [
            {
                "target": "agent system prompt",
                "change": "Add: 'Apply the fix to disk before submitting an answer.'",
                "rationale": "The dominant pattern is non-action after diagnosis.",
                "confidence": 4,
            }
        ],
        "markdown_summary": (
            "## Patterns\n\n"
            "- inspects-but-never-edits: agent diagnoses without writing a fix.\n\n"
            "## Root causes\n\n"
            "- The system prompt does not force a write step before submission.\n\n"
            "## Suggested interventions\n\n"
            "- Add an explicit edit instruction to the agent system prompt.\n"
        ),
        "notes": None,
    }
)


def test_meta_analysis_schema_roundtrip() -> None:
    from cube_harness.analyze.judge import FailurePattern, MetaAnalysis

    obj = MetaAnalysis(
        experiment_id="exp-1",
        recipe="general_blame",
        driver="claude-code-sdk",
        model="claude-opus-4-7",
        timestamp=1_000_000.0,
        n_episodes_judged=2,
        outcome_distribution={"failure": 2},
        primary_blame_distribution={"model_capability": 2},
        success_rate=0.0,
        patterns=[
            FailurePattern(
                name="x",
                description="d",
                affected_trajectories=["a", "b"],
                dominant_blame="model_capability",
            )
        ],
        markdown_summary="Body.",
    )
    restored = MetaAnalysis.model_validate_json(obj.model_dump_json())
    assert restored == obj


def test_meta_analysis_writer_produces_json_and_md(tmp_path: Path) -> None:
    from cube_harness.analyze.judge import MetaAnalysis, write_meta_analysis

    obj = MetaAnalysis(
        experiment_id="exp-1",
        recipe="general_blame",
        driver="claude-code-sdk",
        model="claude-opus-4-7",
        timestamp=1.0,
        n_episodes_judged=1,
        outcome_distribution={"failure": 1},
        primary_blame_distribution={"agent_scaffolding": 1},
        success_rate=0.0,
        markdown_summary="# Body\n\nSome prose.\n",
    )
    json_path, md_path = write_meta_analysis(tmp_path, obj)
    assert json_path.read_text().startswith("{")
    assert "Body" in md_path.read_text()


def test_meta_analysis_journal_copy(tmp_path: Path) -> None:
    from cube_harness.analyze.judge import MetaAnalysis, copy_to_journal, write_meta_analysis

    obj = MetaAnalysis(
        experiment_id="exp-42",
        recipe="general_blame",
        driver="fake-driver",
        model="claude-opus-4-7",
        timestamp=1.0,
        n_episodes_judged=1,
        outcome_distribution={"failure": 1},
        primary_blame_distribution={"agent_scaffolding": 1},
        success_rate=0.0,
        markdown_summary="body",
    )
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    journal_dir = tmp_path / "journal"
    json_path, md_path = write_meta_analysis(exp_dir, obj)
    j_dst, m_dst = copy_to_journal(journal_dir, "exp-42", json_path, md_path)
    assert j_dst.parent == journal_dir / "exp-42"
    assert j_dst.read_text() == json_path.read_text()
    assert m_dst.read_text() == md_path.read_text()


def test_run_meta_analysis_with_fake_driver(tmp_path: Path) -> None:
    from cube_harness.analyze.judge.meta_analysis import run_meta_analysis

    driver = _FakeDriver(output_text=f"Here is the synthesis:\n```json\n{_VALID_META_ANALYSIS_JSON}\n```")
    results = {
        "taskA_ep0": (_valid_judge_output(), JudgeMetadata(model="claude-sonnet-4-6", timestamp=1.0)),
    }
    analysis = asyncio.run(
        run_meta_analysis(
            experiment_dir=tmp_path,
            experiment_id="exp-1",
            recipe_name="general_blame",
            driver=driver,
            results=results,
            model="claude-opus-4-7",
        )
    )
    assert analysis.experiment_id == "exp-1"
    assert analysis.recipe == "general_blame"
    assert analysis.driver == "fake-driver"
    assert analysis.n_episodes_judged == 1
    assert analysis.patterns[0].name == "inspects-but-never-edits"
    assert "inspects" in analysis.markdown_summary
