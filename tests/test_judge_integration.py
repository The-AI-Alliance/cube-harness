"""Integration tests for judge_experiment: mocked SDK, end-to-end pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import msgpack
import pytest
import zstandard

from cube_harness.analyze.judge import _SDKResult, judge_experiment
from cube_harness.eval_log import EpisodeRecord, JudgeConsensus, UsageSummary

# ---------------------------------------------------------------------------
# Canned SDK result factory (mocks _run_claude_code directly)
# ---------------------------------------------------------------------------


def _make_sdk_result(outcome: str = "failure") -> _SDKResult:
    """Build a canned _SDKResult as if the judge returned a valid JSON block."""
    json_payload = {
        "analysis": "Mock judgment for integration test.",
        "outcome": outcome,
        "summary": "Agent failed.",
        "primary_blame": "model_capability",
        "primary_blame_confidence": 3,
        "other_blames": [],
        "evidence": [{"step": 1, "quote": "ACTION bash"}],
        "hypothesis": "Add loop detection.",
        "hypothesis_confidence": 3,
        "intervention": "scaffold_change",
        "intervention_confidence": 3,
    }
    output_text = "```json\n" + json.dumps(json_payload) + "\n```"
    return _SDKResult(
        output_text=output_text,
        prompt_tokens=1500,
        completion_tokens=200,
        cost_usd=0.05,
        duration_s=2.5,
        actions=[{"tool": "Read", "input": "/x"}],
    )


# ---------------------------------------------------------------------------
# Step file writer
# ---------------------------------------------------------------------------


def _write_step(path: Path, data: dict[str, Any]) -> None:
    cctx = zstandard.ZstdCompressor()
    path.write_bytes(cctx.compress(msgpack.packb(data, use_bin_type=True)))


# ---------------------------------------------------------------------------
# Fixture: minimal experiment dir with 2 episodes
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_experiment(tmp_path: Path) -> Path:
    """Build a minimal experiment dir with 2 episodes, ready to judge."""
    exp = tmp_path / "exp"
    (exp / "episodes").mkdir(parents=True)

    (exp / "experiment_config.json").write_text(
        json.dumps(
            {
                "name": "test-exp",
                "agent_config": {"_type": "fake.agent.Config"},
                "benchmark_config": {"_type": "fake.bench.Config"},
                "infra": {"_type": "fake.infra.Config"},
            }
        )
    )
    (exp / "experiment_summary.json").write_text(json.dumps({"n_completed": 2, "n_correct": 1}))
    # Pre-stage judge_context.md so the pre-judge path is bypassed.
    (exp / "judge_context.md").write_text("# Test context\n\nMocked context.\n")

    for tid, correct in [("task_a_ep0", True), ("task_b_ep1", False)]:
        ep = exp / "episodes" / tid
        (ep / "steps").mkdir(parents=True)
        _write_step(
            ep / "steps" / "000_obs.msgpack.zst",
            {"output": {"obs": {"contents": [{"data": "ACTION bash task description"}]}}},
        )
        _write_step(
            ep / "steps" / "001_act.msgpack.zst",
            {"output": {"actions": [{"name": "bash", "arguments": {"command": "ls"}}]}},
        )
        _write_step(
            ep / "steps" / "002_obs.msgpack.zst",
            {"output": {"obs": {"contents": [{"data": "ls output"}], "done": True}}},
        )
        record = EpisodeRecord(
            evaluation_id="exp-1",
            sample_id=tid.split("_ep")[0],
            sample_hash="h",
            tool_names=["bash"],
            is_correct=correct,
            score=1.0 if correct else 0.0,
            num_turns=2,
            n_agent_steps=1,
            n_env_steps=2,
            wall_time_s=1.0,
            usage=UsageSummary(input_tokens=10, output_tokens=5),
            trajectory_id=tid,
            timestamp=0.0,
        )
        (ep / "episode_record.json").write_text(record.model_dump_json())
        (ep / "episode_config.json").write_text("{}")
    return exp


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_judge_experiment_end_to_end(
    fake_experiment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run judge_experiment with mocked SDK; verify all expected artifacts."""
    sdk_result = _make_sdk_result("failure")

    async def _mock_run_claude_code(**kwargs: Any) -> _SDKResult:
        return sdk_result

    monkeypatch.setattr("cube_harness.analyze.judge._run_claude_code", _mock_run_claude_code)
    monkeypatch.setattr("cube_harness.analyze.judge.collect_source_paths", lambda view: {})

    results = judge_experiment(
        fake_experiment,
        model="claude-sonnet-4-6",
        n=2,
        seed=0,
        n_parallel=1,
        skip_pre_judge=True,
        max_turns=5,
    )

    # 1. Both episodes judged
    assert len(results) == 2

    # 2. Per-episode record has judge_output and judge_metadata
    for ep_dir in sorted((fake_experiment / "episodes").iterdir()):
        record = EpisodeRecord.model_validate_json((ep_dir / "episode_record.json").read_text())
        assert record.judge_output is not None
        assert record.judge_output.outcome.value == "failure"
        assert record.judge_metadata is not None
        assert record.judge_metadata.cost_usd > 0

    # 3. judge_trace.json sidecar written (actions from mock result)
    for ep_dir in sorted((fake_experiment / "episodes").iterdir()):
        trace = json.loads((ep_dir / "judge_trace.json").read_text())
        assert trace["trace_mode"] == "actions"
        assert any(a["tool"] == "Read" for a in trace["actions"])

    # 4. Aggregate summary + CSV written
    summary = json.loads((fake_experiment / "experiment_judge_summary.json").read_text())
    assert summary["n_judged"] == 2
    assert "interventions" in summary
    csv_path = fake_experiment / "experiment_judge_report.csv"
    assert csv_path.exists()
    csv_text = csv_path.read_text()
    assert "outcome" in csv_text
    assert "intervention" in csv_text

    # 5. _judge_transcript cleaned up — no leftover steps/ or summary
    for ep_dir in sorted((fake_experiment / "episodes").iterdir()):
        td = ep_dir / "_judge_transcript"
        assert not (td / "steps").exists()
        assert not (td / "episode_summary.txt").exists()


def test_judge_experiment_with_n_judges_writes_consensus(
    fake_experiment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K=3 judges per episode → consensus + judge_outputs.jsonl sidecar.

    Mocks _judge_episode_impl to avoid transcript-directory conflicts when
    K concurrent coroutines run on the same episode dir.
    """
    from cube_harness.eval_log import JudgeMetadata, JudgeOutput

    call_count = 0

    async def _mock_judge_episode_impl(*args: Any, **kwargs: Any) -> tuple[JudgeOutput, JudgeMetadata, list[Any]]:
        nonlocal call_count
        call_count += 1
        out = JudgeOutput(
            analysis="mock",
            outcome="failure",
            summary="fail",
            primary_blame="model_capability",
            primary_blame_confidence=3,
            other_blames=[],
            evidence=[{"step": 1, "quote": "ACTION bash"}],
            hypothesis="add loop detection",
            hypothesis_confidence=2,
            intervention="scaffold_change",
            intervention_confidence=2,
        )
        meta = JudgeMetadata(model="m", timestamp=0.0, cost_usd=0.01, duration_s=0.5)
        return out, meta, [{"tool": "Read", "input": "/x"}]

    monkeypatch.setattr("cube_harness.analyze.judge._judge_episode_impl", _mock_judge_episode_impl)
    monkeypatch.setattr("cube_harness.analyze.judge.collect_source_paths", lambda view: {})

    judge_experiment(
        fake_experiment,
        model="m",
        n=1,
        n_parallel=1,
        skip_pre_judge=True,
        max_turns=5,
        n_judges=3,
    )

    ep_dir = sorted((fake_experiment / "episodes").iterdir())[0]
    record = EpisodeRecord.model_validate_json((ep_dir / "episode_record.json").read_text())
    assert record.judge_consensus is not None
    assert isinstance(record.judge_consensus, JudgeConsensus)
    assert record.judge_consensus.n_judges == 3
    assert record.judge_consensus.outcome_agreement == 1.0  # all canned same
    sidecar_lines = (ep_dir / "judge_outputs.jsonl").read_text().splitlines()
    assert len(sidecar_lines) == 3
