"""Tests for evidence quote verification (difflib-based fuzzy match)."""

from __future__ import annotations

from pathlib import Path

from cube_harness.analyze.judge import (
    _find_quote_in_text,
    _normalize_for_match,
    _verify_evidence,
)
from cube_harness.eval_log import JudgeOutput

# ---------------------------------------------------------------------------
# Local copy of _valid_judge_output (avoids cross-test-file import issues)
# ---------------------------------------------------------------------------


def _valid_judge_output(**overrides: object) -> JudgeOutput:
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


# ---------------------------------------------------------------------------
# _normalize_for_match
# ---------------------------------------------------------------------------


def test_normalize_collapses_whitespace_and_folds_case() -> None:
    assert _normalize_for_match("Hello   WORLD\n\t!") == "hello world !"


def test_normalize_empty_string() -> None:
    assert _normalize_for_match("") == ""


# ---------------------------------------------------------------------------
# _find_quote_in_text
# ---------------------------------------------------------------------------


def test_find_quote_exact_match_returns_true_none() -> None:
    verified, ratio = _find_quote_in_text("the agent failed", "Step 5: the agent failed here.")
    assert verified is True
    assert ratio is None


def test_find_quote_exact_match_is_case_insensitive() -> None:
    verified, _ = _find_quote_in_text("THE AGENT", "the agent failed")
    assert verified is True


def test_find_quote_fuzzy_match_with_typo_returns_true() -> None:
    """One-character typo should still match above threshold."""
    verified, ratio = _find_quote_in_text(
        "the agnet failed",  # typo: agnet
        "Step 5: the agent failed here.",
    )
    assert verified is True
    assert ratio is not None and ratio >= 0.85


def test_find_quote_completely_different_returns_false() -> None:
    verified, ratio = _find_quote_in_text(
        "fixed the bug successfully in all test cases",
        "Step 5: completely unrelated content that has nothing in common with the query.",
    )
    assert verified is False
    assert ratio is not None and ratio < 0.85


def test_find_quote_empty_quote_returns_false() -> None:
    verified, ratio = _find_quote_in_text("", "some text here")
    assert verified is False
    assert ratio is None


# ---------------------------------------------------------------------------
# _verify_evidence
# ---------------------------------------------------------------------------


def test_verify_evidence_marks_unverified_and_downgrades(tmp_path: Path) -> None:
    transcript_dir = tmp_path / "_judge_transcript"
    steps = transcript_dir / "steps"
    steps.mkdir(parents=True)
    (steps / "005_act.txt").write_text("ACTION bash:\n  command: ls /testbed")

    out = _valid_judge_output(
        evidence=[
            {"step": 5, "quote": "command: ls /testbed"},  # exact match
            {"step": 5, "quote": "this never appeared anywhere"},  # bad quote
        ],
        primary_blame_confidence=4,
    )
    _verify_evidence(out, transcript_dir)
    assert out.evidence[0].verified is True
    assert out.evidence[1].verified is False
    assert out.primary_blame_confidence == 3  # downgraded by 1


def test_verify_evidence_noop_when_no_evidence() -> None:
    out = _valid_judge_output(evidence=[], primary_blame="none", primary_blame_confidence=2)
    _verify_evidence(out, Path("/nonexistent"))
    assert out.primary_blame_confidence == 2  # unchanged


def test_verify_evidence_handles_missing_step_file(tmp_path: Path) -> None:
    transcript_dir = tmp_path / "_judge_transcript"
    (transcript_dir / "steps").mkdir(parents=True)
    out = _valid_judge_output(
        evidence=[{"step": 999, "quote": "anything at all"}],
        primary_blame_confidence=3,
    )
    _verify_evidence(out, transcript_dir)
    assert out.evidence[0].verified is False  # step file doesn't exist
    assert out.primary_blame_confidence == 2  # downgraded


def test_verify_evidence_sets_match_ratio_for_fuzzy(tmp_path: Path) -> None:
    """When a fuzzy (not exact) match is found, match_ratio should be set."""
    transcript_dir = tmp_path / "_judge_transcript"
    steps = transcript_dir / "steps"
    steps.mkdir(parents=True)
    (steps / "003_obs.txt").write_text("Step 3: the agent failed here due to timeout.")

    out = _valid_judge_output(
        evidence=[{"step": 3, "quote": "the agnet failed here"}],  # typo in quote
        primary_blame_confidence=3,
    )
    _verify_evidence(out, transcript_dir)
    item = out.evidence[0]
    assert item.verified is True
    assert item.match_ratio is not None
    assert item.match_ratio >= 0.85


def test_verify_evidence_all_verified_no_downgrade(tmp_path: Path) -> None:
    """When all quotes verify, confidence should not be downgraded."""
    transcript_dir = tmp_path / "_judge_transcript"
    steps = transcript_dir / "steps"
    steps.mkdir(parents=True)
    (steps / "001_act.txt").write_text("ACTION bash:\n  command: pytest /testbed")

    out = _valid_judge_output(
        evidence=[{"step": 1, "quote": "command: pytest /testbed"}],
        primary_blame_confidence=4,
    )
    _verify_evidence(out, transcript_dir)
    assert out.evidence[0].verified is True
    assert out.primary_blame_confidence == 4  # unchanged
