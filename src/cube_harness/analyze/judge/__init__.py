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

CLI: `ch-judge <experiment_dir> [options]`
"""

from cube_harness.analyze.judge.cli import main
from cube_harness.analyze.judge.core import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_FRACTION,
    EXPERIMENT_JUDGE_REPORT_FILENAME,
    EXPERIMENT_JUDGE_SUMMARY_FILENAME,
    _persist_judgment,
    _write_csv_report,
    _write_summary,
    judge_episode,
    judge_experiment,
)
from cube_harness.analyze.judge.sdk import TraceMode, _extract_json_block
from cube_harness.analyze.judge.selection import EpisodeRef, _load_episode_record, discover_episodes, select_episodes
from cube_harness.analyze.judge.transcript import extract_transcript

__all__ = [
    # Public API
    "judge_episode",
    "judge_experiment",
    "discover_episodes",
    "select_episodes",
    "extract_transcript",
    "EpisodeRef",
    "TraceMode",
    # Constants (used by judge_report.py and other consumers)
    "DEFAULT_MODEL",
    "DEFAULT_SAMPLE_FRACTION",
    "EXPERIMENT_JUDGE_SUMMARY_FILENAME",
    "EXPERIMENT_JUDGE_REPORT_FILENAME",
    # Semi-private helpers accessed by judge_report.py
    "_load_episode_record",
    "_write_summary",
    "_write_csv_report",
    "_persist_judgment",
    "_extract_json_block",
    # CLI entry point
    "main",
]
