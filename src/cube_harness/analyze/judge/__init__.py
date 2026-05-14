"""Trajectory judge — post-hoc LLM analysis of cube-harness episodes.

Reads a completed experiment directory, decompresses the `*.msgpack.zst` step files
into a readable transcript, and invokes a coding-agent driver (Claude Code SDK by
default; terminal `claude -p` available) to produce a structured judgment per
episode.

The default judge recipe is `general_blame` — its on-disk shape is identical to
the legacy `JudgeOutput` (now `BaseJudgeOutput`). Other recipes (`profiling`,
`agent_scaffolding`) live under `use_cases/` and ship typed extensions of the
base shape.

Results are written into each episode's `episode_record.json` (sibling fields
`judge_output` and `judge_metadata`), aggregated into
`<experiment_dir>/experiment_judge_summary.json`,
`<experiment_dir>/experiment_judge_report.csv`, and
`<experiment_dir>/experiment_judge_report.json`. Cross-experiment aggregation
lives in `cube_harness.analyze.cross_experiment`.

CLI: `ch-judge <experiment_dir> [options]` — see `cli.py` for flags.
"""

from cube_harness.analyze.judge.audit import (
    AUDIT_FILENAME,
    AUDIT_PROMPT,
    AuditOutput,
    BlameAlternative,
    run_audit_pass,
)
from cube_harness.analyze.judge.benchmark_context_agent import generate_context_file
from cube_harness.analyze.judge.cli import main
from cube_harness.analyze.judge.context import (
    JUDGE_CONTEXT_FILENAME,
    find_default_context_file,
    validate_context_file,
)
from cube_harness.analyze.judge.core import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_FRACTION,
    EXPERIMENT_JUDGE_REPORT_FILENAME,
    EXPERIMENT_JUDGE_REPORT_JSON_FILENAME,
    EXPERIMENT_JUDGE_SUMMARY_FILENAME,
    JudgeBatchConfig,
    judge_episode,
    judge_experiment,
    persist_judgment,
    write_csv_report,
    write_summary,
)
from cube_harness.analyze.judge.driver import (
    JUDGE_ALLOWED_TOOLS,
    AgentDriver,
    ClaudeCodeSDKDriver,
    DriverResult,
    TerminalClaudeDriver,
    ToolAction,
    TraceMode,
)
from cube_harness.analyze.judge.parse import extract_json_block
from cube_harness.analyze.judge.recipe import BaseJudgeOutput, JudgeRecipe, get_default_recipe
from cube_harness.analyze.judge.selection import (
    EpisodeRef,
    SameAgentPreviousIteration,
    SameTaskDifferentAgent,
    Selector,
    TopKBySimilarityStub,
    discover_episodes,
    load_episode_record,
    select_episodes,
)
from cube_harness.analyze.judge.transcript import extract_transcript
from cube_harness.analyze.judge.use_cases import RECIPE_CATALOG

# Eager default. By the time this module-level statement runs, every submodule
# (audit, benchmark_context_agent, core, driver, recipe, selection, use_cases)
# has loaded, so resolving the catalog here is cycle-free. Prior versions used
# a lazy callable to defer the use_cases import; that's no longer necessary.
DEFAULT_RECIPE: JudgeRecipe = get_default_recipe()


__all__ = [
    # Public API — judge
    "judge_episode",
    "judge_experiment",
    "JudgeBatchConfig",
    "discover_episodes",
    "select_episodes",
    "extract_transcript",
    # Public API — recipes
    "JudgeRecipe",
    "BaseJudgeOutput",
    "RECIPE_CATALOG",
    "DEFAULT_RECIPE",
    "get_default_recipe",
    # Public API — drivers
    "AgentDriver",
    "ClaudeCodeSDKDriver",
    "TerminalClaudeDriver",
    "DriverResult",
    "ToolAction",
    "TraceMode",
    "JUDGE_ALLOWED_TOOLS",
    # Public API — selectors
    "Selector",
    "SameTaskDifferentAgent",
    "SameAgentPreviousIteration",
    "TopKBySimilarityStub",
    "EpisodeRef",
    # Public API — context file
    "validate_context_file",
    "find_default_context_file",
    "JUDGE_CONTEXT_FILENAME",
    "generate_context_file",
    # Public API — audit
    "AuditOutput",
    "BlameAlternative",
    "AUDIT_FILENAME",
    "AUDIT_PROMPT",
    "run_audit_pass",
    # Constants
    "DEFAULT_MODEL",
    "DEFAULT_SAMPLE_FRACTION",
    "EXPERIMENT_JUDGE_SUMMARY_FILENAME",
    "EXPERIMENT_JUDGE_REPORT_FILENAME",
    "EXPERIMENT_JUDGE_REPORT_JSON_FILENAME",
    # Semi-private helpers accessed by judge_report.py and tests
    "load_episode_record",
    "write_summary",
    "write_csv_report",
    "persist_judgment",
    "extract_json_block",
    # CLI entry point
    "main",
]
