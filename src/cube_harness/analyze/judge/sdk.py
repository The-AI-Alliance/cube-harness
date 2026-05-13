"""Helpers shared between the judge and its drivers.

The Claude SDK invocation moved to `driver.ClaudeCodeSDKDriver`. This module
keeps the shape constants (`TraceMode`, `JUDGE_ALLOWED_TOOLS`) — drivers depend
on them but `__init__.py` re-exports them for backwards compatibility — and
two small helpers used by both drivers and tests:

- `_extract_json_block` — pulls the judge's final JSON object out of free text.
- `_summarise_tool_input` — one-line tool-call rendering for traces.

A deprecated `_SDKResult` dataclass alias is also kept — the integration test in
`tests/test_judge.py` patches `_run_claude_code` and expects this shape; the new
flow uses `DriverResult` instead.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Tools the judge needs: read transcript files, grep through cube/agent source,
# inspect screenshots if any. No write/edit — the judge only produces a JSON answer.
JUDGE_ALLOWED_TOOLS: tuple[str, ...] = ("Read", "Glob", "Grep", "Bash")

TraceMode = Literal["actions", "full", "off"]

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


@dataclass
class _SDKResult:
    """Deprecated — kept for one release window so existing tests that mock
    `_run_claude_code` continue to work. New code should use
    `cube_harness.analyze.judge.driver.DriverResult`."""

    output_text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    duration_s: float
    actions: list[dict[str, Any]] = field(default_factory=list)


__all__ = [
    "_extract_json_block",
    "_summarise_tool_input",
    "_SDKResult",
    "TraceMode",
    "JUDGE_ALLOWED_TOOLS",
]
