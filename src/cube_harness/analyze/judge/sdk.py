"""Claude Code SDK invocation for the trajectory judge."""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Tools the judge needs: read transcript files, grep through cube/agent source,
# inspect screenshots if any. No write/edit — the judge only produces a JSON answer.
JUDGE_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Bash"]

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


TraceMode = Literal["actions", "full", "off"]


@dataclass
class _SDKResult:
    output_text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    duration_s: float
    actions: list[dict[str, Any]] = field(default_factory=list)


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


async def _run_claude_code(
    *,
    system_prompt: str,
    user_prompt: str,
    cwd: Path,
    additional_dirs: list[Path],
    model: str,
    verbose: bool = False,
    trace_mode: TraceMode = "actions",
) -> _SDKResult:
    """Invoke Claude Code via the SDK and return the assistant text + usage.

    When `verbose=True`, stream a one-line summary of each tool call and assistant
    text chunk to stderr as they arrive — useful to see what the judge is doing
    without waiting for the final JSON.
    """
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            query,
        )
    except ImportError as e:
        raise RuntimeError("claude-agent-sdk not installed. Run: pip install 'cube-harness[judge]'") from e

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=JUDGE_ALLOWED_TOOLS,
        permission_mode="bypassPermissions",
        cwd=str(cwd),
        add_dirs=[str(p) for p in additional_dirs],
        model=model,
        include_partial_messages=False,
    )
    if verbose:
        logger.info("Running judge with model %s and options: %r", model, options)

    final_text: list[str] = []
    collected_actions: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    cost_usd = 0.0
    duration_ms = 0
    start = time.time()

    def _emit(line: str) -> None:
        # Verbose progress goes to stderr so stdout stays parseable for `--summary`.
        print(line, file=sys.stderr, flush=True)

    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    final_text.append(block.text)
                    if verbose and block.text.strip():
                        first_line = block.text.strip().splitlines()[0][:140]
                        _emit(f"  · {first_line}")
                elif isinstance(block, ToolUseBlock):
                    if verbose:
                        _emit(f"  > {block.name}({_summarise_tool_input(block.name, block.input)})")
                    if trace_mode == "actions":
                        collected_actions.append(
                            {"tool": block.name, "input": _summarise_tool_input(block.name, block.input)}
                        )
                    elif trace_mode == "full":
                        collected_actions.append(
                            {
                                "tool": block.name,
                                "input": _summarise_tool_input(block.name, block.input),
                                "raw_input": block.input
                                if isinstance(block.input, dict)
                                else {"value": str(block.input)},
                            }
                        )
        elif isinstance(message, ResultMessage):
            usage = getattr(message, "usage", None) or {}
            prompt_tokens = (
                int(usage.get("input_tokens", 0) or 0)
                + int(usage.get("cache_read_input_tokens", 0) or 0)
                + int(usage.get("cache_creation_input_tokens", 0) or 0)
            )
            completion_tokens = int(usage.get("output_tokens", 0) or 0)
            cost_usd = float(getattr(message, "total_cost_usd", 0.0) or 0.0)
            duration_ms = int(getattr(message, "duration_ms", 0) or 0)

    duration_s = duration_ms / 1000.0 if duration_ms else (time.time() - start)
    return _SDKResult(
        output_text="\n".join(final_text),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        duration_s=duration_s,
        actions=collected_actions if trace_mode != "off" else [],
    )
