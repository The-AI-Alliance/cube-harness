"""Pure utility functions for the AgentLab2 XRay viewer.

All functions in this module are pure (or near-pure) — no Gradio imports, no global state.
This makes them independently testable without any UI framework.
"""

import html as html_lib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel

from agentlab2.core import AgentOutput, EnvironmentOutput, Trajectory, TrajectoryStep


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Examples: 800ms, 4.2s, 3m 12s, 1h 5m
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_directory_contents(results_dir: Path) -> list[str]:
    """Return sorted list of experiment directory names with trajectory counts.

    Returns ["Select experiment directory"] + names sorted most-recent first.
    Only includes directories that contain a 'trajectories/' subdirectory.
    """
    sentinel = "Select experiment directory"
    if not results_dir or not results_dir.exists():
        return [sentinel]

    exp_descriptions = []
    for dir_path in results_dir.iterdir():
        if not dir_path.is_dir():
            continue
        traj_dir = dir_path / "trajectories"
        if not traj_dir.exists():
            continue
        n_trajs = len(list(traj_dir.glob("*.jsonl")))
        exp_descriptions.append(f"{dir_path.name} ({n_trajs} trajectories)")

    return [sentinel] + sorted(exp_descriptions, reverse=True)


# ---------------------------------------------------------------------------
# Screenshot extraction
# ---------------------------------------------------------------------------


def get_screenshot_from_step(step: EnvironmentOutput | AgentOutput | None) -> Image.Image | None:
    """Extract the first PIL Image from an EnvironmentOutput's observation contents.

    Returns None if no image is found, step is None, or step is an AgentOutput.
    """
    if not isinstance(step, EnvironmentOutput):
        return None
    for content in step.obs.contents:
        if isinstance(content.data, Image.Image):
            return content.data
    return None


def get_current_screenshot(
    step: EnvironmentOutput | AgentOutput | None,
    prev_step: EnvironmentOutput | AgentOutput | None,
) -> Image.Image | None:
    """Get the best screenshot for the current step.

    If current step is EnvironmentOutput: return its screenshot.
    If current step is AgentOutput: fall back to screenshot from prev_step.
    """
    img = get_screenshot_from_step(step)
    if img is None and prev_step is not None:
        img = get_screenshot_from_step(prev_step)
    return img


def get_all_screenshots(trajectory: Trajectory) -> list[tuple[int, Image.Image]]:
    """Collect all (step_index, Image) pairs from all EnvironmentOutputs in a trajectory."""
    result = []
    for i, traj_step in enumerate(trajectory.steps):
        img = get_screenshot_from_step(traj_step.output)
        if img is not None:
            result.append((i, img))
    return result


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------


def extract_obs_content(step: EnvironmentOutput | None, name_pattern: str) -> str | None:
    """Find text content in EnvironmentOutput.obs.contents by name substring match.

    Performs case-insensitive substring match against content.name.
    Returns the first matching str content.data, or None if not found.

    Examples:
        extract_obs_content(step, "axtree")  -> accessibility tree text
        extract_obs_content(step, "html")    -> page HTML
        extract_obs_content(step, "pruned")  -> pruned HTML
    """
    if not isinstance(step, EnvironmentOutput):
        return None
    pattern_lower = name_pattern.lower()
    for content in step.obs.contents:
        if isinstance(content.data, str) and pattern_lower in (content.name or "").lower():
            return content.data
    return None


# ---------------------------------------------------------------------------
# LLM prompt / chat rendering
# ---------------------------------------------------------------------------


_COLLAPSE_THRESHOLD = 2000  # chars (~20 lines) — messages longer than this start collapsed


def _msg_to_dict(msg: object) -> dict:
    """Normalise a message to a plain dict."""
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if hasattr(msg, "__dict__"):
        return dict(msg.__dict__)
    return {"role": "unknown", "content": str(msg)}


def _preview(text: str, max_chars: int = 80) -> str:
    """Return first non-empty line of text, truncated to max_chars."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:max_chars] + ("…" if len(line) > max_chars else "")
    return ""


def _details_block(label: str, body: str, icon: str = "📄") -> str:
    """Wrap body in a <details> block. Short content is open by default.

    Summary shows: icon + label + first-line preview (when collapsed).
    """
    open_attr = " open" if len(body) <= _COLLAPSE_THRESHOLD else ""
    preview = _preview(body)
    preview_html = f" <span style='color:#888;font-weight:normal'>{html_lib.escape(preview)}</span>" if preview and not open_attr else ""
    escaped = html_lib.escape(body)
    return (
        f"<details{open_attr}>"
        f"<summary>{icon} <strong>{html_lib.escape(label)}</strong>{preview_html}</summary>"
        f"<pre style='white-space:pre-wrap;overflow-wrap:anywhere;margin:4px 0'>{escaped}</pre>"
        f"</details>\n"
    )


def _render_text_content(text: str) -> str:
    """Render a plain text content string.

    Handles the '##name\\nbody' convention used by Content.to_message() for named
    text/dict content, but also works for any plain string.
    """
    if text.startswith("##"):
        newline = text.find("\n")
        if newline != -1:
            name = text[2:newline].strip()
            body = text[newline + 1:]
            return _details_block(name, body)
    return _details_block("text", text)


def _render_content_items(content: str | list | None) -> str:
    """Render a message's content field as HTML.

    Handles the common content types found in LLM message dicts:
      - str:            plain text or '##name\\nbody' encoded text
      - list of items:  multimodal content list with typed items:
          {"type": "text",      "text": ...}
          {"type": "image_url", "image_url": {"url": ...}}
          {"type": "image",     "url": ...}          # alternate image format
          {"type": "audio",     ...}                 # future / other modalities
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return _render_text_content(content)

    # Multimodal list — iterate items, grouping a text label with a following image
    parts: list[str] = []
    items = [i for i in content if isinstance(i, dict)]
    idx = 0
    while idx < len(items):
        item = items[idx]
        item_type = item.get("type", "")
        next_item = items[idx + 1] if idx + 1 < len(items) else None

        if item_type == "text":
            text = item.get("text", "")
            # If the next item is an image, this text is a label for it
            if next_item is not None and next_item.get("type") in ("image_url", "image"):
                url = next_item.get("image_url", {}).get("url", "") or next_item.get("url", "")
                img = f"<img src='{url}' style='max-width:100%;border-radius:4px;margin:4px 0'>"
                parts.append(f"<details open><summary>📷 <strong>{html_lib.escape(text or 'screenshot')}</strong></summary>{img}</details>\n")
                idx += 2
            else:
                parts.append(_render_text_content(text))
                idx += 1
        elif item_type in ("image_url", "image"):
            url = item.get("image_url", {}).get("url", "") or item.get("url", "")
            img = f"<img src='{url}' style='max-width:100%;border-radius:4px;margin:4px 0'>"
            parts.append(f"<details open><summary>📷 <strong>screenshot</strong></summary>{img}</details>\n")
            idx += 1
        else:
            # Unknown / future type — show type name as a placeholder
            parts.append(f"<em>[{html_lib.escape(item_type)}]</em>\n")
            idx += 1

    return "".join(parts)


def _render_assistant_content(msg: dict) -> str:
    """Render assistant message: text content + tool calls as HTML."""
    parts: list[str] = []
    content = msg.get("content") or ""
    if content:
        parts.append(_details_block("reasoning", str(content), icon="💭"))
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            tc = tc.model_dump() if hasattr(tc, "model_dump") else vars(tc)
        fn = tc.get("function", {})
        name = fn.get("name", "?")
        args = fn.get("arguments", "")
        if isinstance(args, str):
            try:
                args = json.dumps(json.loads(args), indent=2)
            except (json.JSONDecodeError, ValueError):
                pass
        parts.append(_details_block(f"tool call: {name}", str(args), icon="🔧"))
    return "".join(parts)


_ROLE_STYLE = {
    "system": "background:#f0f4ff;border-left:3px solid #6c8ebf",
    "user": "background:#f5f5f5;border-left:3px solid #aaa",
    "tool": "background:#fff8e7;border-left:3px solid #e6a817",
    "assistant": "background:#f0fff4;border-left:3px solid #5cb85c",
}


def get_chat_messages_html(step: EnvironmentOutput | AgentOutput | None) -> str:
    """Render the full LLM conversation (prompt + response) as HTML for one agent step.

    Each message is a collapsible <details> block — short content is expanded by default,
    long content (axtree, HTML) starts collapsed. Screenshots render as inline images.
    Returns empty string for non-AgentOutput steps or steps with no llm_calls.
    """
    if not isinstance(step, AgentOutput) or not step.llm_calls:
        return ""

    llm_call = step.llm_calls[0]
    messages = list(llm_call.prompt.messages) + [llm_call.output]
    blocks: list[str] = []

    for i, msg in enumerate(messages):
        msg_dict = _msg_to_dict(msg)
        role = msg_dict.get("role", "unknown")
        tool_call_id = msg_dict.get("tool_call_id")

        label = f"[{i + 1}] {role}"
        if tool_call_id:
            label += f" · tool_result for {tool_call_id}"

        if role == "assistant":
            body_html = _render_assistant_content(msg_dict)
        else:
            body_html = _render_content_items(msg_dict.get("content"))

        style = _ROLE_STYLE.get(role, "background:#fafafa;border-left:3px solid #ccc")
        blocks.append(
            f"<div style='margin:6px 0;padding:8px 12px;border-radius:4px;{style}'>"
            f"<strong>{html_lib.escape(label)}</strong><br>{body_html}</div>\n"
        )

    return "".join(blocks)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters with a trailing indicator."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated]"


# ---------------------------------------------------------------------------
# Step detail markdown
# ---------------------------------------------------------------------------


def get_step_details_markdown(
    step: EnvironmentOutput | AgentOutput | None,
    traj_step: TrajectoryStep | None,
) -> str:
    """Produce a context-aware markdown summary of the current step."""
    if step is None:
        return "No step selected"

    duration_info = ""
    if traj_step and traj_step.start_time is not None and traj_step.end_time is not None:
        duration = traj_step.end_time - traj_step.start_time
        duration_info = f" │ ⏱️ {format_duration(duration)}"

    if isinstance(step, EnvironmentOutput):
        return _format_env_step_details(step, duration_info)
    elif isinstance(step, AgentOutput):
        return _format_agent_step_details(step, duration_info)
    return "Unknown step type"


def _format_env_step_details(step: EnvironmentOutput, duration_info: str) -> str:
    """Format EnvironmentOutput details as markdown."""
    sections = [f"## 🌍 Environment Output{duration_info}\n"]

    if step.done:
        status = "✅ **Success**" if step.reward > 0 else "❌ **Failed**"
        sections.append(f"**Status:** {status} │ **Reward:** {step.reward:.2f}\n")
    else:
        sections.append(f"**Reward:** {step.reward:.2f} │ **Done:** No\n")

    for content in step.obs.contents:
        if isinstance(content.data, str):
            name = content.name or "Content"
            data = _truncate(content.data, 200000)
            sections.append(f"### {name}\n```\n{data}\n```\n")
        elif isinstance(content.data, Image.Image):
            sections.append(f"**{content.name or 'Screenshot'}:** {content.data.size[0]}x{content.data.size[1]}\n")
        elif isinstance(content.data, (dict, list)):
            name = content.name or "Data"
            data_str = json.dumps(content.data, indent=2)
            data_str = _truncate(data_str, 100000)
            sections.append(f"### {name}\n```json\n{data_str}\n```\n")
        elif isinstance(content.data, BaseModel):
            name = content.name or "Data"
            data_str = content.data.model_dump_json(indent=2)
            data_str = _truncate(data_str, 100000)
            sections.append(f"### {name}\n```json\n{data_str}\n```\n")

    if step.info.get("error"):
        sections.append(f"\n### ⚠️ Error\n```\n{step.info['error']}\n```\n")

    return "\n".join(sections)


def _format_agent_step_details(step: AgentOutput, duration_info: str) -> str:
    """Format AgentOutput details as markdown."""
    sections = [f"## 🤖 Agent Output{duration_info}\n"]

    if step.llm_calls:
        llm_call = step.llm_calls[0]
        usage = llm_call.usage
        if usage and usage.prompt_tokens > 0:
            token_parts = [f"📊 **Tokens:** prompt: {usage.prompt_tokens:,}"]
            token_parts.append(f"completion: {usage.completion_tokens:,}")
            if usage.cached_tokens > 0:
                cache_pct = usage.cached_tokens / usage.prompt_tokens * 100
                token_parts.append(f"cached: {usage.cached_tokens:,} ({cache_pct:.0f}%)")
            if usage.cache_creation_tokens > 0:
                token_parts.append(f"cache_created: {usage.cache_creation_tokens:,}")
            if usage.cost > 0:
                token_parts.append(f"💰 **${usage.cost:.4f}**")
            sections.append(" │ ".join(token_parts) + "\n")

    if step.actions:
        sections.append("### Actions\n")
        for i, action in enumerate(step.actions):
            args_str = json.dumps(action.arguments, indent=2)
            sections.append(f"**{i + 1}. {action.name}**\n```json\n{args_str}\n```\n")
    else:
        sections.append("*No actions taken*\n")

    if step.llm_calls:
        llm_call = step.llm_calls[0]
        if llm_call.output:
            msg = llm_call.output
            content = getattr(msg, "content", None)
            if content:
                reasoning = _truncate(str(content), 150000)
                sections.append(f"### Agent Reasoning\n{reasoning}\n")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Always-visible step summary (task goal + agent action)
# ---------------------------------------------------------------------------


def get_task_goal(trajectory: Trajectory | None) -> str:
    """Extract the task goal text from the first EnvironmentOutput's text content.

    The goal is typically the first text content item in the first env step,
    set by Task.setup() via Observation.from_text(goal).
    """
    if trajectory is None:
        return "*No trajectory loaded*"
    if not trajectory.steps:
        return "*No goal text found*"
    for ts in trajectory.steps:
        if isinstance(ts.output, EnvironmentOutput):
            for content in ts.output.obs.contents:
                if isinstance(content.data, str) and content.data.strip():
                    return content.data
    return "*No goal text found*"


def get_agent_action_markdown(agent_out: AgentOutput | None) -> str:
    """Return a compact markdown summary of the agent's actions as function call syntax.

    Renders each action as: `name(key="value", key2=123)`
    Long string values are truncated to 200 chars.
    Returns a placeholder for terminal steps.
    """
    if agent_out is None:
        return "*Terminal step — no agent action*"
    if not agent_out.actions:
        return "*No actions taken*"
    parts = []
    for action in agent_out.actions:
        args_parts = []
        for k, v in (action.arguments or {}).items():
            if isinstance(v, str):
                v_display = v if len(v) <= 200 else v[:200] + "…"
                args_parts.append(f'{k}="{v_display}"')
            else:
                args_parts.append(f"{k}={v!r}")
        call_str = f"{action.name}({', '.join(args_parts)})"
        parts.append(f"`{call_str}`")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Paired step rendering (env + agent shown together)
# ---------------------------------------------------------------------------


def get_paired_step_details_markdown(
    env_out: EnvironmentOutput | None,
    agent_out: AgentOutput | None,
    env_ts: TrajectoryStep | None,
    agent_ts: TrajectoryStep | None,
) -> str:
    """Produce a combined markdown summary showing both env observation and agent action."""
    if env_out is None:
        return "No step selected"

    env_duration = ""
    if env_ts and env_ts.start_time is not None and env_ts.end_time is not None:
        env_duration = f" │ ⏱️ {format_duration(env_ts.end_time - env_ts.start_time)}"

    env_section = _format_env_step_details(env_out, env_duration)

    if agent_out is None:
        agent_section = "\n---\n\n## 🤖 Agent Action\n\n*No agent action — terminal observation.*\n"
    else:
        agent_duration = ""
        if agent_ts and agent_ts.start_time is not None and agent_ts.end_time is not None:
            agent_duration = f" │ ⏱️ {format_duration(agent_ts.end_time - agent_ts.start_time)}"
        agent_section = "\n---\n\n" + _format_agent_step_details(agent_out, agent_duration)

    return env_section + agent_section


def get_paired_error_markdown(
    env_out: EnvironmentOutput | None,
    agent_out: AgentOutput | None,
) -> str:
    """Show errors from both the env output and the agent output for this UI step."""
    parts = []

    if env_out is not None:
        env_err = _extract_error_markdown(env_out, "Environment")
        if env_err:
            parts.append(env_err)

    if agent_out is not None:
        agent_err = _extract_error_markdown(agent_out, "Agent")
        if agent_err:
            parts.append(agent_err)

    return "\n\n---\n\n".join(parts) if parts else "No errors in this step"


def _extract_error_markdown(step: EnvironmentOutput | AgentOutput, label: str) -> str:
    """Extract error string from a single step, returning empty string if none."""
    if step.error is not None:
        err = step.error
        return (
            f"### ⚠️ {label}: {err.error_type}\n"
            f"**Message:** {err.exception_str}\n\n"
            f"**Stack Trace:**\n```\n{err.stack_trace}\n```"
        )
    if isinstance(step, EnvironmentOutput):
        info_error = step.info.get("error")
        if info_error:
            return f"### ⚠️ {label} error (from info)\n```\n{info_error}\n```"
    return ""


# ---------------------------------------------------------------------------
# Error & logs (legacy single-step, kept for backward compatibility)
# ---------------------------------------------------------------------------


def get_step_error_markdown(step: EnvironmentOutput | AgentOutput | None) -> str:
    """Extract error information from a single step as markdown.

    Checks step.error (StepError) for both step types.
    Also checks EnvironmentOutput.info.get('error') as a fallback.
    """
    if step is None:
        return "No errors in this step"
    result = _extract_error_markdown(step, "Error")
    return result if result else "No errors in this step"


def get_step_logs_markdown(
    step: EnvironmentOutput | AgentOutput | None,
    traj: Trajectory | None,
) -> str:
    """Extract log information from a step and trajectory metadata.

    Shows EnvironmentOutput.info entries (excluding 'error' and 'message') and trajectory metadata.
    """
    parts = []

    if isinstance(step, EnvironmentOutput) and step.info:
        log_entries = {k: v for k, v in step.info.items() if k not in ("error", "message")}
        if log_entries:
            parts.append("### Step Info\n")
            for k, v in log_entries.items():
                parts.append(f"**{k}**: `{v}`\n")

    if traj and traj.metadata:
        meta_str = json.dumps(traj.metadata, indent=2)
        parts.append(f"\n### Trajectory Metadata\n```json\n{meta_str}\n```\n")

    return "\n".join(parts) if parts else "No log information available."


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _compute_token_stats_for_trajectory(traj: Trajectory) -> dict[str, int | float]:
    """Sum token usage across all AgentOutput LLM calls in one trajectory."""
    stats: dict[str, int | float] = {
        "prompt": 0,
        "completion": 0,
        "cached": 0,
        "cache_created": 0,
        "cost": 0.0,
    }
    for traj_step in traj.steps:
        if isinstance(traj_step.output, AgentOutput):
            for llm_call in traj_step.output.llm_calls:
                if llm_call.usage:
                    stats["prompt"] = int(stats["prompt"]) + llm_call.usage.prompt_tokens
                    stats["completion"] = int(stats["completion"]) + llm_call.usage.completion_tokens
                    stats["cached"] = int(stats["cached"]) + llm_call.usage.cached_tokens
                    stats["cache_created"] = int(stats["cache_created"]) + llm_call.usage.cache_creation_tokens
                    stats["cost"] = float(stats["cost"]) + llm_call.usage.cost
    return stats


def compute_trajectory_stats(traj: Trajectory) -> dict[str, Any]:
    """Compute per-trajectory statistics.

    Returns dict with: n_env_steps, n_agent_steps, total_actions, total_llm_calls,
    duration, prompt_tokens, completion_tokens, cached_tokens, cache_creation_tokens,
    cost, final_reward.
    """
    n_env_steps = 0
    n_agent_steps = 0
    total_actions = 0
    total_llm_calls = 0

    for traj_step in traj.steps:
        if isinstance(traj_step.output, EnvironmentOutput):
            n_env_steps += 1
        elif isinstance(traj_step.output, AgentOutput):
            n_agent_steps += 1
            total_actions += len(traj_step.output.actions)
            total_llm_calls += len(traj_step.output.llm_calls)

    duration = None
    if traj.start_time is not None and traj.end_time is not None:
        duration = traj.end_time - traj.start_time

    final_reward = 0.0
    if traj.reward_info:
        final_reward = traj.reward_info.get("reward", 0.0)
    else:
        for traj_step in reversed(traj.steps):
            if isinstance(traj_step.output, EnvironmentOutput):
                final_reward = traj_step.output.reward
                break

    token_stats = _compute_token_stats_for_trajectory(traj)

    return {
        "n_env_steps": n_env_steps,
        "n_agent_steps": n_agent_steps,
        "total_actions": total_actions,
        "total_llm_calls": total_llm_calls,
        "duration": duration,
        "prompt_tokens": token_stats["prompt"],
        "completion_tokens": token_stats["completion"],
        "cached_tokens": token_stats["cached"],
        "cache_creation_tokens": token_stats["cache_created"],
        "cost": token_stats["cost"],
        "final_reward": final_reward,
    }


def compute_experiment_stats(trajectories: list[Trajectory]) -> str:
    """Aggregate statistics across all trajectories and return as markdown.

    A trajectory is considered finished if it has both start_time and end_time set.
    """
    if not trajectories:
        return ""

    finished_rewards: list[float] = []
    finished_steps: list[int] = []
    finished_durations: list[float] = []
    n_failed = 0

    total_prompt = 0
    total_completion = 0
    total_cached = 0
    total_cache_created = 0
    total_cost = 0.0

    for traj in trajectories:
        stats = compute_trajectory_stats(traj)
        is_finished = traj.start_time is not None and traj.end_time is not None

        if is_finished:
            finished_rewards.append(stats["final_reward"])
            finished_steps.append(stats["n_env_steps"])
            finished_durations.append(stats["duration"])
        else:
            n_failed += 1

        total_prompt += stats["prompt_tokens"]
        total_completion += stats["completion_tokens"]
        total_cached += stats["cached_tokens"]
        total_cache_created += stats["cache_creation_tokens"]
        total_cost += stats["cost"]

    n_finished = len(finished_rewards)
    n_total = n_finished + n_failed

    stats_parts = [f"📊 **{n_total}** trajectories"]
    if n_failed > 0:
        stats_parts.append(f"│ ✅ Finished: **{n_finished}** │ ❌ Failed: **{n_failed}**")
    else:
        stats_parts.append(f"│ ✅ All Finished: **{n_finished}**")

    if n_finished > 0:
        avg_reward = sum(finished_rewards) / n_finished
        avg_steps = sum(finished_steps) / n_finished
        success_rate = sum(1 for r in finished_rewards if r > 0) / n_finished * 100
        stats_parts.append(f"│ Avg Reward: **{avg_reward:.2f}**")
        stats_parts.append(f"│ Success Rate: **{success_rate:.0f}%**")
        stats_parts.append(f"│ Avg Steps: **{avg_steps:.1f}**")
        if finished_durations:
            avg_duration = sum(finished_durations) / len(finished_durations)
            stats_parts.append(f"│ Avg Duration: **{format_duration(avg_duration)}**")

    result = " ".join(stats_parts)

    if total_prompt > 0:
        token_parts = [f"📊 prompt: **{total_prompt:,}**"]
        token_parts.append(f"completion: **{total_completion:,}**")
        token_parts.append(f"total: **{total_prompt + total_completion:,}**")
        if total_cached > 0:
            cache_pct = total_cached / total_prompt * 100
            token_parts.append(f"cached: **{total_cached:,}** ({cache_pct:.0f}%)")
        if total_cache_created > 0:
            token_parts.append(f"cache_created: **{total_cache_created:,}**")
        if total_cost > 0:
            token_parts.append(f"💰 **${total_cost:.4f}**")
        result += "\n\n" + " │ ".join(token_parts)

    return result


# ---------------------------------------------------------------------------
# Agent / Task / Seed hierarchy tables
# ---------------------------------------------------------------------------
#
# NOTE: The per-trajectory stats shown in the task and seed tables (n_steps, tokens, cost)
# are derived by iterating over all loaded TrajectoryStep objects.  When trajectories are
# loaded as metadata stubs (steps=[]), these values will show "-" until the full trajectory
# is loaded — either by the user clicking on a seed, or by the background bulk-loading
# thread in xray.py.
#
# This is a temporary workaround.  The long-term solution is to have the evaluation loop
# persist per-episode summary stats (n_steps, prompt_tokens, completion_tokens, total_cost,
# duration) directly into the *.metadata.json file as each episode completes.  That would
# make all table columns immediately available at experiment-open time without any bulk loading.
# ---------------------------------------------------------------------------


def build_agent_table(trajectories: list[Trajectory]) -> list[dict[str, Any]]:
    """Build one row per unique agent for the top-level agent table.

    Groups trajectories by metadata.get('agent_name', 'unknown').
    Columns: agent_name, n_tasks, n_trajs, avg_reward, total_cost

    total_cost shows "-" when no cost data is available (e.g. unloaded trajectory stubs).
    """
    groups: dict[str, list[Trajectory]] = {}
    for traj in trajectories:
        agent_key = traj.metadata.get("agent_name", "unknown")
        groups.setdefault(agent_key, []).append(traj)

    rows = []
    for agent_key in sorted(groups.keys()):
        agent_trajs = groups[agent_key]
        task_ids = {t.metadata.get("task_id", "unknown") for t in agent_trajs}
        all_stats = [compute_trajectory_stats(t) for t in agent_trajs]
        rewards = [s["final_reward"] for s in all_stats]
        total_cost = sum(float(s["cost"]) for s in all_stats)
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        cost_str = f"${total_cost:.4f}" if total_cost > 0 else "-"

        rows.append(
            {
                "agent_name": agent_key,
                "n_tasks": len(task_ids),
                "n_trajs": len(agent_trajs),
                "avg_reward": round(avg_reward, 3),
                "total_cost": cost_str,
            }
        )
    return rows


def build_task_table(trajectories: list[Trajectory], agent_key: str) -> list[dict[str, Any]]:
    """Build one row per unique task for a selected agent.

    Mirrors seed table columns, showing averages across all seeds under each task.
    Filters trajectories to those matching agent_key.
    Columns: task_id, n_seeds, avg_reward, avg_steps, avg_duration, avg_tokens, avg_cost

    avg_duration is computed from Trajectory.start/end_time (available for metadata stubs).
    avg_steps, avg_tokens, avg_cost show "-" when step data hasn't been loaded yet.
    """
    agent_trajs = [t for t in trajectories if t.metadata.get("agent_name", "unknown") == agent_key]

    groups: dict[str, list[Trajectory]] = {}
    for traj in agent_trajs:
        task_id = traj.metadata.get("task_id", "unknown")
        groups.setdefault(task_id, []).append(traj)

    rows = []
    for task_id in sorted(groups.keys()):
        task_trajs = groups[task_id]
        all_stats = [compute_trajectory_stats(t) for t in task_trajs]

        rewards = [s["final_reward"] for s in all_stats]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        durations = [
            t.end_time - t.start_time
            for t in task_trajs
            if t.start_time is not None and t.end_time is not None
        ]
        avg_duration_str = format_duration(sum(durations) / len(durations)) if durations else "-"

        n_steps_list = [s["n_env_steps"] for s in all_stats]
        avg_steps = sum(n_steps_list) / len(n_steps_list) if n_steps_list else 0
        avg_steps_str = f"{avg_steps:.1f}" if any(n > 0 for n in n_steps_list) else "-"

        total_tokens_list = [int(s["prompt_tokens"]) + int(s["completion_tokens"]) for s in all_stats]
        avg_tokens = sum(total_tokens_list) / len(total_tokens_list) if total_tokens_list else 0
        avg_tokens_str = f"{avg_tokens:,.0f}" if avg_tokens > 0 else "-"

        costs = [float(s["cost"]) for s in all_stats]
        avg_cost = sum(costs) / len(costs) if costs else 0.0
        avg_cost_str = f"${avg_cost:.4f}" if avg_cost > 0 else "-"

        rows.append(
            {
                "task_id": task_id,
                "n_seeds": len(task_trajs),
                "avg_reward": round(avg_reward, 3),
                "avg_steps": avg_steps_str,
                "avg_duration": avg_duration_str,
                "avg_tokens": avg_tokens_str,
                "avg_cost": avg_cost_str,
            }
        )
    return rows


def build_seed_table(
    trajectories: list[Trajectory],
    agent_key: str,
    task_id: str,
) -> list[dict[str, Any]]:
    """Build one row per trajectory (seed) for a selected agent + task.

    Filters trajectories by agent_key and task_id.
    Columns: traj_id, reward, n_steps, duration, tokens, cost
    """
    filtered = [
        t
        for t in trajectories
        if t.metadata.get("agent_name", "unknown") == agent_key and t.metadata.get("task_id", "unknown") == task_id
    ]

    rows = []
    for traj in sorted(filtered, key=lambda t: (t.start_time is None, t.start_time or 0)):
        stats = compute_trajectory_stats(traj)
        duration_str = format_duration(stats["duration"]) if stats["duration"] is not None else "-"
        total_tokens = int(stats["prompt_tokens"]) + int(stats["completion_tokens"])
        tokens_str = f"{total_tokens:,}" if total_tokens > 0 else "-"
        cost_str = f"${float(stats['cost']):.4f}" if float(stats["cost"]) > 0 else "-"
        n_steps = stats["n_env_steps"]

        rows.append(
            {
                "traj_id": traj.id,
                "reward": round(stats["final_reward"], 3),
                "n_steps": n_steps,
                "duration": duration_str,
                "tokens": tokens_str,
                "cost": cost_str,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Timeline HTML generation
# ---------------------------------------------------------------------------

_ENV_COLOR = "#a1c9f4"
_AGENT_COLOR = "#8de5a1"
_CURRENT_BORDER_COLOR = "#ffd700"
_SUCCESS_BORDER_COLOR = "#32cd32"
_FAILURE_BORDER_COLOR = "#dc3545"
_MIN_WIDTH = 12
_MAX_WIDTH = 240


def _compute_step_width(
    duration: float | None,
    min_duration: float,
    max_duration: float,
    min_width: int = _MIN_WIDTH,
    max_width: int = _MAX_WIDTH,
) -> int:
    """Compute pixel width for a timeline segment based on its duration."""
    if duration is None or max_duration <= min_duration:
        return min_width
    normalized = (duration - min_duration) / (max_duration - min_duration)
    return int(min_width + normalized * (max_width - min_width))


def _build_segment_html(
    step_idx: int,
    is_current: bool,
    total_width: int,
    tooltip: str,
    env_frac: float,
    done: bool = False,
    reward: float = 0.0,
) -> str:
    """Build the HTML div for one timeline segment with an env/agent split color bar.

    The segment is split horizontally: left env_frac shows env color, remainder shows agent color.
    env_frac=1.0 means the whole bar is env color (no agent step follows).
    """
    border = f"3px solid {_CURRENT_BORDER_COLOR}" if is_current else "1px solid #ccc"
    box_shadow = "0 0 8px rgba(255, 215, 0, 0.8)" if is_current else "none"

    done_border = ""
    if done:
        done_color = _SUCCESS_BORDER_COLOR if reward > 0 else _FAILURE_BORDER_COLOR
        done_border = f"border-bottom: 4px solid {done_color};"

    step_num = step_idx + 1  # 1-based display

    # Build the inner split bar using a CSS linear-gradient
    env_pct = int(env_frac * 100)
    agent_pct = 100 - env_pct
    if agent_pct == 0:
        gradient = _ENV_COLOR
    else:
        gradient = (
            f"linear-gradient(to right, {_ENV_COLOR} {env_pct}%, {_AGENT_COLOR} {env_pct}%)"
        )

    # Use native setter to properly trigger Gradio's change detection
    onclick = (
        f"const inp = document.querySelector('#timeline_click_input input, #timeline_click_input textarea');"
        f" if(inp) {{"
        f" const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;"
        f" nativeSetter.call(inp, {step_idx});"
        f" inp.dispatchEvent(new Event('input', {{bubbles: true}}));"
        f" inp.dispatchEvent(new Event('change', {{bubbles: true}}));"
        f" }}"
    )

    return (
        f'<div class="timeline-step" data-step="{step_idx}" title="{tooltip}" onclick="{onclick}" style="'
        f"display: inline-flex; align-items: center; justify-content: center;"
        f" min-width: {total_width}px; height: 36px; margin: 2px;"
        f" background: {gradient}; border: {border}; border-radius: 4px;"
        f" cursor: pointer; font-size: 11px; font-weight: bold; color: #333;"
        f" box-shadow: {box_shadow}; {done_border} transition: transform 0.1s;"
        f'"'
        f" onmouseover=\"this.style.transform='scale(1.1)'\" onmouseout=\"this.style.transform='scale(1)'\">"
        f"{step_num}</div>"
    )


def generate_timeline_html(trajectory: Trajectory | None, current_step: int) -> str:
    """Generate an HTML timeline with one segment per UI step (EnvironmentOutput).

    current_step is a UI step index (0-based index into env steps).
    Each segment's width scales with total (env+agent) duration.
    Inside each segment a left-to-right gradient shows env time (blue) vs agent time (green).
    """
    if trajectory is None or not trajectory.steps:
        return "<div style='padding: 10px; color: #666;'>No trajectory loaded</div>"

    env_steps: list[tuple[int, EnvironmentOutput]] = [
        (i, ts.output)  # type: ignore[misc]
        for i, ts in enumerate(trajectory.steps)
        if isinstance(ts.output, EnvironmentOutput)
    ]

    if not env_steps:
        return "<div style='padding: 10px; color: #666;'>No environment steps found</div>"

    def _raw_duration(raw_idx: int) -> float | None:
        ts = trajectory.steps[raw_idx]
        if ts.start_time is not None and ts.end_time is not None:
            return ts.end_time - ts.start_time
        return None

    # Pre-compute per-UI-step env and agent durations
    env_durs: list[float | None] = []
    agent_durs: list[float | None] = []
    total_durs: list[float | None] = []
    for raw_idx, _ in env_steps:
        ed = _raw_duration(raw_idx)
        next_idx = raw_idx + 1
        has_agent = next_idx < len(trajectory.steps) and isinstance(trajectory.steps[next_idx].output, AgentOutput)
        ad = _raw_duration(next_idx) if has_agent else None
        env_durs.append(ed)
        agent_durs.append(ad)
        total = (ed or 0.0) + (ad or 0.0)
        total_durs.append(total if (ed is not None or ad is not None) else None)

    valid_totals = [d for d in total_durs if d is not None and d > 0]
    min_total = min(valid_totals) if valid_totals else 0.0
    max_total = max(valid_totals) if valid_totals else 1.0

    steps_html = []
    for ui_idx, (raw_idx, env_out) in enumerate(env_steps):
        is_current = ui_idx == current_step
        total_width = _compute_step_width(total_durs[ui_idx], min_total, max_total)

        ed = env_durs[ui_idx] or 0.0
        ad = agent_durs[ui_idx] or 0.0
        total = ed + ad
        env_frac = (ed / total) if total > 0 else 1.0

        # Tooltip with individual timings
        tooltip = f"Step {ui_idx + 1}"
        timing_parts = []
        if ed > 0:
            timing_parts.append(f"env: {format_duration(ed)}")
        if ad > 0:
            timing_parts.append(f"agent: {format_duration(ad)}")
        if timing_parts:
            tooltip += f" ({' + '.join(timing_parts)})"
        if env_out.done:
            tooltip += f" — Done, reward: {env_out.reward:.2f}"
        next_idx = raw_idx + 1
        if next_idx < len(trajectory.steps):
            agent_out = trajectory.steps[next_idx].output
            if isinstance(agent_out, AgentOutput) and agent_out.actions:
                action_names = ", ".join(a.name for a in agent_out.actions[:2])
                tooltip += f" → {action_names}"

        steps_html.append(
            _build_segment_html(ui_idx, is_current, total_width, tooltip, env_frac, env_out.done, env_out.reward)
        )

    legend_html = (
        f'<div style="display: flex; gap: 15px; margin-bottom: 8px; font-size: 12px; color: #666;">'
        f'<div style="display: flex; align-items: center; gap: 4px;">'
        f'<div style="width: 22px; height: 14px;'
        f" background: linear-gradient(to right, {_ENV_COLOR} 50%, {_AGENT_COLOR} 50%);"
        f' border-radius: 3px;"></div>'
        f"<span>Env | Agent time</span></div>"
        f'<div style="display: flex; align-items: center; gap: 4px;">'
        f'<div style="width: 16px; height: 16px; border: 2px solid {_CURRENT_BORDER_COLOR}; border-radius: 3px;"></div>'
        f"<span>Current</span></div>"
        f'<div style="display: flex; align-items: center; gap: 4px;">'
        f'<div style="width: 16px; height: 16px; border-bottom: 3px solid {_SUCCESS_BORDER_COLOR}; background: #ddd; border-radius: 3px;"></div>'
        f"<span>Success</span></div>"
        f'<div style="display: flex; align-items: center; gap: 4px;">'
        f'<div style="width: 16px; height: 16px; border-bottom: 3px solid {_FAILURE_BORDER_COLOR}; background: #ddd; border-radius: 3px;"></div>'
        f"<span>Failure</span></div>"
        f"</div>"
    )

    return (
        f'<div style="padding: 10px; background: #f8f9fa; border-radius: 8px;">'
        f"{legend_html}"
        f'<div id="timeline-container" style="'
        f"display: flex; flex-wrap: wrap; align-items: center; padding: 8px;"
        f" background: white; border-radius: 6px; border: 1px solid #dee2e6;"
        f' max-height: 120px; overflow-y: auto;">'
        f"{''.join(steps_html)}"
        f"</div></div>"
    )
