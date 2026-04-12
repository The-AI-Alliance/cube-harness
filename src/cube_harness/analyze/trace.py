"""ch-trace — compact per-turn episode trace viewer.

Usage:
    ch-trace <episode_dir>
    ch-trace <experiment_dir>/<episode_name>
    ch-trace episodes/workarena.servicenow.create-incident_ep0

Output: one row per turn showing action, result, page title, reward, and validation message.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import msgpack
import zstandard
from rich.console import Console
from rich.table import Table
from rich.text import Text


def _decompress(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = f.read()
    dctx = zstandard.ZstdDecompressor()
    return msgpack.unpackb(dctx.decompress(data), raw=False)


def _page_title(obs_output: dict[str, Any]) -> str:
    """Extract page title from an EnvironmentOutput's obs contents."""
    contents = obs_output.get("obs", {}).get("contents", [])
    for content in contents:
        if not isinstance(content, dict):
            continue
        text = content.get("data", "") or ""
        if not isinstance(text, str):
            continue
        m = re.search(r"RootWebArea '([^']+)'", text)
        if m:
            title = m.group(1)
            # Trim common ServiceNow suffix noise
            title = re.sub(r"\s*\|\s*ServiceNow$", "", title)
            return title[:60]
    return ""


def _action_summary(act_output: dict[str, Any]) -> str:
    """Summarise the first action in an AgentOutput."""
    actions = act_output.get("output", {}).get("actions", [])
    if not actions:
        error = act_output.get("output", {}).get("error")
        return f"[error: {error}]" if error else "(no action)"
    a = actions[0]
    name = a.get("name", "?")
    args = a.get("arguments", {})
    # Compact arg representation
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 20:
            v = v[:18] + "…"
        parts.append(f"{k}={v!r}" if k != "bid" else str(v))
    return f"{name}({', '.join(parts)})"


def _result_from_obs(obs_output: dict[str, Any]) -> str:
    """Extract action result string from the first content of an obs (tool_call_id response)."""
    contents = obs_output.get("obs", {}).get("contents", [])
    for c in contents:
        if isinstance(c, dict) and c.get("tool_call_id"):
            data = c.get("data", "")
            if isinstance(data, str):
                return data[:40]
    return ""


def render_trace(ep_dir: Path, console: Console) -> None:
    steps_dir = ep_dir / "steps"
    if not steps_dir.exists():
        console.print(f"[red]No steps/ directory in {ep_dir}[/red]")
        return

    step_files = sorted(steps_dir.glob("*.msgpack.zst"))
    if not step_files:
        console.print(f"[red]No step files in {steps_dir}[/red]")
        return

    # Load all steps
    steps: list[dict[str, Any]] = [_decompress(f) for f in step_files]

    # Read episode metadata for task_id
    meta_file = ep_dir / "episode.metadata.json"
    task_id = ep_dir.name
    if meta_file.exists():
        import json

        meta = json.loads(meta_file.read_text())
        task_id = meta.get("task_id", task_id)

    console.print(f"\n[bold cyan]Trace: {task_id}[/bold cyan]  ({ep_dir.name})\n")

    turn = 0
    i = 0
    rows = []
    while i < len(steps):
        step = steps[i]
        output = step.get("output", {})
        output_type = output.get("_type", "")

        if "AgentOutput" in output_type:
            action_str = _action_summary(step)
            obs_step = steps[i + 1] if i + 1 < len(steps) else {}
            obs_out = obs_step.get("output", {})
            result = _result_from_obs(obs_out)
            page = _page_title(obs_out)
            reward = obs_out.get("reward", 0.0)
            msg = (obs_out.get("info") or {}).get("message", "")
            rows.append((turn, action_str, result, page, reward, msg))
            turn += 1
            i += 2
        else:
            i += 1

    # Two-line format: action+result on line 1, page+reward+msg on line 2
    for t, action_str, result, page, reward, msg in rows:
        # Result colour
        if result.startswith("Failed") or result.startswith("[error"):
            res_style = "red"
        elif result == "Success":
            res_style = "green"
        else:
            res_style = "yellow"

        rew_style = "green bold" if reward > 0 else "dim"
        line1 = Text()
        line1.append(f"T{t:02d} ", style="dim")
        line1.append(f"{action_str:<44}", style="bold")
        line1.append(f"  [{result[:9]:9}]", style=res_style)
        line2 = Text()
        line2.append("     ", style="dim")
        line2.append(f"{page[:44]:<44}", style="dim")
        line2.append(f"  r=", style="dim")
        line2.append(f"{reward:.1f}", style=rew_style)
        if msg:
            line2.append(f"  {msg[:60]}", style="italic dim")
        console.print(line1)
        console.print(line2)

    # Print final reward from metadata
    meta_file = ep_dir / "episode.metadata.json"
    if meta_file.exists():
        import json

        meta = json.loads(meta_file.read_text())
        final_reward = meta.get("reward_info", {}).get("reward", "?")
        done = meta.get("reward_info", {}).get("done", "?")
        msg = meta.get("reward_info", {}).get("message", "")
        status = "[green]✓ SOLVED[/green]" if final_reward == 1.0 else "[red]✗ FAILED[/red]"
        console.print(f"\n{status}  reward={final_reward}  done={done}  msg={msg!r}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ch-trace: compact per-turn episode trace viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("episode_dir", help="Path to episode directory (contains steps/)")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    args = parser.parse_args()

    ep_dir = Path(args.episode_dir).expanduser().resolve()
    if not ep_dir.exists():
        print(f"Error: {ep_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    console = Console(highlight=False, no_color=args.no_color)
    render_trace(ep_dir, console)
