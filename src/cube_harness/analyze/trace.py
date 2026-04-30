"""ch-trace — compact per-turn episode trace viewer.

Quick alternative to the full XRay viewer when you just need to scan what an agent
did step by step without opening a browser.

Usage:
    ch-trace <episode_dir>
    ch-trace <episode_dir> --eval        # also dump eval fields from last environment step
    ch-trace experiments/workarena-l1/workarena.servicenow.create-incident_ep0

Output: two lines per turn —
    T00 fill(bid=123, value='CHG…')           [Success  ]
         ServiceNow | Create Change Request    r=0.0

    T00 bash(command='python -m pytest…')     [PASSED  ]
         [100%] 5 passed in 0.3s              r=0.0

Data model
----------
Each step is a msgpack+zstd file in <episode_dir>/steps/.  Steps alternate:
    even index — AgentOutput   (agent chose an action)
    odd  index — EnvironmentOutput (env executed it, returned observation)

render_trace() pairs them: for each AgentOutput at index i, the observation is at i+1.

The observation's contents list is heterogeneous:
    - entries with a tool_call_id  →  the direct result string for the preceding action
    - entries without              →  raw page state (AXTree text, screenshot bytes, …)

Context line (line 2): for browser episodes the page title is extracted from the AXTree
"RootWebArea '<title>'" label. For coding/terminal episodes the first non-empty line of
the tool result is shown instead.

Episode-level outcome (final reward, done, validation message) is read from
episode.metadata.json, written by the harness after the episode ends.

--eval prints all fields from the last EnvironmentOutput's info dict.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import msgpack
import zstandard
from rich.console import Console
from rich.text import Text


def _decompress(path: Path) -> dict[str, Any]:
    """Read and decompress a single step file (.msgpack.zst) into a plain dict."""
    with open(path, "rb") as f:
        data = f.read()
    dctx = zstandard.ZstdDecompressor()
    return msgpack.unpackb(dctx.decompress(data), raw=False)


def _context_line(obs_output: dict[str, Any]) -> str:
    """Return a short context string for line 2.

    Browser episodes: extract page title from AXTree "RootWebArea '<title>'" label.
    Coding/terminal episodes: fall back to the first non-empty line of the tool result.
    """
    contents = obs_output.get("obs", {}).get("contents", [])
    for content in contents:
        if not isinstance(content, dict):
            continue
        text = content.get("data", "") or ""
        if not isinstance(text, str):
            continue
        m = re.search(r"RootWebArea '([^']+)'", text)
        if m:
            title = re.sub(r"\s*\|\s*ServiceNow$", "", m.group(1))
            return title[:60]
    for c in contents:
        if isinstance(c, dict) and c.get("tool_call_id"):
            data = c.get("data", "") or ""
            if isinstance(data, str):
                for line in data.splitlines():
                    line = line.strip()
                    if line:
                        return line[:60]
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
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 20:
            v = v[:18] + "…"
        parts.append(f"{k}={v!r}" if k != "bid" else str(v))
    return f"{name}({', '.join(parts)})"


def _result_from_obs(obs_output: dict[str, Any]) -> str:
    """Extract action result string from the first tool_call_id content in an obs."""
    contents = obs_output.get("obs", {}).get("contents", [])
    for c in contents:
        if isinstance(c, dict) and c.get("tool_call_id"):
            data = c.get("data", "")
            if isinstance(data, str):
                return data[:40]
    return ""


def render_trace(ep_dir: Path, console: Console) -> None:
    """Render a compact two-line-per-turn trace for a single episode directory."""
    steps_dir = ep_dir / "steps"
    if not steps_dir.exists():
        console.print(f"[red]No steps/ directory in {ep_dir}[/red]")
        return

    step_files = sorted(steps_dir.glob("*.msgpack.zst"))
    if not step_files:
        console.print(f"[red]No step files in {steps_dir}[/red]")
        return

    steps: list[dict[str, Any]] = [_decompress(f) for f in step_files]

    meta_file = ep_dir / "episode.metadata.json"
    task_id = ep_dir.name
    if meta_file.exists():
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
            context = _context_line(obs_out)
            reward = obs_out.get("reward", 0.0)
            msg = (obs_out.get("info") or {}).get("message", "")
            rows.append((turn, action_str, result, context, reward, msg))
            turn += 1
            i += 2
        else:
            i += 1

    for t, action_str, result, context, reward, msg in rows:
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
        line2.append(f"{context[:44]:<44}", style="dim")
        line2.append("  r=", style="dim")
        line2.append(f"{reward:.1f}", style=rew_style)
        if msg:
            line2.append(f"  {msg[:60]}", style="italic dim")
        console.print(line1)
        console.print(line2)

    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        final_reward = meta.get("reward_info", {}).get("reward", "?")
        done = meta.get("reward_info", {}).get("done", "?")
        msg = meta.get("reward_info", {}).get("message", "")
        status = "[green]✓ SOLVED[/green]" if final_reward == 1.0 else "[red]✗ FAILED[/red]"
        console.print(f"\n{status}  reward={final_reward}  done={done}  msg={msg!r}\n")


def render_eval(ep_dir: Path, console: Console) -> None:
    """Dump all fields from the last EnvironmentOutput step's info dict."""
    steps_dir = ep_dir / "steps"
    if not steps_dir.exists():
        console.print(f"[red]No steps/ directory in {ep_dir}[/red]")
        return

    step_files = sorted(steps_dir.glob("*.msgpack.zst"))
    if not step_files:
        console.print(f"[red]No step files in {steps_dir}[/red]")
        return

    steps: list[dict[str, Any]] = [_decompress(f) for f in step_files]

    info: dict[str, Any] = {}
    for step in reversed(steps):
        output = step.get("output", {})
        if "AgentOutput" not in output.get("_type", ""):
            info = output.get("info") or {}
            if info:
                break

    if not info:
        console.print("[yellow]No eval fields found in last environment step.[/yellow]")
        return

    console.print("\n[bold]Eval info:[/bold]")
    blocks: list[tuple[str, str]] = []
    for key, val in info.items():
        if isinstance(val, bool):
            style = "green" if val else "red"
            console.print(f"  {key:<30} [{style}]{val}[/{style}]")
        elif isinstance(val, str) and ("\n" in val or len(val) > 80):
            blocks.append((key, val))
        else:
            console.print(f"  {key:<30} {val!r}")
    for key, val in blocks:
        console.print(f"\n[bold]{key}:[/bold]")
        console.print(val)


def main() -> None:
    """Entry point for the ch-trace CLI (registered in pyproject.toml [project.scripts])."""
    parser = argparse.ArgumentParser(
        description="ch-trace: compact per-turn episode trace viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("episode_dir", help="Path to episode directory (contains steps/)")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Also dump eval fields from the last environment step's info dict",
    )
    args = parser.parse_args()

    ep_dir = Path(args.episode_dir).expanduser().resolve()
    if not ep_dir.exists():
        print(f"Error: {ep_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    console = Console(highlight=False, no_color=args.no_color)
    render_trace(ep_dir, console)
    if args.eval:
        render_eval(ep_dir, console)
