"""ch-diff — show what an agent edited during an episode.

Reconstructs file edits from the stored step files and displays them as
unified diffs. Covers str_replace and write_file actions; bash+sed edits
are noted but cannot be reconstructed without the original file contents.

Usage:
    ch-diff <episode_dir>
    ch-diff <episode_dir> --files          # list only changed files
    ch-diff <episode_dir> --path src/foo.py  # show only one file

Output: one unified diff block per changed file, in the order edits occurred.
Repeated str_replace calls to the same location are shown as a sequence.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Any

import msgpack
import zstandard
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text


def _decompress(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = f.read()
    dctx = zstandard.ZstdDecompressor()
    return msgpack.unpackb(dctx.decompress(data), raw=False)


def _load_actions(ep_dir: Path) -> list[dict[str, Any]]:
    """Return all agent actions from the episode in turn order."""
    steps_dir = ep_dir / "steps"
    if not steps_dir.exists():
        return []
    actions: list[dict[str, Any]] = []
    for f in sorted(steps_dir.glob("*.msgpack.zst")):
        step = _decompress(f)
        output = step.get("output", {})
        if "AgentOutput" in output.get("_type", ""):
            for a in output.get("actions", []):
                actions.append(a)
    return actions


def _apply_edits(actions: list[dict[str, Any]]) -> dict[str, list[tuple[int, str, str]]]:
    """
    Simulate edits and return per-file edit history.

    Returns {path: [(turn, before, after), ...]} where before/after are
    reconstructed file snapshots around the edit.  write_file replaces the
    whole file; str_replace applies a targeted replacement.
    """
    file_contents: dict[str, str] = {}
    history: dict[str, list[tuple[int, str, str]]] = {}

    for turn, action in enumerate(actions):
        name = action.get("name", "")
        args = action.get("arguments", {})

        if name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            before = file_contents.get(path, "")
            file_contents[path] = content
            history.setdefault(path, []).append((turn, before, content))

        elif name == "str_replace":
            path = args.get("path", "")
            old_str = args.get("old_str", "")
            new_str = args.get("new_str", "")
            current = file_contents.get(path, None)
            if current is None:
                # We don't have the original — track the intent only
                history.setdefault(path, []).append((turn, old_str, new_str))
            else:
                before = current
                after = current.replace(old_str, new_str, 1) if old_str in current else current
                file_contents[path] = after
                history.setdefault(path, []).append((turn, before, after))

    return history


def _sed_files(actions: list[dict[str, Any]]) -> set[str]:
    """Return set of paths that appear to have been edited via bash+sed."""
    paths: set[str] = set()
    sed_re = re.compile(r"sed\s+.*?\s+(\S+\.py)")
    for a in actions:
        if a.get("name") == "bash":
            cmd = a.get("arguments", {}).get("command", "")
            for m in sed_re.finditer(cmd):
                paths.add(m.group(1))
    return paths


def _unified_diff(before: str, after: str, path: str, turn: int) -> str:
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    if not diff:
        return ""
    return "\n".join(diff)


def render_diff(ep_dir: Path, console: Console, filter_path: str | None = None, files_only: bool = False) -> None:
    """Render file edits made during the episode as unified diffs."""
    actions = _load_actions(ep_dir)
    if not actions:
        console.print(f"[red]No actions found in {ep_dir}[/red]")
        return

    history = _apply_edits(actions)
    sed_paths = _sed_files(actions)

    all_paths = sorted(set(list(history.keys()) + list(sed_paths)))
    if filter_path:
        all_paths = [p for p in all_paths if filter_path in p]

    if not all_paths:
        console.print("[yellow]No file edits found.[/yellow]")
        return

    meta_file = ep_dir / "episode.metadata.json"
    task_id = ep_dir.name
    if meta_file.exists():
        import json
        meta = json.loads(meta_file.read_text())
        task_id = meta.get("task_id", task_id)

    console.print(f"\n[bold cyan]Edits: {task_id}[/bold cyan]\n")

    if files_only:
        for p in all_paths:
            tags = []
            if p in history:
                tags.append(f"{len(history[p])} edit(s)")
            if p in sed_paths:
                tags.append("sed [yellow](untracked)[/yellow]")
            console.print(f"  {p}  [{', '.join(tags)}]")
        return

    for path in all_paths:
        edits = history.get(path, [])
        is_sed = path in sed_paths

        header = Text()
        header.append(f"\n{'─'*60}\n", style="dim")
        header.append(f"  {path}", style="bold white")
        if is_sed and not edits:
            header.append("  [yellow](only bash+sed — cannot reconstruct)[/yellow]")
        header.append(f"\n{'─'*60}", style="dim")
        console.print(header)

        if not edits:
            continue

        # Show first→last as one diff if file_contents tracking worked, else show each turn
        first_before = edits[0][1]
        last_after = edits[-1][2]
        combined = _unified_diff(first_before, last_after, path, edits[0][0])

        if combined:
            # Check if we had original file (non-empty before on first edit)
            if first_before:
                console.print(Syntax(combined, "diff", theme="monokai", line_numbers=False))
            else:
                # No original — show str_replace intents per turn
                for turn, old, new in edits:
                    console.print(f"  [dim]T{turn:02d} str_replace intent:[/dim]")
                    intent_diff = _unified_diff(old, new, path, turn)
                    if intent_diff:
                        console.print(Syntax(intent_diff, "diff", theme="monokai"))
                    else:
                        console.print(f"  [yellow]  (no change — already applied or no-op)[/yellow]")
        else:
            console.print(f"  [dim](no net change after {len(edits)} edit(s))[/dim]")

        if len(edits) > 1:
            console.print(f"  [dim]({len(edits)} edit passes on this file)[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ch-diff: show agent file edits as unified diffs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("episode_dir", help="Path to episode directory")
    parser.add_argument("--files", action="store_true", help="List changed files only")
    parser.add_argument("--path", default=None, help="Filter to a specific file path substring")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    args = parser.parse_args()

    ep_dir = Path(args.episode_dir).expanduser().resolve()
    if not ep_dir.exists():
        print(f"Error: {ep_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    console = Console(highlight=False, no_color=args.no_color)
    render_diff(ep_dir, console, filter_path=args.path, files_only=args.files)
