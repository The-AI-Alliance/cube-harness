"""ch-watch — live episode status monitor for an in-progress experiment.

Polls status.json files and updates a Rich table in-place as episodes
complete. Exits when all episodes reach a terminal state.

Usage:
    ch-watch <results_dir>
    ch-watch <results_dir> --interval 10   # poll every 10s (default 15)
    ch-watch <results_dir> --once          # print once and exit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text


_TERMINAL = {"DONE", "MAX_STEPS_REACHED", "ERROR", "CANCELLED"}
_STATUS_STYLE = {
    "DONE": "green",
    "MAX_STEPS_REACHED": "yellow",
    "ERROR": "red",
    "CANCELLED": "dim",
    "RUNNING": "cyan",
    "QUEUED": "dim",
}


def _read_statuses(results_dir: Path) -> list[dict]:
    statuses = []
    for sf in sorted((results_dir / "episodes").glob("*/status.json")):
        try:
            s = json.loads(sf.read_text())
            statuses.append(s)
        except Exception:
            pass
    return statuses


def _build_table(statuses: list[dict], results_dir: Path) -> Table:
    done = sum(1 for s in statuses if s.get("status") in _TERMINAL)
    passed = sum(1 for s in statuses if s.get("reward") == 1.0)
    total = len(statuses)

    title = f"[bold]{results_dir.name}[/bold]  {done}/{total} done  [green]{passed} passed[/green]"
    table = Table(title=title, show_lines=False, box=None, padding=(0, 1))
    table.add_column("Task", style="", min_width=36)
    table.add_column("Status", min_width=18)
    table.add_column("Step", justify="right", min_width=4)
    table.add_column("Reward", justify="right", min_width=6)

    for s in statuses:
        status = s.get("status", "?")
        style = _STATUS_STYLE.get(status, "")
        reward = s.get("reward")
        reward_str = f"[green]{reward:.1f}[/green]" if reward == 1.0 else (f"{reward:.1f}" if reward is not None else "—")
        step = s.get("current_step", 0)
        table.add_row(
            s.get("task_id", "?"),
            Text(status, style=style),
            str(step),
            Text.from_markup(reward_str),
        )
    return table


def watch(results_dir: Path, interval: int, once: bool, console: Console) -> None:
    if not (results_dir / "episodes").exists():
        console.print(f"[red]No episodes/ directory in {results_dir}[/red]")
        return

    if once:
        statuses = _read_statuses(results_dir)
        console.print(_build_table(statuses, results_dir))
        return

    with Live(console=console, refresh_per_second=1) as live:
        while True:
            statuses = _read_statuses(results_dir)
            live.update(_build_table(statuses, results_dir))
            terminal_count = sum(1 for s in statuses if s.get("status") in _TERMINAL)
            if statuses and terminal_count == len(statuses):
                break
            time.sleep(interval)
        # Final render
        live.update(_build_table(statuses, results_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ch-watch: live episode status monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("results_dir", help="Path to experiment results directory")
    parser.add_argument("--interval", type=int, default=15, help="Poll interval in seconds (default 15)")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    console = Console(highlight=False, no_color=args.no_color)
    watch(results_dir, args.interval, args.once, console)
