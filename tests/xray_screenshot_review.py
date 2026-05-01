"""Standalone visual review script for XRay.

Builds a synthetic experiment, launches xray, puppets through key UI states,
and saves labeled screenshots to /tmp/xray_screenshots/ for agent analysis.

Modes
-----
  status    — Agents → Trajectories: all status symbols + retry badge
  dashboard — Dashboard tab: experiment stats, progress bar
  episode   — Drill into a FAILED episode: Trajectories tab → Logs tab
  all       — Runs all three modes in sequence

Usage
-----
  uv run python tests/xray_screenshot_review.py --mode status
  uv run python tests/xray_screenshot_review.py --mode all
  uv run python tests/xray_screenshot_review.py --mode episode --headed
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from playwright.sync_api import Browser, Page, sync_playwright

# Resolve helpers relative to this file so the script can be run from anywhere.
sys.path.insert(0, str(Path(__file__).parent))
from xray_test_helpers import (  # noqa: E402
    EXTENDED_SCENARIOS,
    build_experiment,
    free_port,
    wait_for_server,
)

OUT_DIR = Path("/tmp/xray_screenshots")


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

@contextmanager
def xray_server(results_dir: Path) -> Generator[str, None, None]:
    port = free_port()
    proc = subprocess.Popen(
        ["uv", "run", "python", "-c",
         f"from pathlib import Path; from cube_harness.analyze.xray import run_xray; "
         f"run_xray(Path('{results_dir}'), port={port})"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        wait_for_server(url, timeout=90.0)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Page navigation helpers
# ---------------------------------------------------------------------------

def load_experiment(page: Page, url: str) -> None:
    """Navigate to xray, open the Experiments tab, and select the experiment."""
    page.goto(url)
    page.wait_for_load_state("load")
    page.wait_for_selector("button[role='tab']", timeout=30_000)
    page.get_by_role("tab", name="Experiments").click()
    page.wait_for_selector("#exp_table tbody tr td", timeout=20_000)
    page.locator("#exp_table td[data-row='0'][data-col='0']").click()
    page.wait_for_function(
        "() => [...document.querySelectorAll('button[role=tab]')]"
        ".some(b => /Agents \\(\\d+\\)/.test(b.textContent))",
        timeout=20_000,
    )
    page.wait_for_timeout(600)


def wait_rows(page: Page, elem_id: str, min_rows: int = 1, timeout: int = 10_000) -> None:
    page.wait_for_function(
        f"() => document.querySelectorAll('#{elem_id} table tbody tr').length >= {min_rows}",
        timeout=timeout,
    )


def click_row(locator, timeout: int = 5_000) -> None:
    """Click a table row, bypassing any upload-overlay that Gradio 5 places over DataFrames."""
    locator.click(force=True, timeout=timeout)


def click_traj(page: Page, traj_id: str) -> None:
    row = page.locator("#traj_table tr").filter(has_text=re.compile(rf"\b{traj_id}\b")).first
    click_row(row)
    page.wait_for_timeout(400)


def shot(page: Page, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=False)
    print(f"  📸  {path}")
    return path


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_status(page: Page, url: str) -> None:
    """Agents → Trajectories: all status symbols + retry badge."""
    print("\n[status] navigating status views…")
    load_experiment(page, url)
    shot(page, "status_01_experiments")

    page.get_by_role("tab", name=re.compile(r"Agents")).click()
    wait_rows(page, "agent_table")
    shot(page, "status_02_agents")

    page.get_by_role("tab", name=re.compile(r"Trajectories")).click()
    wait_rows(page, "traj_table", min_rows=5)
    shot(page, "status_03_trajectories_all")

    for traj_id, label in [("task_2_ep0", "max_steps_retry"), ("task_3_ep0", "failed"), ("task_4_ep0", "stale")]:
        page.get_by_role("tab", name=re.compile(r"Trajectories")).click()
        wait_rows(page, "traj_table", min_rows=5)
        click_traj(page, traj_id)
        page.wait_for_timeout(600)
        shot(page, f"status_04_traj_{label}")


def run_dashboard(page: Page, url: str) -> None:
    """Dashboard tab: progress bar and experiment stats."""
    print("\n[dashboard] navigating dashboard…")
    load_experiment(page, url)
    page.get_by_role("tab", name="Dashboard").click()
    page.wait_for_timeout(600)
    shot(page, "dashboard_01_stats")

    # Agents tab for comparison — shows the compact status cell
    page.get_by_role("tab", name=re.compile(r"Agents")).click()
    wait_rows(page, "agent_table")
    shot(page, "dashboard_02_agent_status_cell")


def run_episode(page: Page, url: str) -> None:
    """Drill into the FAILED episode and inspect the Logs tab."""
    print("\n[episode] drilling into failed episode…")
    load_experiment(page, url)

    # Navigate to Trajectories tab and select task_3_ep0 (FAILED)
    page.get_by_role("tab", name=re.compile(r"Trajectories")).click()
    wait_rows(page, "traj_table", min_rows=5)
    shot(page, "episode_01_trajectories")

    click_traj(page, "task_3_ep0")
    page.wait_for_timeout(600)
    shot(page, "episode_02_task3_selected")

    # Navigate to Logs tab in the bottom panel
    page.get_by_role("tab", name="Logs").click()
    page.wait_for_timeout(800)
    shot(page, "episode_03_logs_tab")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

MODES = {
    "status": run_status,
    "dashboard": run_dashboard,
    "episode": run_episode,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="XRay visual review — generates screenshots for agent analysis.")
    parser.add_argument(
        "--mode", choices=[*MODES, "all"], default="all",
        help="Which UI path to screenshot (default: all).",
    )
    parser.add_argument(
        "--headed", action="store_true",
        help="Run browser in headed (visible) mode instead of headless.",
    )
    args = parser.parse_args()

    modes_to_run = list(MODES.values()) if args.mode == "all" else [MODES[args.mode]]

    with tempfile.TemporaryDirectory() as tmp:
        results_dir = Path(tmp)
        exp_dir = results_dir / "exp_20260101_review"
        build_experiment(exp_dir, EXTENDED_SCENARIOS)

        with xray_server(results_dir) as url:
            with sync_playwright() as pw:
                browser: Browser = pw.chromium.launch(headless=not args.headed)
                for fn in modes_to_run:
                    page: Page = browser.new_page(viewport={"width": 1400, "height": 900})
                    fn(page, url)
                    page.close()
                browser.close()

    print(f"\nScreenshots saved to {OUT_DIR}/")
    print("Paths:")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
