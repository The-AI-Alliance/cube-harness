"""Playwright end-to-end tests for the XRay viewer.

Marked ``@pytest.mark.slow`` — excluded from the fast CI suite.
Run explicitly with: uv run pytest tests/test_xray_e2e.py -m slow

There are two modes:

  Default mode   — assertions on DOM content; no screenshots saved.
  Screenshot mode — pass ``--xray-screenshots`` to save PNGs under
                    /tmp/xray_screenshots/ for manual visual inspection.

Usage examples:
  uv run pytest tests/test_xray_e2e.py -m slow
  uv run pytest tests/test_xray_e2e.py -m slow --xray-screenshots
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import pytest

from cube_harness.analyze import xray_utils
from cube_harness.core import Trajectory
from cube_harness.storage import FileStorage
from tests.xray_test_helpers import SCENARIOS as _SCENARIOS
from tests.xray_test_helpers import build_experiment as _build_experiment
from tests.xray_test_helpers import free_port as _free_port
from tests.xray_test_helpers import wait_for_server as _wait_for_server

# ---------------------------------------------------------------------------
# Screenshot mode (--xray-screenshots flag, registered in conftest.py)
# ---------------------------------------------------------------------------

SCREENSHOT_DIR = Path("/tmp/xray_screenshots")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "xray"


@pytest.fixture(scope="module")
def exp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("xray_e2e")
    exp = base / "exp_20260101_120000"
    _build_experiment(exp, _SCENARIOS)
    return base  # results_dir containing the experiment subdir


@pytest.fixture(scope="module")
def xray_server(exp_dir: Path):
    """Launch xray as a subprocess and return the base URL."""
    port = _free_port()
    proc = subprocess.Popen(
        ["uv", "run", "python", "-c",
         f"from pathlib import Path; from cube_harness.analyze.xray import run_xray; "
         f"run_xray(Path('{exp_dir}'), port={port})"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_server(url, timeout=90.0)
    except TimeoutError:
        out, err = proc.stdout.read(), proc.stderr.read()
        proc.terminate()
        raise TimeoutError(
            f"XRay server did not start at {url} within 90s\n"
            f"stdout: {out.decode()[:500]}\nstderr: {err.decode()[:500]}"
        )
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture
def page_with_exp(page, xray_server: str, request: pytest.FixtureRequest):
    """Navigate to xray, select the experiment, and return the page."""
    take_screenshots = request.config.getoption("--xray-screenshots", default=False)
    page.goto(xray_server)
    page.wait_for_load_state("load")
    # Gradio keeps long-poll connections open — networkidle never fires.
    # Wait for the top-level tabs to render.
    page.wait_for_selector("button[role='tab']", timeout=30000)

    if take_screenshots:
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SCREENSHOT_DIR / "00_initial.png"))

    # Navigate to Experiments tab and select the first experiment.
    page.get_by_role("tab", name="Experiments").click()
    page.wait_for_selector("#exp_table tbody tr td", timeout=20000)

    # Click the first checkbox cell to select the experiment.
    # If auto-select already loaded the experiment, on_experiments_change returns
    # gr.skip() and the hierarchy is unchanged — idempotent either way.
    page.locator("#exp_table td[data-row='0'][data-col='0']").click()

    # Wait for the Agents tab label to show a count e.g. "Agents (1)".
    page.wait_for_function(
        "() => [...document.querySelectorAll('button[role=tab]')].some(b => /Agents \\(\\d+\\)/.test(b.textContent))",
        timeout=20000,
    )
    page.wait_for_timeout(500)

    if take_screenshots:
        page.screenshot(path=str(SCREENSHOT_DIR / "01_exp_selected.png"))

    yield page, take_screenshots


# ---------------------------------------------------------------------------
# Fixture-only unit tests (no running server needed)
# ---------------------------------------------------------------------------


class TestFixtureLoading:
    """Verify the static fixture files load correctly with the current storage layer."""

    def test_v2_no_status_loads_without_episode_status(self) -> None:
        storage = FileStorage(FIXTURES_DIR / "v2_no_status")
        trajs = storage.load_all_trajectory_metadata()
        assert len(trajs) == 1
        assert "_episode_status" not in trajs[0].metadata
        # Legacy heuristic still classifies correctly.
        status = xray_utils.trajectory_status(trajs[0])
        assert status in ("success", "fail", "running", "queued", "system_error")

    def test_v2_with_status_reads_episode_status(self) -> None:
        storage = FileStorage(FIXTURES_DIR / "v2_with_status")
        trajs = storage.load_all_trajectory_metadata()
        traj_map = {t.id: t for t in trajs}
        assert traj_map["task_1_ep0"].metadata["_episode_status"] == "COMPLETED"
        assert xray_utils.trajectory_status(traj_map["task_1_ep0"]) == "success"

    def test_v2_with_status_max_steps(self) -> None:
        storage = FileStorage(FIXTURES_DIR / "v2_with_status")
        trajs = storage.load_all_trajectory_metadata()
        traj_map = {t.id: t for t in trajs}
        assert traj_map["task_1_ep1"].metadata["_episode_status"] == "MAX_STEPS_REACHED"
        assert xray_utils.trajectory_status(traj_map["task_1_ep1"]) == "max_steps"

    def test_v2_with_status_retry_count_injected(self) -> None:
        storage = FileStorage(FIXTURES_DIR / "v2_with_status")
        trajs = storage.load_all_trajectory_metadata()
        ep1 = next(t for t in trajs if t.id == "task_1_ep1")
        assert ep1.metadata.get("_retry_count") == 1


# ---------------------------------------------------------------------------
# Browser-based tests (require running server)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestXRayStatusUI:
    def test_page_loads(self, page_with_exp: tuple) -> None:
        page, _ = page_with_exp
        assert "XRay" in page.title() or page.locator("body").inner_text() != ""

    def _table_text(self, page, elem_id: str) -> str:
        """Return the visible text content of the named DataFrame component."""
        return page.locator(f"#{elem_id}").inner_text()

    def _wait_for_table_rows(self, page, elem_id: str, min_rows: int = 1, timeout: int = 10000) -> None:
        """Block until the named table has at least min_rows data rows (counts <tr> elements)."""
        page.wait_for_function(
            f"() => document.querySelectorAll('#{elem_id} table tbody tr').length >= {min_rows}",
            timeout=timeout,
        )

    def _click_task_row_by_id(self, page, task_id: str) -> None:
        """Click the task row whose task_id cell contains task_id.

        Gradio non-interactive DataFrames may not render data-col attributes,
        so we find the row via text content instead.
        """
        row = page.locator("#task_table tr").filter(has_text=re.compile(rf"\b{task_id}\b")).first
        row.click()
        page.wait_for_timeout(500)

    def test_agents_tab_has_status_column(self, page_with_exp: tuple) -> None:
        page, screenshots = page_with_exp
        page.get_by_role("tab", name=re.compile(r"Agents")).click()
        self._wait_for_table_rows(page, "agent_table")
        if screenshots:
            page.screenshot(path=str(SCREENSHOT_DIR / "02_agents_tab.png"))
        table_html = self._table_text(page, "agent_table")
        assert "✓" in table_html

    def test_agents_tab_has_no_n_err_column(self, page_with_exp: tuple) -> None:
        page, _ = page_with_exp
        page.get_by_role("tab", name=re.compile(r"Agents")).click()
        self._wait_for_table_rows(page, "agent_table")
        # Use text_content() for headers which may be inside aria-hidden tables.
        headers = [h.text_content() for h in page.locator("#agent_table table thead th").all()]
        assert not any("n_err" in h for h in headers)
        assert not any("n_running" in h for h in headers)

    def test_tasks_tab_shows_failed_symbol(self, page_with_exp: tuple) -> None:
        page, screenshots = page_with_exp
        # Experiment auto-selects first agent; navigate to Tasks directly.
        page.get_by_role("tab", name=re.compile(r"Tasks")).click()
        self._wait_for_table_rows(page, "task_table", min_rows=4)
        if screenshots:
            page.screenshot(path=str(SCREENSHOT_DIR / "03_tasks_tab.png"))
        table_html = self._table_text(page, "task_table")
        assert "⛔" in table_html

    def test_tasks_tab_shows_max_steps_symbol(self, page_with_exp: tuple) -> None:
        page, _ = page_with_exp
        page.get_by_role("tab", name=re.compile(r"Tasks")).click()
        self._wait_for_table_rows(page, "task_table", min_rows=4)
        table_html = self._table_text(page, "task_table")
        assert "🎬" in table_html

    def test_seeds_tab_shows_retry_badge(self, page_with_exp: tuple) -> None:
        """The MAX_STEPS_REACHED seed has retry_count=1 — ×1 badge should appear."""
        page, screenshots = page_with_exp
        page.get_by_role("tab", name=re.compile(r"Tasks")).click()
        self._wait_for_table_rows(page, "task_table", min_rows=4)
        # Click the task_2 row (MAX_STEPS_REACHED episode).
        self._click_task_row_by_id(page, "task_2")
        page.get_by_role("tab", name=re.compile(r"Seeds")).click()
        # Wait until seed table is rendered and shows task_2's trajectory.
        page.wait_for_function(
            "() => { const el = document.querySelector('#seed_table'); return el && el.innerText.includes('task_2_ep0'); }",
            timeout=8000,
        )
        if screenshots:
            page.screenshot(path=str(SCREENSHOT_DIR / "04_seeds_tab_retry.png"))
        table_html = self._table_text(page, "seed_table")
        assert "×1" in table_html

    def test_stale_symbol_visible(self, page_with_exp: tuple) -> None:
        page, screenshots = page_with_exp
        page.get_by_role("tab", name=re.compile(r"Tasks")).click()
        self._wait_for_table_rows(page, "task_table", min_rows=4)
        # Click the task_4 row (STALE episode).
        self._click_task_row_by_id(page, "task_4")
        page.get_by_role("tab", name=re.compile(r"Seeds")).click()
        # Wait until seed table is rendered and shows task_4's trajectory.
        page.wait_for_function(
            "() => { const el = document.querySelector('#seed_table'); return el && el.innerText.includes('task_4_ep0'); }",
            timeout=8000,
        )
        if screenshots:
            page.screenshot(path=str(SCREENSHOT_DIR / "05_seeds_stale.png"))
        table_html = self._table_text(page, "seed_table")
        assert "👻" in table_html
