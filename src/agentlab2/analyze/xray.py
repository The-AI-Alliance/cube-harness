"""AgentLab2 XRay Viewer.

A Gradio-based experiment viewer with agent/task/seed hierarchy, lazy tab loading,
and rich step inspection capabilities. Compatible with the AL2 data format.

Step model: a "UI step" is one environment observation paired with the agent action
that follows it (if any). Navigation moves between env steps. Step N shows:
  - the Nth EnvironmentOutput (screenshot, axtree, reward, etc.)
  - the AgentOutput that immediately follows it, if one exists (actions, LLM call, etc.)
"""

import argparse
import html as html_lib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from os.path import expanduser
from pathlib import Path
from typing import Any, Callable

import gradio as gr
from PIL import Image

from agentlab2.analyze import xray_utils
from agentlab2.core import AgentOutput, EnvironmentOutput, Trajectory, TrajectoryStep
from agentlab2.storage import FileStorage


# ---------------------------------------------------------------------------
# State identifiers
# ---------------------------------------------------------------------------


@dataclass
class StepId:
    """Identifies a UI step (env-step index) within the currently loaded trajectory."""

    step: int = 0


# ---------------------------------------------------------------------------
# XRayState — all mutable viewer state, captured by closures
# ---------------------------------------------------------------------------


@dataclass
class XRayState:
    """All mutable state for the XRay viewer, captured by handler closures."""

    results_dir: Path
    trajectories: list[Trajectory] = field(default_factory=list)
    selected_agent_key: str | None = None
    selected_task_id: str | None = None
    current_trajectory: Trajectory | None = None
    # Index into env_step_indices — i.e., which UI step is current
    step: int = 0

    # Cached list of raw-step indices that are EnvironmentOutputs
    _env_step_indices: list[int] = field(default_factory=list)
    # Storage instance kept for on-demand full trajectory loading
    _storage: FileStorage | None = field(default=None, repr=False)
    # Set to True once the background bulk-loading thread has finished
    _bg_loading_done: bool = field(default=True, repr=False)
    # Agent name derived from experiment_config.json for backwards-compatibility backfill
    _backfill_name: str | None = field(default=None, repr=False)
    # JSON strings for the Config tabs (set on experiment load, None if unavailable)
    _agent_config_json: str | None = field(default=None, repr=False)
    _exp_config_json: str | None = field(default=None, repr=False)
    # Live polling: tracks which trajectories are done (skip on future ticks) and their mtimes
    _completed_ids: set[str] = field(default_factory=set, repr=False)
    _traj_mtimes: dict[str, float] = field(default_factory=dict, repr=False)
    # Timestamp of last detected file change — used to stop polling stale experiments
    _last_change_time: float = field(default=0.0, repr=False)

    def load_experiment(self, exp_dir: Path) -> bool:
        """Load trajectory metadata stubs from an experiment directory. Returns True on success.

        Only reads *.metadata.json files — steps are loaded lazily when a trajectory is selected.
        After returning, a background thread is started to load all full trajectories so that
        stats (steps, tokens, cost) become available without requiring the user to click each seed.

        For backwards compatibility with trajectories that predate the agent_name metadata field,
        reads experiment_config.json (if present) to derive the agent name and stores it as
        _backfill_name so it is applied consistently to every trajectory loaded from disk.
        """
        self._storage = FileStorage(exp_dir)
        self.trajectories = self._storage.load_all_trajectory_metadata()
        self._load_experiment_config(exp_dir)
        for traj in self.trajectories:
            self._maybe_backfill(traj)
        self.selected_agent_key = None
        self.selected_task_id = None
        self.current_trajectory = None
        self.step = 0
        self._env_step_indices = []
        self._completed_ids = {t.id for t in self.trajectories if t.end_time is not None}
        self._traj_mtimes = self._storage.list_trajectory_ids_with_mtime()
        self._last_change_time = time.time()
        self._bg_loading_done = False
        self._start_background_loading()
        return len(self.trajectories) > 0

    def _load_experiment_config(self, exp_dir: Path) -> None:
        """Read experiment_config.json and populate config fields.

        Sets _backfill_name (agent class short name for backwards compat),
        _agent_config_json (pretty JSON for the Agent Config tab), and
        _exp_config_json (pretty JSON for the Experiment Config tab, with
        agent_config replaced by a placeholder to avoid duplication).
        """
        self._backfill_name = None
        self._agent_config_json = None
        self._exp_config_json = None
        config_path = exp_dir / "experiment_config.json"
        if not config_path.exists():
            return
        try:
            with open(config_path) as f:
                exp_cfg = json.load(f)
            agent_cfg = exp_cfg.get("agent_config", {})
            agent_type = agent_cfg.get("_type", "")
            self._backfill_name = agent_type.split(".")[-1] if agent_type else None
            self._agent_config_json = json.dumps(agent_cfg, indent=2)
            exp_cfg_display = {**exp_cfg, "agent_config": "(see Agent Config tab)"}
            self._exp_config_json = json.dumps(exp_cfg_display, indent=2)
        except Exception:
            pass

    def _maybe_backfill(self, traj: Trajectory) -> None:
        """Inject _backfill_name into a trajectory's metadata if agent_name is absent."""
        if self._backfill_name and "agent_name" not in traj.metadata:
            traj.metadata["agent_name"] = self._backfill_name

    def _start_background_loading(self) -> None:
        """Spawn a daemon thread that loads all trajectory stubs into full trajectories.

        Each trajectory is loaded and cached in-place in self.trajectories so that the
        hierarchy tables (agent/task/seed) can display accurate step/token/cost stats
        once loading completes.

        NOTE: This background thread is a temporary workaround for the missing summary stats
        on trajectory metadata stubs.  The long-term fix is to have the evaluation loop
        persist per-episode stats (n_steps, tokens, cost, duration) directly into the
        *.metadata.json file as it runs, making bulk loading unnecessary.
        See: https://github.com/AgentLab2/AgentLab2/issues/TODO
        """
        if self._storage is None:
            self._bg_loading_done = True
            return

        storage = self._storage  # capture to avoid closure over mutable self._storage

        def _load_all() -> None:
            for i, traj in enumerate(self.trajectories):
                # Skip if already fully loaded (e.g. user clicked it first)
                if traj.steps:
                    continue
                try:
                    full = storage.load_trajectory(traj.id)
                    self._maybe_backfill(full)
                    self.trajectories[i] = full
                    # Keep current_trajectory in sync if it was this stub
                    if self.current_trajectory is not None and self.current_trajectory.id == traj.id:
                        self.current_trajectory = full
                        self._env_step_indices = self._build_env_indices()
                except Exception:
                    pass  # leave stub; table will show "-" for unavailable stats
            self._bg_loading_done = True

        thread = threading.Thread(target=_load_all, daemon=True)
        thread.start()

    def refresh_experiment(self) -> bool:
        """Incrementally reload new or changed trajectories from disk. Returns True if anything changed.

        Uses mtime-based change detection: only trajectories whose files have changed since the
        last check are reloaded. Completed trajectories (end_time set) are skipped entirely.
        Called on each bg_timer tick while the experiment is still running.
        """
        if self._storage is None:
            return False
        id_mtimes = self._storage.list_trajectory_ids_with_mtime()
        changed = False
        known_ids = {t.id for t in self.trajectories}

        for traj_id, mtime in id_mtimes.items():
            if traj_id in self._completed_ids:
                continue
            prev_mtime = self._traj_mtimes.get(traj_id, 0.0)
            if mtime <= prev_mtime and traj_id in known_ids:
                continue
            try:
                full = self._storage.load_trajectory(traj_id)
                self._maybe_backfill(full)
                self._traj_mtimes[traj_id] = mtime
                changed = True
                self._last_change_time = time.time()
                if traj_id in known_ids:
                    idx = next(i for i, t in enumerate(self.trajectories) if t.id == traj_id)
                    self.trajectories[idx] = full
                    if self.current_trajectory is not None and self.current_trajectory.id == traj_id:
                        self.current_trajectory = full
                        self._env_step_indices = self._build_env_indices()
                else:
                    self.trajectories.append(full)
                if full.end_time is not None:
                    self._completed_ids.add(traj_id)
            except Exception:
                pass
        return changed

    def is_experiment_complete(self) -> bool:
        """Return True when every known trajectory has finished (end_time is set)."""
        if not self.trajectories:
            return False
        return all(t.id in self._completed_ids for t in self.trajectories)

    def is_experiment_stale(self, timeout_s: float = 1200.0) -> bool:
        """Return True if no file changes have been detected for timeout_s seconds.

        Used to stop polling when an experiment was killed and workers are no longer writing.
        Only meaningful after bulk loading is done and the experiment is not yet complete.
        """
        return time.time() - self._last_change_time > timeout_s

    def select_agent(self, agent_key: str) -> None:
        """Select an agent; resets task, trajectory, and step."""
        self.selected_agent_key = agent_key
        self.selected_task_id = None
        self.current_trajectory = None
        self.step = 0
        self._env_step_indices = []

    def select_task(self, task_id: str) -> None:
        """Select a task for the current agent; resets trajectory and step."""
        self.selected_task_id = task_id
        self.current_trajectory = None
        self.step = 0
        self._env_step_indices = []

    def select_trajectory(self, traj_id: str) -> None:
        """Select a trajectory by ID; loads full steps lazily if not yet loaded."""
        idx = next((i for i, t in enumerate(self.trajectories) if t.id == traj_id), None)
        if idx is None:
            self.current_trajectory = None
            self.step = 0
            self._env_step_indices = []
            return
        traj = self.trajectories[idx]
        # Stub has steps=[]; load full trajectory on first access and cache it in place
        if not traj.steps and self._storage is not None:
            try:
                traj = self._storage.load_trajectory(traj_id)
                self._maybe_backfill(traj)
                self.trajectories[idx] = traj
            except Exception:
                pass  # keep stub; renders will show empty state gracefully
        self.current_trajectory = traj
        self.step = 0
        self._env_step_indices = self._build_env_indices()

    def _build_env_indices(self) -> list[int]:
        """Return raw indices of all EnvironmentOutput steps in current trajectory."""
        if self.current_trajectory is None:
            return []
        return [i for i, ts in enumerate(self.current_trajectory.steps) if isinstance(ts.output, EnvironmentOutput)]

    def total_ui_steps(self) -> int:
        """Number of UI steps = number of EnvironmentOutputs in current trajectory."""
        return len(self._env_step_indices)

    def get_env_output(self) -> EnvironmentOutput | None:
        """Return the EnvironmentOutput for the current UI step."""
        if not self._env_step_indices or self.step >= len(self._env_step_indices):
            return None
        raw_idx = self._env_step_indices[self.step]
        output = self.current_trajectory.steps[raw_idx].output  # type: ignore[union-attr]
        return output if isinstance(output, EnvironmentOutput) else None

    def get_agent_output(self) -> AgentOutput | None:
        """Return the AgentOutput immediately following the current env step, or None."""
        if not self._env_step_indices or self.step >= len(self._env_step_indices):
            return None
        raw_idx = self._env_step_indices[self.step] + 1
        if self.current_trajectory is None or raw_idx >= len(self.current_trajectory.steps):
            return None
        output = self.current_trajectory.steps[raw_idx].output
        return output if isinstance(output, AgentOutput) else None

    def get_env_traj_step(self) -> TrajectoryStep | None:
        """Return the TrajectoryStep (with timing) for the current env output."""
        if not self._env_step_indices or self.step >= len(self._env_step_indices):
            return None
        raw_idx = self._env_step_indices[self.step]
        return self.current_trajectory.steps[raw_idx]  # type: ignore[union-attr]

    def get_agent_traj_step(self) -> TrajectoryStep | None:
        """Return the TrajectoryStep for the agent output following the current env step."""
        if not self._env_step_indices or self.step >= len(self._env_step_indices):
            return None
        raw_idx = self._env_step_indices[self.step] + 1
        if self.current_trajectory is None or raw_idx >= len(self.current_trajectory.steps):
            return None
        ts = self.current_trajectory.steps[raw_idx]
        return ts if isinstance(ts.output, AgentOutput) else None


# ---------------------------------------------------------------------------
# Lazy tab loading decorator
# ---------------------------------------------------------------------------


def if_active(tab_name: str, n_out: int = 1) -> Callable:
    """Decorator factory that makes a handler a no-op when the given tab is not active.

    The wrapped function receives `active_tab` as its first positional argument (a str).
    When active_tab != tab_name: returns gr.skip() (or a tuple of n_out gr.skip()).
    When active_tab == tab_name: calls the original function (render functions read
    state via closure and take no extra arguments).

    Usage:
        step_id.change(
            fn=if_active("AXTree")(render_axtree),
            inputs=[active_tab, step_id],
            outputs=axtree_code,
        )
    """

    def decorator(fn: Callable) -> Callable:
        def wrapper(active_tab: str, *_args: Any, **_kwargs: Any) -> Any:
            if active_tab != tab_name:
                if n_out == 1:
                    return gr.skip()
                return tuple(gr.skip() for _ in range(n_out))
            # Render functions read state via closure — no args to forward.
            return fn()

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# CSS and keyboard shortcuts JS
# ---------------------------------------------------------------------------


_CSS = """
html {
    color-scheme: light only;
}
.compact-header {
    padding: 8px 16px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 8px;
    color: white;
}
.compact-header, .compact-header * {
    color: white !important;
}
.step-details {
    max-height: 600px;
    overflow-y: auto;
    padding: 12px;
}
.step-details pre {
    max-height: 300px;
    overflow-y: auto;
}
.error-box {
    background: #fee2e2;
    border: 1px solid #ef4444;
    border-radius: 6px;
    padding: 8px 12px;
    margin-top: 8px;
}
.info-panel {
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}
.info-panel-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 4px 10px;
    color: #6b7280;
}
.info-panel-body {
    padding: 6px 10px;
    max-height: 100px;
    overflow-y: auto;
    font-size: 13px;
    line-height: 1.5;
}
.info-panel-body code {
    background: rgba(0,0,0,0.05);
    border-radius: 3px;
    padding: 1px 4px;
    font-size: 12px;
}
code {
    white-space: pre-wrap;
}
th {
    white-space: normal !important;
    word-wrap: break-word !important;
}
#timeline_click_input {
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}
"""

_FORCE_LIGHT_JS = "() => { document.body.classList.remove('dark'); }"

_SHORTCUT_JS = """
<script>
function shortcuts(e) {
    switch (e.target.tagName.toLowerCase()) {
        case "input":
        case "textarea":
        case "select":
        case "button":
            return;
        default:
            if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                if (e.key === 'ArrowLeft') {
                    document.getElementById("xray_prev_btn").click();
                } else {
                    document.getElementById("xray_next_btn").click();
                }
            }
    }
}
document.addEventListener('keydown', shortcuts, false);
</script>
"""


# ---------------------------------------------------------------------------
# HTML rendering helpers (tables + info panels)
# ---------------------------------------------------------------------------


def _render_goal_panel(text: str) -> str:
    """Render the task goal as a styled HTML panel with a fixed title bar."""
    safe = html_lib.escape(text)
    # Preserve newlines
    safe = safe.replace("\n", "<br>")
    return (
        '<div class="info-panel" style="background:#f0f4ff; border-color:#c7d2fe;">'
        '<div class="info-panel-title" style="background:#e0e7ff; color:#4338ca;">📋 Goal</div>'
        f'<div class="info-panel-body">{safe}</div>'
        "</div>"
    )


def _render_action_panel(text: str) -> str:
    """Render the agent action as a styled HTML panel with a fixed title bar."""
    safe = html_lib.escape(text)
    safe = safe.replace("\n", "<br>")
    # Convert escaped backtick spans back to <code> tags
    safe = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)
    # Replace *italic* markers (used in placeholder messages like *Terminal step*)
    safe = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", safe)
    return (
        '<div class="info-panel" style="background:#f0fdf4; border-color:#bbf7d0;">'
        '<div class="info-panel-title" style="background:#dcfce7; color:#15803d;">🤖 Action</div>'
        f'<div class="info-panel-body">{safe}</div>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# run_xray — main entry point
# ---------------------------------------------------------------------------


def run_xray(
    results_dir: Path,
    debug: bool = False,
    port: int | None = None,
    share: bool = False,
) -> None:
    """Launch the AgentLab2 XRay Gradio viewer.

    Args:
        results_dir: Path to the root results directory containing experiment subdirectories.
        debug: Enable Gradio debug mode with hot reloading.
        port: Server port. If None, Gradio picks an available port.
        share: Enable Gradio share link for remote access.
    """
    if isinstance(results_dir, str):
        results_dir = Path(results_dir)

    # Single state instance captured by all handler closures below
    state = XRayState(results_dir=results_dir)

    # ------------------------------------------------------------------
    # Handler functions (closures capturing `state`)
    # ------------------------------------------------------------------

    def _make_tab_labels(
        agent_rows: list[dict[str, Any]],
        task_rows: list[dict[str, Any]],
        seed_rows: list[dict[str, Any]],
    ) -> tuple[gr.Tab, gr.Tab, gr.Tab]:
        """Return gr.Tab updates with counts embedded in labels."""
        return (
            gr.Tab(label=f"Agents ({len(agent_rows)})"),
            gr.Tab(label=f"Tasks ({len(task_rows)})"),
            gr.Tab(label=f"Seeds ({len(seed_rows)})"),
        )

    def refresh_exp_dir_choices(current_choice: str) -> gr.Dropdown:
        return gr.Dropdown(choices=xray_utils.get_directory_contents(state.results_dir), value=current_choice)

    def _config_jsons() -> tuple[str, str]:
        """Return (agent_config_json, exp_config_json) from current state, with empty fallback."""
        return (
            state._agent_config_json or "",
            state._exp_config_json or "",
        )

    def on_select_experiment(exp_name: str) -> tuple[str, Any, Any, Any, StepId, gr.Tab, gr.Tab, gr.Tab, str, str]:
        if exp_name in ("Select experiment directory", "") or not exp_name:
            return "", None, None, None, StepId(), gr.Tab(label="Agents (0)"), gr.Tab(label="Tasks (0)"), gr.Tab(label="Seeds (0)"), "", ""
        dir_name = exp_name.split(" (")[0]
        exp_dir = state.results_dir / dir_name
        state.load_experiment(exp_dir)
        exp_stats = xray_utils.compute_experiment_stats(state.trajectories)
        agent_rows = xray_utils.build_agent_table(state.trajectories)

        # Auto-select first agent
        if not agent_rows:
            seed_rows: list[dict[str, Any]] = []
            task_rows: list[dict[str, Any]] = []
            tab_labels = _make_tab_labels(agent_rows, task_rows, seed_rows)
            return exp_stats, _rows_to_table(agent_rows), [], [], StepId(), *tab_labels, *_config_jsons()
        first_agent_key = agent_rows[0]["agent_name"]
        state.select_agent(first_agent_key)
        agent_table_data = _rows_to_table(agent_rows, first_agent_key, "agent_name")

        # Auto-select first task
        task_rows = xray_utils.build_task_table(state.trajectories, first_agent_key)
        if not task_rows:
            tab_labels = _make_tab_labels(agent_rows, task_rows, [])
            return exp_stats, agent_table_data, _rows_to_table(task_rows), [], StepId(), *tab_labels, *_config_jsons()
        first_task_id = task_rows[0]["task_id"]
        state.select_task(first_task_id)
        task_table_data = _rows_to_table(task_rows, first_task_id, "task_id")

        # Auto-select first seed/trajectory
        seed_rows = xray_utils.build_seed_table(state.trajectories, first_agent_key, first_task_id)
        if not seed_rows:
            tab_labels = _make_tab_labels(agent_rows, task_rows, seed_rows)
            return exp_stats, agent_table_data, task_table_data, _rows_to_table(seed_rows), StepId(), *tab_labels, *_config_jsons()
        first_traj_id = seed_rows[0]["traj_id"]
        state.select_trajectory(first_traj_id)
        seed_table_data = _rows_to_table(seed_rows, first_traj_id, "traj_id")

        tab_labels = _make_tab_labels(agent_rows, task_rows, seed_rows)
        return exp_stats, agent_table_data, task_table_data, seed_table_data, StepId(step=0), *tab_labels, *_config_jsons()

    def on_select_agent(evt: gr.SelectData, agent_df: Any) -> tuple[Any, Any, Any, StepId, gr.Tab, gr.Tab, gr.Tab]:
        if agent_df is None or len(agent_df) == 0:
            return [], [], [], StepId(), gr.Tab(label="Agents (0)"), gr.Tab(label="Tasks (0)"), gr.Tab(label="Seeds (0)")
        row = evt.index[0]
        # Strip HTML tags to get the raw key value
        agent_key = re.sub(r"<[^>]+>", "", str(agent_df.iloc[row, 0]))
        state.select_agent(agent_key)
        agent_rows = xray_utils.build_agent_table(state.trajectories)
        agent_table_data = _rows_to_table(agent_rows, agent_key, "agent_name")
        task_rows = xray_utils.build_task_table(state.trajectories, agent_key)
        if not task_rows:
            tab_labels = _make_tab_labels(agent_rows, task_rows, [])
            return agent_table_data, _rows_to_table(task_rows), [], StepId(), *tab_labels
        # Auto-select first task and first seed
        first_task_id = task_rows[0]["task_id"]
        state.select_task(first_task_id)
        task_table_data = _rows_to_table(task_rows, first_task_id, "task_id")
        seed_rows = xray_utils.build_seed_table(state.trajectories, agent_key, first_task_id)
        if not seed_rows:
            tab_labels = _make_tab_labels(agent_rows, task_rows, seed_rows)
            return agent_table_data, task_table_data, _rows_to_table(seed_rows), StepId(), *tab_labels
        first_traj_id = seed_rows[0]["traj_id"]
        state.select_trajectory(first_traj_id)
        seed_table_data = _rows_to_table(seed_rows, first_traj_id, "traj_id")
        tab_labels = _make_tab_labels(agent_rows, task_rows, seed_rows)
        return agent_table_data, task_table_data, seed_table_data, StepId(step=0), *tab_labels

    def on_select_task(evt: gr.SelectData, task_df: Any) -> tuple[Any, Any, StepId, gr.Tab]:
        if task_df is None or len(task_df) == 0:
            return [], [], StepId(), gr.Tab(label="Seeds (0)")
        row = evt.index[0]
        task_id = re.sub(r"<[^>]+>", "", str(task_df.iloc[row, 0]))
        state.select_task(task_id)
        if state.selected_agent_key is None:
            return [], [], StepId(), gr.Tab(label="Seeds (0)")
        task_rows = xray_utils.build_task_table(state.trajectories, state.selected_agent_key)
        task_table_data = _rows_to_table(task_rows, task_id, "task_id")
        seed_rows = xray_utils.build_seed_table(state.trajectories, state.selected_agent_key, task_id)
        seeds_tab_update = gr.Tab(label=f"Seeds ({len(seed_rows)})")
        if not seed_rows:
            return task_table_data, _rows_to_table(seed_rows), StepId(), seeds_tab_update
        # Auto-select first seed
        first_traj_id = seed_rows[0]["traj_id"]
        state.select_trajectory(first_traj_id)
        seed_table_data = _rows_to_table(seed_rows, first_traj_id, "traj_id")
        return task_table_data, seed_table_data, StepId(step=0), seeds_tab_update

    def on_select_seed(evt: gr.SelectData, seed_df: Any) -> tuple[Any, StepId]:
        if seed_df is None or len(seed_df) == 0:
            return [], StepId(step=0)
        row = evt.index[0]
        traj_id = re.sub(r"<[^>]+>", "", str(seed_df.iloc[row, 1]))  # col 0 is status emoji
        state.select_trajectory(traj_id)
        if state.selected_agent_key is None or state.selected_task_id is None:
            return [], StepId(step=0)
        seed_rows = xray_utils.build_seed_table(state.trajectories, state.selected_agent_key, state.selected_task_id)
        seed_table_data = _rows_to_table(seed_rows, traj_id, "traj_id")
        return seed_table_data, StepId(step=0)

    def on_bg_load_tick() -> tuple[Any, Any, Any, Any, Any, gr.Timer, gr.Tab, gr.Tab, gr.Tab]:
        """Periodic refresh: bulk-loads stubs then live-polls for new/changed trajectories.

        Stops polling when the experiment is complete or has been stale for 2 minutes
        (no file changes detected — indicates the experiment was killed).
        """
        if state._bg_loading_done:
            state.refresh_experiment()

        exp_stats = xray_utils.compute_experiment_stats(state.trajectories)
        agent_rows = xray_utils.build_agent_table(state.trajectories)
        agent_key = state.selected_agent_key
        active_agent = agent_rows[0]["agent_name"] if (agent_rows and agent_key is None) else agent_key
        agent_table_data = _rows_to_table(agent_rows, active_agent, "agent_name")

        task_key = state.selected_task_id
        task_rows = xray_utils.build_task_table(state.trajectories, active_agent) if active_agent else []
        task_table_data = _rows_to_table(task_rows, task_key, "task_id")

        seed_rows = (
            xray_utils.build_seed_table(state.trajectories, active_agent, task_key)
            if active_agent and task_key
            else []
        )
        traj_id = state.current_trajectory.id if state.current_trajectory else None
        seed_table_data = _rows_to_table(seed_rows, traj_id, "traj_id")

        n_completed = len(state._completed_ids)
        n_total = len(state.trajectories)
        n_running = sum(1 for t in state.trajectories if t.end_time is None)
        progress_html = xray_utils.build_progress_html(n_completed, n_total, n_running)

        done = state._bg_loading_done and (state.is_experiment_complete() or state.is_experiment_stale())
        tab_labels = _make_tab_labels(agent_rows, task_rows, seed_rows)
        return exp_stats, agent_table_data, task_table_data, seed_table_data, progress_html, gr.Timer(active=not done), *tab_labels

    def navigate_prev(step_id: StepId) -> StepId:
        step = max(0, step_id.step - 1)
        state.step = step
        return StepId(step=step)

    def navigate_next(step_id: StepId) -> StepId:
        step = min(state.total_ui_steps() - 1, step_id.step + 1)
        state.step = step
        return StepId(step=step)

    def handle_timeline_click(clicked_step: int | None) -> StepId:
        if clicked_step is not None and state.current_trajectory:
            step = int(max(0, min(clicked_step, state.total_ui_steps() - 1)))
            state.step = step
            return StepId(step=step)
        return StepId(step=state.step)

    # ------------------------------------------------------------------
    # Always-rendered handlers (update on every step change)
    # ------------------------------------------------------------------

    def get_compact_header_info() -> str:
        if not state.current_trajectory:
            return "No trajectory selected"
        task_id = state.current_trajectory.metadata.get("task_id", "unknown")
        agent_name = state.current_trajectory.metadata.get("agent_name", "")
        header = f"**{task_id}**"
        if agent_name:
            header += f" │ {agent_name}"
        if state.current_trajectory.reward_info:
            reward = state.current_trajectory.reward_info.get("reward", 0.0)
            done = state.current_trajectory.reward_info.get("done", True)
        else:
            reward = 0.0
            done = False
            for traj_step in reversed(state.current_trajectory.steps):
                if isinstance(traj_step.output, EnvironmentOutput):
                    reward = traj_step.output.reward
                    done = traj_step.output.done
                    break
        reward_emoji = "✅" if reward > 0 and done else "❌" if done else "⏳"
        return header + f" │ {reward_emoji} Reward: {reward:.2f}"

    def get_step_counter() -> str:
        if not state.current_trajectory:
            return "Step 0/0"
        return f"Step {state.step + 1}/{state.total_ui_steps()}"

    def update_timeline() -> str:
        return xray_utils.generate_timeline_html(state.current_trajectory, state.step)

    def update_trajectory_stats() -> str:
        if not state.current_trajectory:
            return ""
        stats = xray_utils.compute_trajectory_stats(state.current_trajectory)
        n_env = stats["n_env_steps"]
        n_agent = stats["n_agent_steps"]
        total_actions = stats["total_actions"]
        total_llm_calls = stats["total_llm_calls"]

        line1 = (
            f"🌍 Env Steps: **{n_env}** │ 🤖 Agent Steps: **{n_agent}**"
            f" │ ⚡ Actions: **{total_actions}** │ 💬 LLM Calls: **{total_llm_calls}**"
        )
        if stats["duration"] is not None:
            line1 += f" │ ⏱️ **{xray_utils.format_duration(stats['duration'])}**"

        prompt_tokens = int(stats["prompt_tokens"])
        completion_tokens = int(stats["completion_tokens"])
        cached_tokens = int(stats["cached_tokens"])
        cache_creation_tokens = int(stats["cache_creation_tokens"])
        cost = float(stats["cost"])

        if prompt_tokens > 0:
            token_parts = [f"📊 prompt: **{prompt_tokens:,}**"]
            token_parts.append(f"completion: **{completion_tokens:,}**")
            token_parts.append(f"total: **{prompt_tokens + completion_tokens:,}**")
            if cached_tokens > 0:
                cache_pct = cached_tokens / prompt_tokens * 100
                token_parts.append(f"cached: **{cached_tokens:,}** ({cache_pct:.0f}%)")
            if cache_creation_tokens > 0:
                token_parts.append(f"cache_created: **{cache_creation_tokens:,}**")
            if cost > 0:
                token_parts.append(f"💰 **${cost:.4f}**")
            return line1 + "\n\n" + " │ ".join(token_parts)

        return line1

    def get_task_goal() -> str:
        """Return the task goal as a rendered HTML panel."""
        return _render_goal_panel(xray_utils.get_task_goal(state.current_trajectory))

    def get_agent_action_md() -> str:
        """Return the current step's agent action as a rendered HTML panel."""
        return _render_action_panel(xray_utils.get_agent_action_markdown(state.get_agent_output()))

    # ------------------------------------------------------------------
    # Lazy tab render handlers (only run when their tab is active).
    # Each reads state via closure and takes no arguments.
    # ------------------------------------------------------------------

    def _render_screenshots() -> tuple[Image.Image | None, Image.Image | None]:
        env_out = state.get_env_output()
        current_img = xray_utils.get_screenshot_from_step(env_out)
        # Show previous env screenshot as "before" in the accordion
        prev_img = None
        if state.step > 0 and state._env_step_indices:
            prev_raw_idx = state._env_step_indices[state.step - 1]
            prev_ts = state.current_trajectory.steps[prev_raw_idx]  # type: ignore[union-attr]
            prev_img = xray_utils.get_screenshot_from_step(prev_ts.output)
        return current_img, prev_img

    def _render_gallery() -> list[tuple[Image.Image, str]]:
        if not state.current_trajectory:
            return []
        screenshots = xray_utils.get_all_screenshots(state.current_trajectory)
        return [(img, f"Step {i + 1}") for i, img in screenshots]

    def _render_step_details() -> str:
        env_out = state.get_env_output()
        agent_out = state.get_agent_output()
        env_ts = state.get_env_traj_step()
        agent_ts = state.get_agent_traj_step()
        return xray_utils.get_paired_step_details_markdown(env_out, agent_out, env_ts, agent_ts)

    def _render_axtree() -> str:
        env_out = state.get_env_output()
        if env_out is None:
            return "No environment step selected."
        content = xray_utils.extract_obs_content(env_out, "axtree")
        if content is None:
            return "No AXTree content found in this step."
        return content

    def _render_chat() -> str:
        agent_out = state.get_agent_output()
        result = xray_utils.get_chat_messages_markdown(agent_out)
        if not result:
            return "No agent action follows this observation (terminal step)."
        return result

    def _render_error() -> str:
        env_out = state.get_env_output()
        agent_out = state.get_agent_output()
        return xray_utils.get_paired_error_markdown(env_out, agent_out)

    def _render_logs() -> str:
        env_out = state.get_env_output()
        return xray_utils.get_step_logs_markdown(env_out, state.current_trajectory)

    def _render_debug() -> tuple[str, str, str]:
        env_out = state.get_env_output()
        agent_out = state.get_agent_output()
        if env_out is None:
            return "No step selected", "No step selected", "No step selected"
        env_json = env_out.model_dump_json(indent=2)
        llm_calls_json = "No agent step follows this observation"
        llm_tools_json = "No agent step follows this observation"
        if agent_out is not None:
            if agent_out.llm_calls:
                calls_data = [call.model_dump() for call in agent_out.llm_calls]
                llm_calls_json = json.dumps(calls_data, indent=2, default=str)
                llm_call = agent_out.llm_calls[0]
                if llm_call.prompt.tools:
                    llm_tools_json = json.dumps(llm_call.prompt.tools, indent=2)
                else:
                    llm_tools_json = "No tools in LLM call"
            else:
                llm_calls_json = "No LLM calls in agent step"
                llm_tools_json = "No LLM calls in agent step"
        return env_json, llm_calls_json, llm_tools_json

    # ------------------------------------------------------------------
    # Tab activation helpers — no-arg named functions avoid lambda warnings.
    # Gradio tab.select fires with no extra inputs, so these take no args.
    # ------------------------------------------------------------------

    def _activate_screenshots() -> str:
        return "Screenshots"

    def _activate_gallery() -> str:
        return "Screenshot Gallery"

    def _activate_step_details() -> str:
        return "Step Details"

    def _activate_axtree() -> str:
        return "AXTree"

    def _activate_chat() -> str:
        return "Chat Messages"

    def _activate_error() -> str:
        return "Task Error"

    def _activate_logs() -> str:
        return "Logs"

    def _activate_debug() -> str:
        return "Debug"

    # ------------------------------------------------------------------
    # Build the Gradio UI
    # ------------------------------------------------------------------

    with gr.Blocks(theme=gr.themes.Soft(), css=_CSS, head=_SHORTCUT_JS, js=_FORCE_LIGHT_JS) as demo:  # type: ignore[attr-defined]
        active_tab = gr.State(value="Screenshots")
        step_id = gr.State(value=StepId())

        with gr.Accordion("Help", open=False):
            gr.Markdown(
                """\
# AgentLab2 XRay

1. **Select experiment directory** from the dropdown.
2. **Hierarchy**: click Agents → Tasks → Seeds to drill into a specific episode.
3. **Navigate steps** using Prev/Next buttons or Ctrl/Cmd + Arrow keys.
4. Each **step** shows the environment observation and the agent action that follows it.
5. **Tabs** below update lazily — only the active tab renders on step change.
"""
            )

        with gr.Row():
            exp_dir_choice = gr.Dropdown(
                choices=xray_utils.get_directory_contents(results_dir),
                value="Select experiment directory",
                label="Experiment Directory",
                show_label=False,
                scale=6,
            )
            refresh_button = gr.Button("↺ Refresh", scale=0, size="sm")

        with gr.Accordion("📂 Trajectory Hierarchy", open=True):
            with gr.Tabs():
                with gr.Tab("Dashboard"):
                    progress_bar = gr.HTML("")
                    experiment_stats = gr.Markdown("")
                with gr.Tab("Agents") as agents_tab:
                    agent_table = gr.DataFrame(
                        headers=["agent_name", "n_tasks", "n_trajs", "avg_reward", "total_cost"],
                        datatype="html",
                        max_height=260,
                        show_label=False,
                        interactive=False,
                    )
                with gr.Tab("Tasks") as tasks_tab:
                    task_table = gr.DataFrame(
                        headers=["task_id", "n_seeds", "avg_reward", "avg_steps", "avg_duration", "avg_tokens", "avg_cost"],
                        datatype="html",
                        max_height=260,
                        show_label=False,
                        interactive=False,
                    )
                with gr.Tab("Seeds") as seeds_tab:
                    seed_table = gr.DataFrame(
                        headers=["status", "traj_id", "reward", "n_steps", "duration", "tokens", "cost"],
                        datatype="html",
                        max_height=260,
                        show_label=False,
                        interactive=False,
                    )
                with gr.Tab("Agent Config"):
                    agent_config_code = gr.Code(language="json", show_label=False)
                with gr.Tab("Exp Config"):
                    exp_config_code = gr.Code(language="json", show_label=False)

        # Timer: ticks every 1s to bulk-load stubs and then live-poll for new/changed trajectories.
        # Starts inactive; activated on experiment select; deactivates when experiment is complete
        # or when no file changes are detected for 2 minutes (killed experiment).
        bg_timer = gr.Timer(value=1.0, active=False)

        with gr.Row(variant="panel", elem_classes="compact-header"):
            with gr.Column(scale=1, min_width=200):
                header_info = gr.Markdown("**Select a trajectory**")
            with gr.Column(scale=3):
                stats_display = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=0, min_width=90):
                prev_btn = gr.Button("◀ Prev", size="sm", elem_id="xray_prev_btn", min_width=80)
            with gr.Column(scale=0, min_width=110):
                step_counter = gr.Markdown("Step 0/0")
            with gr.Column(scale=1):
                timeline_html = gr.HTML(label="Timeline")
            with gr.Column(scale=0, min_width=90):
                next_btn = gr.Button("Next ▶", size="sm", elem_id="xray_next_btn", min_width=80)

        with gr.Row(visible=True, elem_id="timeline_click_input"):
            timeline_click_input = gr.Number(show_label=False, container=False)

        # Always-visible panels: task goal (stable per trajectory) + agent action (per step)
        with gr.Row():
            with gr.Column(scale=2):
                task_goal_md = gr.HTML(value="")
            with gr.Column(scale=3):
                agent_action_md = gr.HTML(value="")

        with gr.Tabs() as main_tabs:
            with gr.Tab("Screenshots") as screenshots_tab:
                screenshot = gr.Image(
                    label="Current Screenshot",
                    show_label=True,
                    interactive=False,
                    show_download_button=False,
                    height=500,
                )
                with gr.Accordion("📷 Previous Screenshot", open=False):
                    prev_screenshot = gr.Image(
                        show_label=False,
                        interactive=False,
                        show_download_button=False,
                        height=400,
                    )

            with gr.Tab("Screenshot Gallery") as gallery_tab:
                screenshot_gallery = gr.Gallery(
                    columns=2,
                    show_download_button=False,
                    show_label=False,
                    object_fit="contain",
                    preview=True,
                )

            with gr.Tab("Step Details") as step_details_tab:
                step_details = gr.Markdown(
                    value="Select a trajectory to view step details",
                    elem_classes="step-details",
                )

            with gr.Tab("AXTree") as axtree_tab:
                axtree_code = gr.Code(language=None, show_label=False, max_lines=40)

            with gr.Tab("Chat Messages") as chat_tab:
                chat_md = gr.Markdown()

            with gr.Tab("Task Error") as error_tab:
                error_md = gr.Markdown()

            with gr.Tab("Logs") as logs_tab:
                logs_md = gr.Markdown()

            with gr.Tab("Debug") as debug_tab:
                with gr.Tabs():
                    with gr.Tab("Env JSON"):
                        raw_json = gr.Code(language="json", show_label=False)
                    with gr.Tab("LLM Calls"):
                        llm_calls_code = gr.Code(language="json", show_label=False)
                    with gr.Tab("LLM Tools"):
                        llm_tools_code = gr.Code(language="json", show_label=False)

        # ------------------------------------------------------------------
        # Event wiring
        # ------------------------------------------------------------------

        refresh_button.click(fn=refresh_exp_dir_choices, inputs=exp_dir_choice, outputs=exp_dir_choice)

        def on_select_experiment_with_timer(
            exp_name: str,
        ) -> tuple[str, Any, Any, Any, StepId, gr.Tab, gr.Tab, gr.Tab, str, str, gr.Timer]:
            """Wrap on_select_experiment to also activate the background-loading timer."""
            result = on_select_experiment(exp_name)
            # Activate timer only if background loading was actually started (trajectories exist)
            timer_active = not state._bg_loading_done
            return (*result, gr.Timer(active=timer_active))

        exp_dir_choice.change(
            fn=on_select_experiment_with_timer,
            inputs=exp_dir_choice,
            outputs=[experiment_stats, agent_table, task_table, seed_table, step_id, agents_tab, tasks_tab, seeds_tab, agent_config_code, exp_config_code, bg_timer],
        )

        bg_timer.tick(
            fn=on_bg_load_tick,
            outputs=[experiment_stats, agent_table, task_table, seed_table, progress_bar, bg_timer, agents_tab, tasks_tab, seeds_tab],
            show_progress="hidden",
        )

        agent_table.select(
            fn=on_select_agent,
            inputs=agent_table,
            outputs=[agent_table, task_table, seed_table, step_id, agents_tab, tasks_tab, seeds_tab],
        )
        task_table.select(fn=on_select_task, inputs=task_table, outputs=[task_table, seed_table, step_id, seeds_tab])
        seed_table.select(fn=on_select_seed, inputs=seed_table, outputs=[seed_table, step_id])

        # Timeline click
        timeline_click_input.change(fn=handle_timeline_click, inputs=timeline_click_input, outputs=step_id)

        # Navigation buttons
        prev_btn.click(fn=navigate_prev, inputs=step_id, outputs=step_id)
        next_btn.click(fn=navigate_next, inputs=step_id, outputs=step_id)

        # Always-rendered on step change
        step_id.change(fn=get_compact_header_info, outputs=header_info)
        step_id.change(fn=get_step_counter, outputs=step_counter)
        step_id.change(fn=update_timeline, outputs=timeline_html)
        step_id.change(fn=update_trajectory_stats, outputs=stats_display)
        step_id.change(fn=get_task_goal, outputs=task_goal_md)
        step_id.change(fn=get_agent_action_md, outputs=agent_action_md)

        # Lazy renders on step change (active_tab checked by if_active; step_id is the trigger)
        step_id.change(
            fn=if_active("Screenshots", 2)(_render_screenshots),
            inputs=[active_tab, step_id],
            outputs=[screenshot, prev_screenshot],
        )
        step_id.change(
            fn=if_active("Screenshot Gallery")(_render_gallery),
            inputs=[active_tab, step_id],
            outputs=screenshot_gallery,
        )
        step_id.change(
            fn=if_active("Step Details")(_render_step_details),
            inputs=[active_tab, step_id],
            outputs=step_details,
        )
        step_id.change(
            fn=if_active("AXTree")(_render_axtree),
            inputs=[active_tab, step_id],
            outputs=axtree_code,
        )
        step_id.change(
            fn=if_active("Chat Messages")(_render_chat),
            inputs=[active_tab, step_id],
            outputs=chat_md,
        )
        step_id.change(
            fn=if_active("Task Error")(_render_error),
            inputs=[active_tab, step_id],
            outputs=error_md,
        )
        step_id.change(
            fn=if_active("Logs")(_render_logs),
            inputs=[active_tab, step_id],
            outputs=logs_md,
        )
        step_id.change(
            fn=if_active("Debug", 3)(_render_debug),
            inputs=[active_tab, step_id],
            outputs=[raw_json, llm_calls_code, llm_tools_code],
        )

        # Tab selection: update active_tab state AND immediately re-render the newly visible tab.
        # Tab .select fires with no extra inputs — handlers take no arguments.
        screenshots_tab.select(fn=_activate_screenshots, outputs=active_tab)
        screenshots_tab.select(fn=_render_screenshots, outputs=[screenshot, prev_screenshot])

        gallery_tab.select(fn=_activate_gallery, outputs=active_tab)
        gallery_tab.select(fn=_render_gallery, outputs=screenshot_gallery)

        step_details_tab.select(fn=_activate_step_details, outputs=active_tab)
        step_details_tab.select(fn=_render_step_details, outputs=step_details)

        axtree_tab.select(fn=_activate_axtree, outputs=active_tab)
        axtree_tab.select(fn=_render_axtree, outputs=axtree_code)

        chat_tab.select(fn=_activate_chat, outputs=active_tab)
        chat_tab.select(fn=_render_chat, outputs=chat_md)

        error_tab.select(fn=_activate_error, outputs=active_tab)
        error_tab.select(fn=_render_error, outputs=error_md)

        logs_tab.select(fn=_activate_logs, outputs=active_tab)
        logs_tab.select(fn=_render_logs, outputs=logs_md)

        debug_tab.select(fn=_activate_debug, outputs=active_tab)
        debug_tab.select(fn=_render_debug, outputs=[raw_json, llm_calls_code, llm_tools_code])

        # Initial dropdown refresh on page load
        demo.load(fn=refresh_exp_dir_choices, inputs=exp_dir_choice, outputs=exp_dir_choice)

    demo.queue()
    demo.launch(server_port=port, share=share, debug=debug)


def _rows_to_table(rows: list[dict[str, Any]], active_key: str | None = None, key_col: str = "") -> list[list[Any]]:
    """Convert a list of dicts to a list of lists for Gradio DataFrame.

    When active_key and key_col are provided, cells in the matching row are
    wrapped in a highlight span (used with datatype='html' DataFrames).
    """
    if not rows:
        return []
    result = []
    for row in rows:
        is_active = active_key is not None and str(row.get(key_col, "")) == str(active_key)
        if is_active:
            cells = [f'<span style="font-weight:600;color:#1d4ed8">{v}</span>' for v in row.values()]
        else:
            cells = [str(v) for v in row.values()]
        result.append(cells)
    return result


def main() -> None:
    """CLI entry point for al2-xray."""
    default_results_dir = expanduser("~/agentlab_results/al2")
    parser = argparse.ArgumentParser(description="AgentLab2 XRay Experiment Viewer")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=default_results_dir,
        help="Path to results directory containing experiments",
    )
    parser.add_argument("--debug", action="store_true", help="Enable Gradio debug mode")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: auto)")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    args = parser.parse_args()

    run_xray(Path(args.results_dir), debug=args.debug, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
