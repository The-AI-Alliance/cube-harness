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
from pathlib import Path
from typing import Any, Callable

import gradio as gr
from PIL import Image

from agentlab2 import EXP_DIR
from agentlab2.analyze import inspect_results, xray_utils
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
    # One FileStorage per loaded experiment
    _storages: list[FileStorage] = field(default_factory=list, repr=False)
    # Parallel to self.trajectories: _traj_storages[i] is the storage that owns trajectories[i].
    # Index-based (not traj_id-based) so experiments with overlapping task IDs don't collide.
    _traj_storages: list[FileStorage] = field(default_factory=list, repr=False)
    # Names of currently selected experiments (for change-detection in the UI)
    _selected_exp_names: list[str] = field(default_factory=list, repr=False)
    # Per-storage timestamp tag (parsed from exp dir name); keyed by id(storage).
    # Always appended to agent_name so each trajectory is unambiguously identified.
    _exp_tags: dict[int, str] = field(default_factory=dict, repr=False)
    # Set to True once the background bulk-loading thread has finished
    _bg_loading_done: bool = field(default=True, repr=False)
    # Incremented on each load_experiments call; background threads check this to self-abort
    # when superseded by a newer load (prevents stale writes to self.trajectories).
    _bg_gen: int = field(default=0, repr=False)
    # Per-storage backfill names for backwards-compat (agent class short name from config); keyed by id(storage).
    _backfill_names: dict[int, str | None] = field(default_factory=dict, repr=False)
    # Per-storage config JSON strings for the Config tabs; keyed by id(storage).
    # Value: (agent_config_json, exp_config_json), both may be None if unavailable.
    _storage_configs: dict[int, tuple[str | None, str | None]] = field(default_factory=dict, repr=False)
    # Live polling: tracks which trajectories are done (skip on future ticks) and their mtimes
    _completed_ids: set[str] = field(default_factory=set, repr=False)
    _traj_mtimes: dict[str, float] = field(default_factory=dict, repr=False)
    # Timestamp of last detected file change — used for stale experiment detection
    _last_change_time: float = field(default=0.0, repr=False)

    def load_experiments(self, exp_dirs: list[Path]) -> bool:
        """Load trajectory metadata stubs from one or more experiment directories.

        Each directory gets its own FileStorage instance. Trajectories from all
        directories are merged into self.trajectories. The parallel _traj_storages
        list maps each trajectory to its owning storage by index, avoiding collisions
        when multiple experiments share identical task/episode IDs.

        Returns True if at least one trajectory was loaded.
        """
        # Increment generation FIRST so any running background thread sees the change
        # immediately and aborts before it can write stale data into our new trajectory list.
        self._bg_gen += 1
        self._storages = [FileStorage(d) for d in exp_dirs]
        self._selected_exp_names = [d.name for d in exp_dirs]
        self.trajectories = []
        self._traj_storages = []
        self._exp_tags = {}
        self._backfill_names = {}
        self._storage_configs = {}
        for exp_dir, storage in zip(exp_dirs, self._storages):
            trajs = storage.load_all_trajectory_metadata()
            self._load_experiment_config(exp_dir, storage)
            self._exp_tags[id(storage)] = xray_utils._parse_exp_date(exp_dir)
            for traj in trajs:
                self._maybe_backfill(traj, storage)
                self._apply_exp_tag(traj, storage)
            self.trajectories.extend(trajs)
            self._traj_storages.extend([storage] * len(trajs))
        self.selected_agent_key = None
        self.selected_task_id = None
        self.current_trajectory = None
        self.step = 0
        self._env_step_indices = []
        self._completed_ids = {t.id for t in self.trajectories if t.end_time is not None}
        self._traj_mtimes = {}
        for storage in self._storages:
            self._traj_mtimes.update(storage.list_trajectory_ids_with_mtime())
        self._last_change_time = time.time()
        self._bg_loading_done = False
        self._start_background_loading()
        return len(self.trajectories) > 0

    def load_experiment(self, exp_dir: Path) -> bool:
        """Convenience wrapper: load a single experiment directory."""
        return self.load_experiments([exp_dir])

    def _load_experiment_config(self, exp_dir: Path, storage: FileStorage) -> None:
        """Read experiment_config.json and store per-storage config JSON strings.

        Populates _storage_configs[id(storage)] with (agent_config_json, exp_config_json)
        and _backfill_names[id(storage)] with the agent class short name for backwards compat.

        Falls back to the first episode_configs/*.json when experiment_config.json
        is absent (e.g. experiments run before save_config() was added).
        """
        agent_cfg: dict = {}
        exp_cfg_display: dict = {}
        config_path = exp_dir / "experiment_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    exp_cfg = json.load(f)
                agent_cfg = exp_cfg.get("agent_config", {})
                exp_cfg_display = {**exp_cfg, "agent_config": "(see Agent Config tab)"}
            except Exception:
                pass
        if not agent_cfg:
            # Fallback: extract agent_config from the first available episode config
            episode_cfgs = (
                sorted((exp_dir / "episode_configs").glob("*.json")) if (exp_dir / "episode_configs").exists() else []
            )
            for ep_path in episode_cfgs:
                try:
                    with open(ep_path) as f:
                        ep_cfg = json.load(f)
                    agent_cfg = ep_cfg.get("agent_config", {})
                    if agent_cfg:
                        break
                except Exception:
                    continue
        agent_type = agent_cfg.get("_type", "") if agent_cfg else ""
        self._backfill_names[id(storage)] = agent_type.split(".")[-1] if agent_type else None
        self._storage_configs[id(storage)] = (
            json.dumps(agent_cfg, indent=2) if agent_cfg else None,
            json.dumps(exp_cfg_display, indent=2) if exp_cfg_display else None,
        )

    def _maybe_backfill(self, traj: Trajectory, storage: FileStorage) -> None:
        """Inject backfill name into a trajectory's metadata if agent_name is absent."""
        name = self._backfill_names.get(id(storage))
        if name and "agent_name" not in traj.metadata:
            traj.metadata["agent_name"] = name

    def _apply_exp_tag(self, traj: Trajectory, storage: FileStorage) -> None:
        """Append this storage's timestamp tag to traj's agent_name."""
        tag = self._exp_tags.get(id(storage), "")
        if tag:
            traj.metadata["agent_name"] = traj.metadata.get("agent_name", "unknown") + f" [{tag}]"

    def get_config_jsons(self) -> tuple[str, str]:
        """Return (agent_config_json, exp_config_json) for the currently selected agent.

        Looks up which storage owns the selected agent's trajectories, then returns
        that storage's config strings. Returns ("", "") when nothing is selected or found.
        """
        if self.selected_agent_key is None:
            return "", ""
        for i, traj in enumerate(self.trajectories):
            if traj.metadata.get("agent_name") == self.selected_agent_key:
                storage = self._traj_storages[i]
                agent_cfg, exp_cfg = self._storage_configs.get(id(storage), (None, None))
                return agent_cfg or "", exp_cfg or ""
        return "", ""

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
        if not self._storages:
            self._bg_loading_done = True
            return

        # Capture a snapshot to avoid closure over mutable state
        my_gen = self._bg_gen  # This thread's generation; abort if superseded
        # Capture hard references to the owned lists so that load_experiments
        # reassigning self.trajectories/self._traj_storages never redirects our writes.
        my_trajs = self.trajectories
        my_storages = list(self._traj_storages)  # index-parallel snapshot; no traj_id collision

        def _load_all() -> None:
            for i, traj in enumerate(my_trajs):
                # Abort if a newer load_experiments call has started
                if self._bg_gen != my_gen:
                    return
                # Skip if already fully loaded (e.g. user clicked it first)
                if traj.steps:
                    continue
                storage = my_storages[i]
                try:
                    full = storage.load_trajectory(traj.id)
                    self._maybe_backfill(full, storage)
                    self._apply_exp_tag(full, storage)
                    if self._bg_gen == my_gen:
                        my_trajs[i] = full
                        if self.current_trajectory is not None and self.current_trajectory.id == traj.id:
                            self.current_trajectory = full
                            self._env_step_indices = self._build_env_indices()
                except Exception:
                    pass  # leave stub; table will show "-" for unavailable stats
            if self._bg_gen == my_gen:
                self._bg_loading_done = True

        thread = threading.Thread(target=_load_all, daemon=True)
        thread.start()

    def refresh_experiment(self) -> bool:
        """Incrementally reload new or changed trajectories from disk. Returns True if anything changed.

        Uses mtime-based change detection: only trajectories whose files have changed since the
        last check are reloaded. Completed trajectories (end_time set) are skipped entirely.
        Called on each bg_timer tick while the experiment is still running.
        """
        if not self._storages:
            return False
        changed = False
        known_ids = {t.id for t in self.trajectories}

        for storage in self._storages:
            id_mtimes = storage.list_trajectory_ids_with_mtime()
            for traj_id, mtime in id_mtimes.items():
                if traj_id in self._completed_ids:
                    continue
                prev_mtime = self._traj_mtimes.get(traj_id, 0.0)
                if mtime <= prev_mtime and traj_id in known_ids:
                    continue
                try:
                    full = storage.load_trajectory(traj_id)
                    self._maybe_backfill(full, storage)
                    self._apply_exp_tag(full, storage)
                    self._traj_mtimes[traj_id] = mtime
                    changed = True
                    # Find the existing slot owned by this storage (avoids ID collision)
                    idx = next(
                        (
                            i
                            for i, t in enumerate(self.trajectories)
                            if t.id == traj_id and self._traj_storages[i] is storage
                        ),
                        None,
                    )
                    if idx is not None:
                        self.trajectories[idx] = full
                        self._traj_storages[idx] = storage
                        if self.current_trajectory is not None and self.current_trajectory.id == traj_id:
                            self.current_trajectory = full
                            self._env_step_indices = self._build_env_indices()
                    else:
                        self.trajectories.append(full)
                        self._traj_storages.append(storage)
                        known_ids.add(traj_id)
                    if full.end_time is not None:
                        self._completed_ids.add(traj_id)
                except Exception:
                    pass
        if changed:
            self._last_change_time = time.time()
        return changed

    def is_experiment_complete(self) -> bool:
        """Return True when every known trajectory has finished (end_time is set)."""
        if not self.trajectories:
            return False
        return all(t.end_time is not None for t in self.trajectories)

    def is_experiment_stale(self, timeout_s: float = 1200.0) -> bool:
        """Return True if no file changes have been detected for timeout_s seconds.

        Used to stop the live-polling timer when an experiment appears to have stalled
        (e.g., the runner crashed without setting end_time on every trajectory).
        Default timeout is 20 minutes.
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
        """Select a trajectory by ID; loads full steps lazily if not yet loaded.

        When multiple experiments share the same task/episode IDs, prefers the trajectory
        whose agent_name matches selected_agent_key, falling back to the first match.
        """
        # Prefer the slot whose agent matches the current selection (multi-experiment safety)
        idx = next(
            (
                i
                for i, t in enumerate(self.trajectories)
                if t.id == traj_id and t.metadata.get("agent_name") == self.selected_agent_key
            ),
            None,
        )
        if idx is None:
            idx = next((i for i, t in enumerate(self.trajectories) if t.id == traj_id), None)
        if idx is None:
            self.current_trajectory = None
            self.step = 0
            self._env_step_indices = []
            return
        traj = self.trajectories[idx]
        # Stub has steps=[]; load full trajectory on first access and cache it in place
        if not traj.steps:
            storage = self._traj_storages[idx]
            try:
                traj = storage.load_trajectory(traj_id)
                self._maybe_backfill(traj, storage)
                self._apply_exp_tag(traj, storage)
                self.trajectories[idx] = traj
                self._traj_storages[idx] = storage
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
.help-content {
    max-height: 260px;
    overflow-y: auto;
    padding-right: 8px;
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
/* Experiments table: hide the "select all" header checkbox */
#exp_table table thead tr th:first-child input[type="checkbox"] {
    display: none;
}
/* Experiments table: narrow the checkbox column */
#exp_table table th:first-child,
#exp_table table td:first-child {
    width: 36px !important;
    min-width: 36px !important;
    max-width: 36px !important;
}
/* Experiments table: prevent renaming column headers */
#exp_table table thead th [contenteditable] {
    pointer-events: none !important;
}
"""

_FORCE_LIGHT_JS = "() => { document.body.classList.remove('dark'); }"

_SHORTCUT_JS = """
<script>
function shortcuts(e) {
    if (!e.shiftKey || e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = e.target.tagName.toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        const prev = document.querySelector('#xray_prev_btn button');
        if (prev) prev.click();
    } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        const next = document.querySelector('#xray_next_btn button');
        if (next) next.click();
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


def _render_thoughts_panel(text: str) -> str:
    """Render the agent's thoughts as a styled HTML panel (same green as action, small bottom gap)."""
    safe = html_lib.escape(text)
    safe = safe.replace("\n", "<br>")
    safe = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", safe)
    return (
        '<div class="info-panel" style="background:#f0fdf4; border-color:#bbf7d0; margin-bottom:6px;">'
        '<div class="info-panel-title" style="background:#dcfce7; color:#15803d;">💭 Thoughts</div>'
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

    def _load_and_build_hierarchy() -> tuple[str, Any, Any, Any, StepId, gr.Tab, gr.Tab, gr.Tab, str, str]:
        """Build experiment stats + agent/task/seed tables after state.load_experiments().

        Auto-selects the first agent, first task, and first seed when available.
        Returns the 10-tuple expected by both on_experiments_change and on_bg_load_tick callers.
        """
        exp_stats = xray_utils.compute_experiment_stats(state.trajectories)
        agent_rows = xray_utils.build_agent_table(state.trajectories)

        # Auto-select first agent
        if not agent_rows:
            seed_rows_: list[dict[str, Any]] = []
            task_rows_: list[dict[str, Any]] = []
            tab_labels = _make_tab_labels(agent_rows, task_rows_, seed_rows_)
            return exp_stats, _rows_to_table(agent_rows), [], [], StepId(), *tab_labels, *state.get_config_jsons()
        first_agent_key = agent_rows[0]["agent_name"]
        state.select_agent(first_agent_key)
        agent_table_data = _rows_to_table(agent_rows, first_agent_key, "agent_name")

        # Auto-select first task
        task_rows = xray_utils.build_task_table(state.trajectories, first_agent_key)
        if not task_rows:
            tab_labels = _make_tab_labels(agent_rows, task_rows, [])
            return (
                exp_stats,
                agent_table_data,
                _rows_to_table(task_rows),
                [],
                StepId(),
                *tab_labels,
                *state.get_config_jsons(),
            )
        first_task_id = task_rows[0]["task_id"]
        state.select_task(first_task_id)
        task_table_data = _rows_to_table(task_rows, first_task_id, "task_id")

        # Auto-select first seed/trajectory
        seed_rows = xray_utils.build_seed_table(state.trajectories, first_agent_key, first_task_id)
        if not seed_rows:
            tab_labels = _make_tab_labels(agent_rows, task_rows, seed_rows)
            return (
                exp_stats,
                agent_table_data,
                task_table_data,
                _rows_to_table(seed_rows),
                StepId(),
                *tab_labels,
                *state.get_config_jsons(),
            )
        first_traj_id = seed_rows[0]["traj_id"]
        state.select_trajectory(first_traj_id)
        seed_table_data = _rows_to_table(seed_rows, first_traj_id, "traj_id")

        tab_labels = _make_tab_labels(agent_rows, task_rows, seed_rows)
        return (
            exp_stats,
            agent_table_data,
            task_table_data,
            seed_table_data,
            StepId(step=0),
            *tab_labels,
            *state.get_config_jsons(),
        )

    def on_experiments_change(
        exp_df: Any,
    ) -> tuple[str, Any, Any, Any, StepId, gr.Tab, gr.Tab, gr.Tab, str, str, gr.Timer]:
        """Handle checkbox changes in the Experiments table.

        Extracts selected experiment names, loads them (merging trajectories), and
        rebuilds the agent/task/seed hierarchy. Returns gr.skip() for all outputs
        when the selection hasn't actually changed (avoids spurious Gradio events).
        """
        _empty = (
            "",
            None,
            None,
            None,
            StepId(),
            gr.Tab(label="Agents (0)"),
            gr.Tab(label="Tasks (0)"),
            gr.Tab(label="Seeds (0)"),
            "",
            "",
            gr.Timer(active=False),
        )
        if exp_df is None or len(exp_df) == 0:
            return _empty
        selected_names = [str(exp_df.iloc[i, 1]) for i in range(len(exp_df)) if exp_df.iloc[i, 0]]
        if set(selected_names) == set(state._selected_exp_names):
            return tuple(gr.skip() for _ in range(11))  # type: ignore[return-value]
        if not selected_names:
            state._bg_gen += 1
            state._selected_exp_names = []
            state.trajectories = []
            state.selected_agent_key = None
            return _empty
        exp_dirs = [state.results_dir / name for name in selected_names]
        state.load_experiments(exp_dirs)
        hierarchy = _load_and_build_hierarchy()
        timer_active = not state._bg_loading_done
        return (*hierarchy, gr.Timer(active=timer_active))

    def on_archive_selected() -> tuple[Any, str, Any, Any, Any, StepId, gr.Tab, gr.Tab, gr.Tab, str, str, gr.Timer]:
        """Archive all currently selected experiments and reset state."""
        for name in list(state._selected_exp_names):
            xray_utils.archive_experiment(state.results_dir, name)
        state._bg_gen += 1
        state._selected_exp_names = []
        state.trajectories = []
        state.selected_agent_key = None
        _empty_hierarchy = (
            "",
            None,
            None,
            None,
            StepId(),
            gr.Tab(label="Agents (0)"),
            gr.Tab(label="Tasks (0)"),
            gr.Tab(label="Seeds (0)"),
            "",
            "",
            gr.Timer(active=False),
        )
        return (_exp_table_value(), *_empty_hierarchy)

    def on_select_agent(
        evt: gr.SelectData, agent_df: Any
    ) -> tuple[Any, Any, Any, StepId, gr.Tab, gr.Tab, gr.Tab, str, str]:
        if evt is None or evt.index is None or agent_df is None or len(agent_df) == 0:
            return (
                [],
                [],
                [],
                StepId(),
                gr.Tab(label="Agents (0)"),
                gr.Tab(label="Tasks (0)"),
                gr.Tab(label="Seeds (0)"),
                "",
                "",
            )
        row = evt.index[0]
        # Strip HTML tags to get the raw key value
        agent_key = re.sub(r"<[^>]+>", "", str(agent_df.iloc[row, 0]))
        state.select_agent(agent_key)
        agent_rows = xray_utils.build_agent_table(state.trajectories)
        agent_table_data = _rows_to_table(agent_rows, agent_key, "agent_name")
        task_rows = xray_utils.build_task_table(state.trajectories, agent_key)
        if not task_rows:
            tab_labels = _make_tab_labels(agent_rows, task_rows, [])
            return agent_table_data, _rows_to_table(task_rows), [], StepId(), *tab_labels, *state.get_config_jsons()
        # Auto-select first task and first seed
        first_task_id = task_rows[0]["task_id"]
        state.select_task(first_task_id)
        task_table_data = _rows_to_table(task_rows, first_task_id, "task_id")
        seed_rows = xray_utils.build_seed_table(state.trajectories, agent_key, first_task_id)
        if not seed_rows:
            tab_labels = _make_tab_labels(agent_rows, task_rows, seed_rows)
            return (
                agent_table_data,
                task_table_data,
                _rows_to_table(seed_rows),
                StepId(),
                *tab_labels,
                *state.get_config_jsons(),
            )
        first_traj_id = seed_rows[0]["traj_id"]
        state.select_trajectory(first_traj_id)
        seed_table_data = _rows_to_table(seed_rows, first_traj_id, "traj_id")
        tab_labels = _make_tab_labels(agent_rows, task_rows, seed_rows)
        return (
            agent_table_data,
            task_table_data,
            seed_table_data,
            StepId(step=0),
            *tab_labels,
            *state.get_config_jsons(),
        )

    def on_select_task(evt: gr.SelectData, task_df: Any) -> tuple[Any, Any, StepId, gr.Tab]:
        if evt is None or evt.index is None or task_df is None or len(task_df) == 0:
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
        if evt is None or evt.index is None or seed_df is None or len(seed_df) == 0:
            return [], StepId(step=0)
        row = evt.index[0]
        traj_id = re.sub(r"<[^>]+>", "", str(seed_df.iloc[row, 1]))  # col 0 is status emoji
        state.select_trajectory(traj_id)
        if state.selected_agent_key is None or state.selected_task_id is None:
            return [], StepId(step=0)
        seed_rows = xray_utils.build_seed_table(state.trajectories, state.selected_agent_key, state.selected_task_id)
        seed_table_data = _rows_to_table(seed_rows, traj_id, "traj_id")
        return seed_table_data, StepId(step=0)

    def on_bg_load_tick() -> tuple[Any, Any, Any, Any, str, gr.Timer, gr.Tab, gr.Tab, gr.Tab]:
        """Periodic refresh handler: bulk-loads stubs, then live-polls for new/changed trajectories.

        Two phases share a single timer:
        1. While _bg_loading_done is False: background thread is still bulk-loading stubs.
        2. Once _bg_loading_done is True: calls refresh_experiment() to pick up new or
           changed trajectory files written by a running experiment. Timer deactivates only
           when is_experiment_complete() returns True (all trajectories have end_time set).
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
            xray_utils.build_seed_table(state.trajectories, active_agent, task_key) if active_agent and task_key else []
        )
        traj_id = state.current_trajectory.id if state.current_trajectory else None
        seed_table_data = _rows_to_table(seed_rows, traj_id, "traj_id")

        n_total = len(state.trajectories)
        # Use end_time directly (not _completed_ids) to avoid undercounting when multiple
        # experiments share the same task/episode IDs (the set deduplicates colliding IDs).
        n_completed = sum(1 for t in state.trajectories if t.end_time is not None)
        n_running = sum(1 for t in state.trajectories if t.end_time is None)
        # Per-agent breakdown (shown when > 1 agent loaded, e.g. multi-experiment)
        agent_names = sorted({t.metadata.get("agent_name", "unknown") for t in state.trajectories})
        per_agent: list[tuple[str, int, int, int]] | None = None
        if len(agent_names) > 1:
            per_agent = []
            for aname in agent_names:
                atrajs = [t for t in state.trajectories if t.metadata.get("agent_name", "unknown") == aname]
                per_agent.append(
                    (
                        aname,
                        sum(1 for t in atrajs if t.end_time is not None),
                        len(atrajs),
                        sum(1 for t in atrajs if t.end_time is None),
                    )
                )
        progress_html = xray_utils.build_progress_html(
            n_completed, n_total, n_running, per_agent, state._selected_exp_names or None
        )

        experiment_done = state.is_experiment_complete() or state.is_experiment_stale()
        still_active = not state._bg_loading_done or not experiment_done
        timer_update = gr.Timer(active=still_active)
        tab_labels = _make_tab_labels(agent_rows, task_rows, seed_rows)
        return exp_stats, agent_table_data, task_table_data, seed_table_data, progress_html, timer_update, *tab_labels

    def navigate_prev() -> StepId:
        """Step backward; reads state.step from closure so JS button.click() works too."""
        step = max(0, state.step - 1)
        state.step = step
        return StepId(step=step)

    def navigate_next() -> StepId:
        """Step forward; reads state.step from closure so JS button.click() works too."""
        step = min(state.total_ui_steps() - 1, state.step + 1)
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
        step_info = f"Step {state.step + 1}/{state.total_ui_steps()}"
        return header + f" │ {reward_emoji} Reward: {reward:.2f} │ {step_info}"

    def update_timeline() -> str:
        return xray_utils.generate_timeline_html(state.current_trajectory, state.step)

    def update_trajectory_stats() -> str:
        if not state.current_trajectory:
            return ""
        stats = xray_utils.compute_trajectory_stats(state.current_trajectory)

        parts: list[str] = []
        if stats["duration"] is not None:
            parts.append(f"⏱️ **{xray_utils.format_duration(stats['duration'])}**")

        prompt_tokens = int(stats["prompt_tokens"])
        completion_tokens = int(stats["completion_tokens"])
        cached_tokens = int(stats["cached_tokens"])
        cache_creation_tokens = int(stats["cache_creation_tokens"])
        cost = float(stats["cost"])

        if prompt_tokens > 0:
            parts.append(f"📊 prompt: **{prompt_tokens:,}**")
            parts.append(f"completion: **{completion_tokens:,}**")
            parts.append(f"total: **{prompt_tokens + completion_tokens:,}**")
            if cached_tokens > 0:
                cache_pct = cached_tokens / prompt_tokens * 100
                parts.append(f"cached: **{cached_tokens:,}** ({cache_pct:.0f}%)")
            if cache_creation_tokens > 0:
                parts.append(f"cache_created: **{cache_creation_tokens:,}**")
        if cost > 0:
            parts.append(f"💰 **${cost:.4f}**")

        return " │ ".join(parts)

    def get_task_goal() -> str:
        """Return the task goal as a rendered HTML panel."""
        return _render_goal_panel(xray_utils.get_task_goal(state.current_trajectory))

    def get_agent_action_md() -> str:
        """Return the current step's thoughts (if any) and action as stacked HTML panels."""
        agent_out = state.get_agent_output()
        panels = []
        if agent_out and agent_out.thoughts:
            thoughts = agent_out.thoughts.strip()
            if len(thoughts) > 500:
                thoughts = thoughts[:500] + "…"
            panels.append(_render_thoughts_panel(thoughts))
        panels.append(_render_action_panel(xray_utils.get_agent_action_markdown(agent_out)))
        return "\n".join(panels)

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

    _MAX_EXTRA_CHAT_BRANCHES = 3

    def _render_chat() -> tuple:
        agent_out = state.get_agent_output()
        items = list(xray_utils.get_chat_branches(agent_out).items())

        main_html = items[0][1] if items else "<em>No agent action follows this observation (terminal step).</em>"
        extra_items = items[1:]

        results: list = [main_html]
        for i in range(_MAX_EXTRA_CHAT_BRANCHES):
            if i < len(extra_items):
                name, html = extra_items[i]
                results.append(gr.Tab(label=name.capitalize(), visible=True))
                results.append(html)
            else:
                results.append(gr.Tab(visible=False))
                results.append("")
        return tuple(results)

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
    # Experiment-level analysis tabs (lazy, rendered on tab select)
    # ------------------------------------------------------------------

    def _render_constants_variables() -> tuple[list[list], list[list]]:
        if not state.trajectories:
            return [], []
        df = inspect_results.trajectories_to_df(state.trajectories)
        if df is None:
            return [], []
        const_df, var_df = inspect_results.format_constants_and_variables(df)
        const_rows = const_df.values.tolist() if not const_df.empty else []
        var_rows = var_df.values.tolist() if not var_df.empty else []
        return const_rows, var_rows

    def _render_global_report() -> list[list]:
        if not state.trajectories:
            return []
        df = inspect_results.trajectories_to_df(state.trajectories)
        if df is None:
            return []
        inspect_results.set_index_from_variables(df)
        report = inspect_results.global_report(df)
        report = report.reset_index()
        for col in report.columns:
            report[col] = report[col].astype(str)
        return report.values.tolist()

    def _render_error_report() -> str:
        if not state.trajectories:
            return "No trajectories loaded."
        df = inspect_results.trajectories_to_df(state.trajectories)
        if df is None:
            return "No data."
        return inspect_results.error_report(df)

    # ------------------------------------------------------------------
    # Tab activation helpers — no-arg named functions avoid lambda warnings.
    # Gradio tab.select fires with no extra inputs, so these take no args.
    # ------------------------------------------------------------------

    def _activate_screenshots() -> str:
        return "Screenshots"

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
        active_tab = gr.State(value="Chat Messages")
        step_id = gr.State(value=StepId())

        with gr.Tabs():
            with gr.Tab("Help"):
                gr.Markdown(
                    """\
## AgentLab2 XRay

### Loading experiments
1. Open the **Experiments** tab — check one or more rows to load them simultaneously.
2. Use **↺ Refresh** to pick up new experiments; **🗃 Archive selected** moves them to `_archive/`.
3. When an experiment is running, the viewer polls for new trajectories every second until complete.

### Browsing results
4. Drill down via the **Agents → Tasks → Seeds** tabs to select a specific episode.
5. The **Dashboard** tab shows a live progress bar and aggregate stats (reward, tokens, cost).
6. **Agent Config** / **Exp Config** tabs display the configuration used for the experiment.

### Inspecting a trajectory
7. The **timeline** shows one segment per step; width scales with wall-clock duration.
   - Blue = environment time, green = agent time. Coloured strips on top = profiling breakdown.
   - Click any segment to jump to that step. The gold border marks the current step.
   - Green / red bottom border = success / failure at that step.
8. The **💭 Rationale** panel (when available) shows the chain-of-thought that led to the action.
9. The **🤖 Action** panel shows the action(s) the agent took.
10. **Navigate steps** with the ◀ / ▶ buttons or **Shift + ← / →** arrow keys.

### Tabs (lazy — only the active tab re-renders on step change)
- **Chat Messages**: full LLM prompt + response; extra branches for auxiliary LLM calls (e.g. summarize).
- **Screenshots**: current and previous environment screenshots.
- **Step Details**: detailed env observation + agent output with token stats.
- **AXTree**: raw accessibility tree text.
- **Task Error**: environment and agent errors for this step.
- **Logs**: step info dict and trajectory metadata.
- **Debug**: raw JSON for the env step, LLM calls, and tool schemas.
""",
                    elem_classes="help-content",
                )
            with gr.Tab("Experiments"):
                with gr.Row():
                    exp_refresh_btn = gr.Button("↺ Refresh", scale=0, size="sm")
                    exp_archive_btn = gr.Button("🗃 Archive selected", scale=0, size="sm", variant="secondary")
                exp_table = gr.DataFrame(
                    headers=["☑", "experiment", "date", "n_trajs"],
                    datatype=["bool", "str", "str", "number"],
                    col_count=(4, "fixed"),
                    interactive=True,
                    static_columns=[1, 2, 3],
                    max_height=260,
                    show_label=False,
                    elem_id="exp_table",
                )
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
            with gr.Tab("Constants & Variables") as cv_tab:
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Constants** (same across all trajectories)")
                        cv_const_table = gr.DataFrame(
                            headers=["parameter", "value"],
                            max_height=400,
                            show_label=False,
                            interactive=False,
                        )
                    with gr.Column():
                        gr.Markdown("**Variables** (differ across trajectories)")
                        cv_var_table = gr.DataFrame(
                            headers=["parameter", "n_unique", "sample_values"],
                            max_height=400,
                            show_label=False,
                            interactive=False,
                        )
            with gr.Tab("Global Report") as report_tab:
                report_table = gr.DataFrame(
                    max_height=400,
                    show_label=False,
                    interactive=False,
                )
            with gr.Tab("Error Report") as err_report_tab:
                err_report_md = gr.Markdown()

        # Timer: ticks every 1s to bulk-load stubs and then live-poll for new/changed trajectories.
        # Starts inactive; activated on experiment select; deactivates when experiment is complete.
        bg_timer = gr.Timer(value=1.0, active=False)

        with gr.Row(variant="panel", elem_classes="compact-header"):
            with gr.Column(scale=1, min_width=200):
                header_info = gr.Markdown("**Select a trajectory**")
            with gr.Column(scale=3):
                stats_display = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=0, min_width=40):
                prev_btn = gr.Button("◀", size="sm", elem_id="xray_prev_btn", min_width=36)
            with gr.Column(scale=1):
                timeline_html = gr.HTML(label="Timeline")
            with gr.Column(scale=0, min_width=40):
                next_btn = gr.Button("▶", size="sm", elem_id="xray_next_btn", min_width=36)

        with gr.Row(visible=True, elem_id="timeline_click_input"):
            timeline_click_input = gr.Number(show_label=False, container=False)

        # Always-visible panels: task goal (stable per trajectory) + agent action (per step)
        with gr.Row():
            with gr.Column(scale=2):
                task_goal_md = gr.HTML(value="")
            with gr.Column(scale=3):
                agent_action_md = gr.HTML(value="")

        with gr.Tabs():
            with gr.Tab("Chat Messages") as chat_tab:
                with gr.Tabs():
                    with gr.Tab("Main"):
                        chat_act_md = gr.HTML()
                    with gr.Tab("Branch 1", visible=False) as chat_branch_tab_1:
                        chat_branch_md_1 = gr.HTML()
                    with gr.Tab("Branch 2", visible=False) as chat_branch_tab_2:
                        chat_branch_md_2 = gr.HTML()
                    with gr.Tab("Branch 3", visible=False) as chat_branch_tab_3:
                        chat_branch_md_3 = gr.HTML()

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

            with gr.Tab("Step Details") as step_details_tab:
                step_details = gr.Markdown(
                    value="Select a trajectory to view step details",
                    elem_classes="step-details",
                )

            with gr.Tab("AXTree") as axtree_tab:
                axtree_code = gr.Code(language=None, show_label=False, max_lines=40)

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

        def _exp_table_value() -> list[list[Any]]:
            rows = xray_utils.get_experiments_table_rows(state.results_dir)
            return [[r["selected"], r["experiment"], r["date"], r["n_trajs"]] for r in rows]

        _hierarchy_outputs = [
            experiment_stats,
            agent_table,
            task_table,
            seed_table,
            step_id,
            agents_tab,
            tasks_tab,
            seeds_tab,
            agent_config_code,
            exp_config_code,
            bg_timer,
        ]

        exp_table.change(fn=on_experiments_change, inputs=exp_table, outputs=_hierarchy_outputs)
        exp_refresh_btn.click(fn=_exp_table_value, outputs=exp_table)
        exp_archive_btn.click(fn=on_archive_selected, outputs=[exp_table, *_hierarchy_outputs])

        bg_timer.tick(
            fn=on_bg_load_tick,
            outputs=[
                experiment_stats,
                agent_table,
                task_table,
                seed_table,
                progress_bar,
                bg_timer,
                agents_tab,
                tasks_tab,
                seeds_tab,
            ],
        )

        agent_table.select(
            fn=on_select_agent,
            inputs=agent_table,
            outputs=[
                agent_table,
                task_table,
                seed_table,
                step_id,
                agents_tab,
                tasks_tab,
                seeds_tab,
                agent_config_code,
                exp_config_code,
            ],
        )
        task_table.select(fn=on_select_task, inputs=task_table, outputs=[task_table, seed_table, step_id, seeds_tab])
        seed_table.select(fn=on_select_seed, inputs=seed_table, outputs=[seed_table, step_id])

        # Timeline click
        timeline_click_input.change(fn=handle_timeline_click, inputs=timeline_click_input, outputs=step_id)

        # Navigation buttons — handlers read state.step from closure (inputs=[]) so that
        # JS button.click() also works without Gradio losing the gr.State value.
        prev_btn.click(fn=navigate_prev, inputs=[], outputs=step_id)
        next_btn.click(fn=navigate_next, inputs=[], outputs=step_id)

        # Always-rendered on step change
        step_id.change(fn=get_compact_header_info, outputs=header_info)
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
            fn=if_active("Step Details")(_render_step_details),
            inputs=[active_tab, step_id],
            outputs=step_details,
        )
        step_id.change(
            fn=if_active("AXTree")(_render_axtree),
            inputs=[active_tab, step_id],
            outputs=axtree_code,
        )
        _chat_outputs = [
            chat_act_md,
            chat_branch_tab_1,
            chat_branch_md_1,
            chat_branch_tab_2,
            chat_branch_md_2,
            chat_branch_tab_3,
            chat_branch_md_3,
        ]
        step_id.change(
            fn=if_active("Chat Messages", 7)(_render_chat),
            inputs=[active_tab, step_id],
            outputs=_chat_outputs,
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

        step_details_tab.select(fn=_activate_step_details, outputs=active_tab)
        step_details_tab.select(fn=_render_step_details, outputs=step_details)

        axtree_tab.select(fn=_activate_axtree, outputs=active_tab)
        axtree_tab.select(fn=_render_axtree, outputs=axtree_code)

        chat_tab.select(fn=_activate_chat, outputs=active_tab)
        chat_tab.select(fn=_render_chat, outputs=_chat_outputs)

        error_tab.select(fn=_activate_error, outputs=active_tab)
        error_tab.select(fn=_render_error, outputs=error_md)

        logs_tab.select(fn=_activate_logs, outputs=active_tab)
        logs_tab.select(fn=_render_logs, outputs=logs_md)

        debug_tab.select(fn=_activate_debug, outputs=active_tab)
        debug_tab.select(fn=_render_debug, outputs=[raw_json, llm_calls_code, llm_tools_code])

        cv_tab.select(fn=_render_constants_variables, outputs=[cv_const_table, cv_var_table])
        report_tab.select(fn=_render_global_report, outputs=report_table)
        err_report_tab.select(fn=_render_error_report, outputs=err_report_md)

        # Populate experiment table on page load
        demo.load(fn=_exp_table_value, outputs=exp_table)

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
    parser = argparse.ArgumentParser(description="AgentLab2 XRay Experiment Viewer")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(EXP_DIR),
        help="Path to results directory containing experiments",
    )
    parser.add_argument("--debug", action="store_true", help="Enable Gradio debug mode")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: auto)")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    args = parser.parse_args()

    run_xray(Path(args.results_dir), debug=args.debug, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
