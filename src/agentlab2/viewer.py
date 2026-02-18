"""
Experiment viewer for AgentLab2 (X-Ray 2).

A Gradio-based UI for exploring experiment results, trajectories, and agent outputs.
Supports live monitoring of in-progress experiments via filesystem polling.
"""

import argparse
import json
from dataclasses import dataclass, field
from os.path import expanduser
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from agentlab2.core import AgentOutput, EnvironmentOutput, Trajectory, TrajectoryStep
from agentlab2.storage import FileStorage

_STATUS_EMOJIS = {"running": "\u23f3", "success": "\u2705", "error": "\u274c", "completed": "\u2b1c"}
_LIVE_REFRESH_INTERVAL_S = 3.0


def _trajectory_status(traj: Trajectory) -> str:
    if traj.end_time is None:
        return "running"
    if traj.reward_info and traj.reward_info.get("reward", 0) > 0:
        return "success"
    for step in traj.steps:
        if hasattr(step.output, "error") and step.output.error is not None:
            return "error"
    return "completed"


@dataclass
class TrajectoryId:
    """Identifies a specific trajectory within an experiment."""

    exp_dir: str | None = None
    trajectory_name: str | None = None


@dataclass
class StepId:
    """Identifies a specific step within a trajectory."""

    trajectory_id: TrajectoryId | None = None
    step: int = 0


@dataclass
class ViewerState:
    """State for the viewer application."""

    results_dir: Path
    exp_dirs: list[Path] = field(default_factory=list)
    trajectories: dict[str, Trajectory] = field(default_factory=dict)
    current_trajectory: Trajectory | None = None
    step: int = 0
    _current_exp_dir: Path | None = field(default=None)
    _completed_ids: set[str] = field(default_factory=set)
    _traj_mtimes: dict[str, float] = field(default_factory=dict)
    _expected_total: int | None = field(default=None)

    def load_experiment(self, exp_dir: Path) -> dict[str, Any]:
        """Load all trajectories from an experiment directory."""
        self.exp_dirs = [exp_dir]
        self._current_exp_dir = exp_dir
        self.trajectories = {}
        self._completed_ids = set()
        self._traj_mtimes = {}

        traj_dir = exp_dir / "trajectories"
        if not traj_dir.exists():
            return {"error": f"No trajectories directory found in {exp_dir}"}

        storage = FileStorage(exp_dir)
        trajectories = storage.load_all_trajectories()

        for n, traj in enumerate(trajectories):
            key = traj.id or f"traj{n}"
            self.trajectories[key] = traj
            if traj.end_time is not None:
                self._completed_ids.add(key)

        self._traj_mtimes = storage.list_trajectory_ids_with_mtime()

        config_dir = exp_dir / "episode_configs"
        if config_dir.exists():
            self._expected_total = len(list(config_dir.glob("episode_*_task_*.json")))
        else:
            self._expected_total = None

        return {"loaded": len(self.trajectories)}

    def refresh_experiment(self) -> bool:
        """Incrementally reload changed/new trajectories. Returns True if anything changed."""
        if self._current_exp_dir is None:
            return False
        storage = FileStorage(self._current_exp_dir)
        id_mtimes = storage.list_trajectory_ids_with_mtime()
        changed = False
        for traj_id, mtime in id_mtimes.items():
            if traj_id in self._completed_ids:
                continue
            prev_mtime = self._traj_mtimes.get(traj_id, 0.0)
            if mtime > prev_mtime or traj_id not in self.trajectories:
                try:
                    traj = storage.load_trajectory(traj_id)
                except Exception:
                    continue
                self.trajectories[traj_id] = traj
                self._traj_mtimes[traj_id] = mtime
                changed = True
                if traj.end_time is not None:
                    self._completed_ids.add(traj_id)

        config_dir = self._current_exp_dir / "episode_configs"
        if config_dir.exists():
            self._expected_total = len(list(config_dir.glob("episode_*_task_*.json")))

        return changed

    def is_experiment_complete(self) -> bool:
        if not self.trajectories:
            return False
        return all(traj_id in self._completed_ids for traj_id in self.trajectories)

    def select_trajectory(self, traj_id: str) -> None:
        """Select a trajectory by ID."""
        if traj_id in self.trajectories:
            self.current_trajectory = self.trajectories[traj_id]
            self.step = 0

    def get_env_steps(self) -> list[tuple[int, EnvironmentOutput]]:
        """Get all environment output steps with their indices."""
        if not self.current_trajectory:
            return []
        return [
            (i, step.output)
            for i, step in enumerate(self.current_trajectory.steps)
            if isinstance(step.output, EnvironmentOutput)
        ]

    def get_agent_steps(self) -> list[tuple[int, AgentOutput]]:
        """Get all agent output steps with their indices."""
        if not self.current_trajectory:
            return []
        return [
            (i, step.output)
            for i, step in enumerate(self.current_trajectory.steps)
            if isinstance(step.output, AgentOutput)
        ]

    def get_step_at(self, idx: int) -> EnvironmentOutput | AgentOutput | None:
        """Get step at specific index."""
        if not self.current_trajectory or idx < 0 or idx >= len(self.current_trajectory.steps):
            return None
        return self.current_trajectory.steps[idx].output

    def get_trajectory_step_at(self, idx: int) -> TrajectoryStep | None:
        """Get full TrajectoryStep (with timing info) at specific index."""
        if not self.current_trajectory or idx < 0 or idx >= len(self.current_trajectory.steps):
            return None
        return self.current_trajectory.steps[idx]

    def total_steps(self) -> int:
        """Total number of steps in current trajectory."""
        if not self.current_trajectory:
            return 0
        return len(self.current_trajectory.steps)


# CSS styling
_CSS = """
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
.success-box {
    background: #dcfce7;
    border: 1px solid #22c55e;
    border-radius: 6px;
    padding: 8px 12px;
    margin-top: 8px;
}
.action-card {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 6px;
    padding: 12px;
    margin: 8px 0;
}
code {
    white-space: pre-wrap;
}
th {
    white-space: normal !important;
    word-wrap: break-word !important;
}
.nav-buttons button {
    min-width: 80px;
}
#timeline_click_input {
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}
"""

# Keyboard shortcut JavaScript
_SHORTCUT_JS = """
<script>
function shortcuts(e) {
    var event = document.all ? window.event : e;
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
                    document.getElementById("prev_btn").click();
                } else {
                    document.getElementById("next_btn").click();
                }
            }
    }
}
document.addEventListener('keydown', shortcuts, false);
</script>
"""


def get_directory_contents(results_dir: Path) -> list[str]:
    """Get list of experiment directories with summary info."""
    if not results_dir or not results_dir.exists():
        return ["Select experiment directory"]

    exp_descriptions = []
    for dir_path in results_dir.iterdir():
        if not dir_path.is_dir():
            continue

        traj_dir = dir_path / "trajectories"
        if not traj_dir.exists():
            continue

        exp_desc = dir_path.name
        n_trajs = len(list(traj_dir.glob("*.jsonl")))
        exp_desc += f" ({n_trajs} trajectories)"
        exp_descriptions.append(exp_desc)

    return ["Select experiment directory"] + sorted(exp_descriptions, reverse=True)


def get_screenshot_from_step(step: EnvironmentOutput | AgentOutput | None) -> Image.Image | None:
    """Extract screenshot image from a step if available."""
    if not step:
        return None

    if isinstance(step, EnvironmentOutput):
        for content in step.obs.contents:
            if isinstance(content.data, Image.Image):
                return content.data
    return None


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
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


def _build_progress_html(n_completed: int, n_total: int, n_running: int) -> str:
    pct = (n_completed / n_total * 100) if n_total > 0 else 0
    bar = (
        f'<div style="background:#e5e7eb;border-radius:6px;height:18px;overflow:hidden;margin-bottom:4px;">'
        f'<div style="background:linear-gradient(90deg,#22c55e,#16a34a);height:100%;width:{pct:.1f}%;'
        f'border-radius:6px;transition:width 0.5s;"></div></div>'
    )
    label = f"<div style='font-size:13px;color:#555;'>{n_completed}/{n_total} episodes completed"
    if n_running > 0:
        label += f", {n_running} running"
    label += "</div>"
    return bar + label


def run_viewer(results_dir: Path, debug: bool = False, port: int | None = None, share: bool = False) -> None:
    """Run the Gradio viewer application.

    Args:
        results_dir: Path to results directory containing experiments.
        debug: Enable debug mode with hot reloading.
        port: Server port number. If None, Gradio picks an available port.
        share: Enable Gradio share link for remote access.
    """
    if isinstance(results_dir, str):
        results_dir = Path(results_dir)

    state = ViewerState(results_dir=results_dir)

    # --- Handler functions as closures that capture `state` ---

    def refresh_exp_dir_choices(current_choice: str) -> gr.Dropdown:
        return gr.Dropdown(choices=get_directory_contents(state.results_dir), value=current_choice)

    def _compute_experiment_stats(
        finished_rewards: list[float],
        finished_steps: list[int],
        finished_durations: list[float],
        n_failed: int,
        token_stats: dict[str, int],
    ) -> str:
        n_finished = len(finished_rewards)
        n_total = n_finished + n_failed

        if n_total == 0:
            return ""

        stats_parts = [f"\U0001f4ca **{n_total}** trajectories"]

        if n_failed > 0:
            stats_parts.append(f"\u2502 \u2705 Finished: **{n_finished}** \u2502 \u274c Failed: **{n_failed}**")
        else:
            stats_parts.append(f"\u2502 \u2705 All Finished: **{n_finished}**")

        if n_finished > 0:
            avg_reward = sum(finished_rewards) / n_finished
            avg_steps = sum(finished_steps) / n_finished
            success_rate = sum(1 for r in finished_rewards if r > 0) / n_finished * 100

            stats_parts.append(f"\u2502 Avg Reward: **{avg_reward:.2f}**")
            stats_parts.append(f"\u2502 Success Rate: **{success_rate:.0f}%**")
            stats_parts.append(f"\u2502 Avg Steps: **{avg_steps:.1f}**")

            if finished_durations:
                avg_duration = sum(finished_durations) / len(finished_durations)
                stats_parts.append(f"\u2502 Avg Duration: **{format_duration(avg_duration)}**")

        stats = " ".join(stats_parts)

        total_prompt = token_stats.get("prompt", 0)
        total_completion = token_stats.get("completion", 0)
        total_cached = token_stats.get("cached", 0)
        total_cache_created = token_stats.get("cache_created", 0)
        total_cost = token_stats.get("cost", 0.0)

        if total_prompt > 0:
            token_parts = [f"\U0001f4ca prompt: **{total_prompt:,}**"]
            token_parts.append(f"completion: **{total_completion:,}**")
            token_parts.append(f"total: **{total_prompt + total_completion:,}**")
            if total_cached > 0:
                cache_pct = total_cached / total_prompt * 100
                token_parts.append(f"cached: **{total_cached:,}** ({cache_pct:.0f}%)")
            if total_cache_created > 0:
                token_parts.append(f"cache_created: **{total_cache_created:,}**")
            if total_cost > 0:
                token_parts.append(f"\U0001f4b0 **${total_cost:.4f}**")
            stats += "\n\n" + " \u2502 ".join(token_parts)

        return stats

    def _build_experiment_view() -> tuple[list[list[Any]], str, str]:
        """Build trajectory table data, experiment stats, and progress HTML from current state."""
        traj_data: list[list[Any]] = []
        finished_rewards: list[float] = []
        finished_steps: list[int] = []
        finished_durations: list[float] = []
        n_failed = 0
        n_running = 0

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        total_cache_creation_tokens = 0
        total_cost = 0.0

        sorted_trajectories = sorted(
            state.trajectories.values(),
            key=lambda t: (t.start_time is None, t.start_time or 0),
        )

        for traj in sorted_trajectories:
            n_steps = len(traj.steps)
            status = _trajectory_status(traj)
            status_emoji = _STATUS_EMOJIS[status]

            if traj.reward_info:
                final_reward = traj.reward_info.get("reward", 0.0)
                final_message = traj.reward_info.get("message", "")
            else:
                final_reward = 0.0
                final_message = ""
                for step in reversed(traj.steps):
                    if isinstance(step.output, EnvironmentOutput):
                        final_reward = step.output.reward
                        final_message = step.output.info.get("message", "")
                        break

            traj_tokens = 0
            traj_cost = 0.0
            for traj_step in traj.steps:
                if isinstance(traj_step.output, AgentOutput):
                    for llm_call in traj_step.output.llm_calls:
                        if llm_call.usage:
                            traj_tokens += llm_call.usage.prompt_tokens + llm_call.usage.completion_tokens
                            traj_cost += llm_call.usage.cost
                            total_prompt_tokens += llm_call.usage.prompt_tokens
                            total_completion_tokens += llm_call.usage.completion_tokens
                            total_cached_tokens += llm_call.usage.cached_tokens
                            total_cache_creation_tokens += llm_call.usage.cache_creation_tokens
                            total_cost += llm_call.usage.cost

            task_id = traj.metadata.get("task_id", "unknown")

            is_finished = traj.start_time is not None and traj.end_time is not None
            if is_finished:
                duration = traj.end_time - traj.start_time
                duration_str = format_duration(duration)
                finished_rewards.append(final_reward)
                finished_steps.append(n_steps)
                finished_durations.append(duration)
            else:
                duration_str = "-"
                if status == "running":
                    n_running += 1
                else:
                    n_failed += 1

            tokens_str = f"{traj_tokens:,}" if traj_tokens > 0 else "-"
            cost_str = f"${traj_cost:.4f}" if traj_cost > 0 else "-"

            traj_data.append(
                [
                    status_emoji,
                    traj.id,
                    task_id,
                    n_steps,
                    f"{final_reward:.2f}",
                    final_message,
                    duration_str,
                    tokens_str,
                    cost_str,
                ]
            )

        token_stats = {
            "prompt": total_prompt_tokens,
            "completion": total_completion_tokens,
            "cached": total_cached_tokens,
            "cache_created": total_cache_creation_tokens,
            "cost": total_cost,
        }
        exp_stats = _compute_experiment_stats(
            finished_rewards, finished_steps, finished_durations, n_failed, token_stats
        )

        n_completed = len(state._completed_ids)
        n_total = state._expected_total or len(state.trajectories)
        progress_html = _build_progress_html(n_completed, n_total, n_running)

        return traj_data, exp_stats, progress_html

    def on_select_experiment(exp_name: str) -> tuple[str, list[list[Any]] | None, str, TrajectoryId | None]:
        if exp_name == "Select experiment directory" or not exp_name:
            return "", None, "", None

        dir_name = exp_name.split(" (")[0]
        exp_dir = state.results_dir / dir_name

        result = state.load_experiment(exp_dir)
        if "error" in result:
            return "", None, "", None

        traj_data, exp_stats, progress_html = _build_experiment_view()

        first_traj = None
        if state.trajectories:
            first_id = list(state.trajectories.keys())[0]
            state.select_trajectory(first_id)
            first_traj = TrajectoryId(exp_dir=str(exp_dir), trajectory_name=first_id)

        return exp_stats, traj_data, progress_html, first_traj

    def on_select_trajectory(evt: gr.SelectData, traj_table: Any) -> StepId | None:
        if traj_table is None or len(traj_table) == 0:
            return None

        row = evt.index[0]
        traj_id = traj_table.iloc[row, 1]
        state.select_trajectory(traj_id)
        return StepId(trajectory_id=TrajectoryId(trajectory_name=traj_id), step=0)

    def new_trajectory(traj_id: TrajectoryId) -> StepId:
        if traj_id and traj_id.trajectory_name:
            state.select_trajectory(traj_id.trajectory_name)
        return StepId(trajectory_id=traj_id, step=0)

    def navigate_prev(step_id: StepId) -> StepId:
        if step_id and step_id.step is not None:
            step = max(0, step_id.step - 1)
            state.step = step
            return StepId(trajectory_id=step_id.trajectory_id, step=step)
        return step_id

    def navigate_next(step_id: StepId) -> StepId:
        if step_id and step_id.step is not None and state.current_trajectory:
            step = min(state.total_steps() - 1, step_id.step + 1)
            state.step = step
            return StepId(trajectory_id=step_id.trajectory_id, step=step)
        return step_id

    def update_screenshot() -> Image.Image | None:
        step = state.get_step_at(state.step)
        img = get_screenshot_from_step(step)

        if img is None and state.step > 0:
            prev_step = state.get_step_at(state.step - 1)
            img = get_screenshot_from_step(prev_step)

        return img

    def update_llm_tools() -> str:
        step = state.get_step_at(state.step)

        if isinstance(step, AgentOutput) and step.llm_calls:
            llm_call = step.llm_calls[0]
            if llm_call.prompt.tools:
                return json.dumps(llm_call.prompt.tools, indent=2)
        return "No tools in current step"

    def update_raw_json() -> str:
        step = state.get_step_at(state.step)
        if step:
            return step.model_dump_json(indent=2)
        return "No step selected"

    def update_llm_calls() -> str:
        step = state.get_step_at(state.step)

        if isinstance(step, AgentOutput) and step.llm_calls:
            calls_data = [call.model_dump() for call in step.llm_calls]
            return json.dumps(calls_data, indent=2, default=str)
        return "No LLM calls in current step"

    def get_compact_header_info() -> str:
        if not state.current_trajectory:
            return "No trajectory selected"

        task_id = state.current_trajectory.metadata.get("task_id", "unknown")

        final_reward = 0.0
        is_done = False
        for step in reversed(state.current_trajectory.steps):
            if isinstance(step.output, EnvironmentOutput):
                final_reward = step.output.reward
                is_done = step.output.done
                break

        reward_emoji = "\u2705" if final_reward > 0 and is_done else "\u274c" if is_done else "\u23f3"

        return f"**{task_id}** \u2502 {reward_emoji} Reward: {final_reward:.2f}"

    def get_step_counter() -> str:
        if not state.current_trajectory:
            return "Step 0/0"
        current_step = state.step + 1
        total = state.total_steps()
        return f"Step {current_step}/{total}"

    def get_step_details() -> str:
        step = state.get_step_at(state.step)
        traj_step = state.get_trajectory_step_at(state.step)

        if step is None:
            return "No step selected"

        duration_info = ""
        if traj_step and traj_step.start_time is not None and traj_step.end_time is not None:
            duration = traj_step.end_time - traj_step.start_time
            duration_info = f" \u2502 \u23f1\ufe0f {format_duration(duration)}"

        if isinstance(step, EnvironmentOutput):
            sections = []
            sections.append(f"## \U0001f30d Environment Output{duration_info}\n")

            if step.done:
                status = "\u2705 **Success**" if step.reward > 0 else "\u274c **Failed**"
                sections.append(f"**Status:** {status} \u2502 **Reward:** {step.reward:.2f}\n")
            else:
                sections.append(f"**Reward:** {step.reward:.2f} \u2502 **Done:** No\n")

            for content in step.obs.contents:
                if isinstance(content.data, str):
                    name = content.name or "Content"
                    data = content.data[:2000] + "..." if len(content.data) > 2000 else content.data
                    sections.append(f"### {name}\n```\n{data}\n```\n")
                elif isinstance(content.data, Image.Image):
                    sections.append(
                        f"**{content.name or 'Screenshot'}:** {content.data.size[0]}x{content.data.size[1]}\n"
                    )
                elif isinstance(content.data, (dict, list)):
                    name = content.name or "Data"
                    data_str = json.dumps(content.data, indent=2)
                    data_str = data_str[:1000] + "..." if len(data_str) > 1000 else data_str
                    sections.append(f"### {name}\n```json\n{data_str}\n```\n")

            if step.info.get("error"):
                sections.append(f"\n### \u26a0\ufe0f Error\n```\n{step.info['error']}\n```\n")

            return "\n".join(sections)

        elif isinstance(step, AgentOutput):
            sections = []
            sections.append(f"## \U0001f916 Agent Output{duration_info}\n")

            if step.llm_calls:
                llm_call = step.llm_calls[0]
                usage = llm_call.usage
                if usage and usage.prompt_tokens > 0:
                    token_parts = [f"\U0001f4ca **Tokens:** prompt: {usage.prompt_tokens:,}"]
                    token_parts.append(f"completion: {usage.completion_tokens:,}")
                    if usage.cached_tokens > 0:
                        cache_pct = (usage.cached_tokens / usage.prompt_tokens * 100) if usage.prompt_tokens > 0 else 0
                        token_parts.append(f"cached: {usage.cached_tokens:,} ({cache_pct:.0f}%)")
                    if usage.cache_creation_tokens > 0:
                        token_parts.append(f"cache_created: {usage.cache_creation_tokens:,}")
                    if usage.cost > 0:
                        token_parts.append(f"\U0001f4b0 **${usage.cost:.4f}**")
                    sections.append(" \u2502 ".join(token_parts) + "\n")

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
                    if hasattr(msg, "content") and msg.content:
                        reasoning = msg.content[:1500] + "..." if len(msg.content) > 1500 else msg.content
                        sections.append(f"### Agent Reasoning\n{reasoning}\n")

            return "\n".join(sections)

        return "Unknown step type"

    def get_prev_screenshot() -> Image.Image | None:
        for i in range(state.step - 1, -1, -1):
            step = state.get_step_at(i)
            if isinstance(step, EnvironmentOutput):
                return get_screenshot_from_step(step)
        return None

    def update_trajectory_stats() -> str:
        if not state.current_trajectory:
            return ""

        traj = state.current_trajectory
        n_env = len(state.get_env_steps())
        n_agent = len(state.get_agent_steps())

        total_actions = sum(len(step.actions) for _, step in state.get_agent_steps())

        total_llm_calls = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        total_cache_creation_tokens = 0
        total_cost = 0.0

        for _, step in state.get_agent_steps():
            total_llm_calls += len(step.llm_calls)
            for llm_call in step.llm_calls:
                if llm_call.usage:
                    total_prompt_tokens += llm_call.usage.prompt_tokens
                    total_completion_tokens += llm_call.usage.completion_tokens
                    total_cached_tokens += llm_call.usage.cached_tokens
                    total_cache_creation_tokens += llm_call.usage.cache_creation_tokens
                    total_cost += llm_call.usage.cost

        stats = (
            f"\U0001f30d Env Steps: **{n_env}** \u2502 \U0001f916 Agent Steps: **{n_agent}** "
            f"\u2502 \u26a1 Actions: **{total_actions}** \u2502 \U0001f4ac LLM Calls: **{total_llm_calls}**"
        )

        if traj.start_time is not None and traj.end_time is not None:
            duration = traj.end_time - traj.start_time
            stats += f" \u2502 \u23f1\ufe0f **{format_duration(duration)}**"

        if total_prompt_tokens > 0:
            token_stats = (
                f"\U0001f4ca prompt: **{total_prompt_tokens:,}** \u2502 "
                f"completion: **{total_completion_tokens:,}** \u2502 "
                f"total: **{total_prompt_tokens + total_completion_tokens:,}**"
            )
            if total_cached_tokens > 0:
                cache_pct = total_cached_tokens / total_prompt_tokens * 100
                token_stats += f" \u2502 cached: **{total_cached_tokens:,}** ({cache_pct:.0f}%)"
            if total_cache_creation_tokens > 0:
                token_stats += f" \u2502 cache_created: **{total_cache_creation_tokens:,}**"
            if total_cost > 0:
                token_stats += f" \u2502 \U0001f4b0 **${total_cost:.4f}**"
            stats += f"\n\n{token_stats}"

        return stats

    def generate_timeline_html() -> str:
        if not state.current_trajectory or not state.current_trajectory.steps:
            return "<div style='padding: 10px; color: #666;'>No trajectory loaded</div>"

        env_color = "#a1c9f4"
        agent_color = "#8de5a1"
        current_highlight = "#ffd700"

        durations = []
        for traj_step in state.current_trajectory.steps:
            if traj_step.start_time is not None and traj_step.end_time is not None:
                durations.append(traj_step.end_time - traj_step.start_time)
            else:
                durations.append(None)

        valid_durations = [d for d in durations if d is not None and d > 0]
        if valid_durations:
            max_duration = max(valid_durations)
            min_duration = min(valid_durations)
        else:
            max_duration = min_duration = 1.0

        min_width = 12
        max_width = 240

        steps_html = []
        for i, traj_step in enumerate(state.current_trajectory.steps):
            step = traj_step.output
            is_current = i == state.step
            is_env = isinstance(step, EnvironmentOutput)

            duration = durations[i]
            if duration is not None and max_duration > min_duration:
                normalized = (duration - min_duration) / (max_duration - min_duration)
                width = int(min_width + normalized * (max_width - min_width))
            else:
                width = min_width

            bg_color = env_color if is_env else agent_color
            border = f"3px solid {current_highlight}" if is_current else "1px solid #ccc"
            box_shadow = "0 0 8px rgba(255, 215, 0, 0.8)" if is_current else "none"

            done_border = ""
            if is_env and step.done:
                done_color = "#32cd32" if step.reward > 0 else "#dc3545"
                done_border = f"border-bottom: 4px solid {done_color};"

            step_num = i + 1
            tooltip = f"Step {step_num}: {'Environment' if is_env else 'Agent'}"
            if duration is not None:
                tooltip += f" ({format_duration(duration)})"
            if is_env and step.done:
                tooltip += f" - Done, reward: {step.reward:.2f}"

            onclick_handler = (
                f"const inp = document.querySelector('#timeline_click_input input'); if(inp) {{ "
                f"const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set; "
                f"nativeSetter.call(inp, {i}); "
                f"inp.dispatchEvent(new Event('input', {{ bubbles: true }})); "
                f"inp.dispatchEvent(new Event('change', {{ bubbles: true }})); }}"
            )

            step_html = f"""
            <div class="timeline-step" data-step="{i}" title="{tooltip}" onclick="{onclick_handler}" style="
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: {width}px;
                height: 36px;
                margin: 2px;
                background-color: {bg_color};
                border: {border};
                border-radius: 4px;
                cursor: pointer;
                font-size: 11px;
                font-weight: bold;
                color: #333;
                box-shadow: {box_shadow};
                {done_border}
                transition: transform 0.1s;
            " onmouseover="this.style.transform='scale(1.1)'" onmouseout="this.style.transform='scale(1)'">
                {step_num}
            </div>"""
            steps_html.append(step_html)

        legend_html = f"""
        <div style="display: flex; gap: 15px; margin-bottom: 8px; font-size: 12px; color: #666;">
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 16px; height: 16px; background: {env_color}; border-radius: 3px;"></div>
                <span>Environment</span>
            </div>
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 16px; height: 16px; background: {agent_color}; border-radius: 3px;"></div>
                <span>Agent</span>
            </div>
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 16px; height: 16px; border: 2px solid {current_highlight}; border-radius: 3px;"></div>
                <span>Current</span>
            </div>
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 16px; height: 16px; border-bottom: 3px solid #32cd32; background: #eee; border-radius: 3px;"></div>
                <span>Success</span>
            </div>
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 16px; height: 16px; border-bottom: 3px solid #dc3545; background: #eee; border-radius: 3px;"></div>
                <span>Failure</span>
            </div>
        </div>
        """

        timeline_html = f"""
        <div style="padding: 10px; background: #f8f9fa; border-radius: 8px;">
            {legend_html}
            <div id="timeline-container" style="
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                padding: 8px;
                background: white;
                border-radius: 6px;
                border: 1px solid #dee2e6;
                max-height: 120px;
                overflow-y: auto;
            ">
                {"".join(steps_html)}
            </div>
        </div>
        """

        return timeline_html

    def handle_timeline_click(clicked_step: int, traj_id: TrajectoryId) -> StepId:
        if clicked_step is not None and state.current_trajectory:
            clicked_step = int(clicked_step)
            clicked_step = max(0, min(clicked_step, state.total_steps() - 1))
            state.step = clicked_step
            return StepId(trajectory_id=traj_id, step=clicked_step)
        return StepId(trajectory_id=traj_id, step=0)

    def on_live_toggle(is_live: bool) -> tuple[gr.Timer, gr.Row]:
        return gr.Timer(value=_LIVE_REFRESH_INTERVAL_S, active=is_live), gr.Row(visible=is_live)

    def on_timer_tick() -> tuple[str, list[list[Any]] | None, str, gr.Timer]:
        state.refresh_experiment()

        if state.current_trajectory and state.current_trajectory.id in state.trajectories:
            updated = state.trajectories[state.current_trajectory.id]
            if len(updated.steps) != len(state.current_trajectory.steps):
                state.current_trajectory = updated

        traj_data, exp_stats, progress_html = _build_experiment_view()

        if state.is_experiment_complete():
            return exp_stats, traj_data, progress_html, gr.Timer(active=False)

        return exp_stats, traj_data, progress_html, gr.Timer()

    # --- Build Gradio UI ---

    with gr.Blocks(theme=gr.themes.Soft(), css=_CSS, head=_SHORTCUT_JS) as demo:  # type: ignore
        traj_id = gr.State(value=TrajectoryId())
        step_id = gr.State(value=StepId())

        with gr.Accordion("Help", open=False):
            gr.Markdown(
                """\
# AgentLab2 X-Ray 2

1. **Select your experiment directory** from the dropdown.
2. **Select a trajectory** from the table to view its steps.
3. **Navigate steps** using the Previous/Next buttons or Ctrl/Cmd + Arrow keys.
4. **Live Mode**: Toggle to auto-refresh during active experiments.
"""
            )

        with gr.Row():
            exp_dir_choice = gr.Dropdown(
                choices=get_directory_contents(results_dir),
                value="Select experiment directory",
                label="Experiment Directory",
                show_label=False,
                scale=6,
            )
            refresh_button = gr.Button("Refresh", scale=0, size="sm")

        with gr.Row():
            live_toggle = gr.Checkbox(label="Live Mode", value=True, scale=0, min_width=90)

        timer = gr.Timer(value=_LIVE_REFRESH_INTERVAL_S, active=True)

        with gr.Accordion("\U0001f4c2 Trajectories", open=True):
            live_progress_row = gr.Row(visible=True)
            with live_progress_row:
                progress_bar = gr.HTML("")
            experiment_stats = gr.Markdown("", elem_id="experiment_stats")
            trajectory_table = gr.DataFrame(
                headers=["Status", "Name", "Task ID", "Steps", "Reward", "Message", "Duration", "Tokens", "Cost"],
                max_height=300,
                show_label=False,
                interactive=False,
            )

        with gr.Row(variant="panel", elem_classes="compact-header"):
            with gr.Column(scale=1, min_width=200):
                header_info = gr.Markdown("**Select a trajectory**")
            with gr.Column(scale=3):
                stats_display = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=0, min_width=80):
                prev_btn = gr.Button("\u25c0 Prev", size="sm", elem_id="prev_btn", min_width=70)
            with gr.Column(scale=0, min_width=100):
                step_counter = gr.Markdown("Step 0/0")
            with gr.Column(scale=1):
                timeline_html = gr.HTML(label="Timeline")
            with gr.Column(scale=0, min_width=80):
                next_btn = gr.Button("Next \u25b6", size="sm", elem_id="next_btn", min_width=70)

        with gr.Row(visible=True, elem_id="timeline_click_input"):
            timeline_click_input = gr.Number(show_label=False, container=False)

        with gr.Row():
            with gr.Column(scale=1):
                screenshot = gr.Image(
                    label="Current Screenshot",
                    show_label=True,
                    interactive=False,
                    show_download_button=False,
                    height=500,
                )
                with gr.Accordion("\U0001f4f7 Previous Screenshot", open=False):
                    prev_screenshot = gr.Image(
                        show_label=False,
                        interactive=False,
                        show_download_button=False,
                        height=400,
                    )

            with gr.Column(scale=1):
                step_details = gr.Markdown(
                    value="Select a trajectory to view step details",
                    elem_classes="step-details",
                )

        with gr.Accordion("\U0001f527 Debug / Raw Data", open=False):
            with gr.Tabs():
                with gr.Tab("Raw JSON"):
                    raw_json = gr.Code(language="json", show_label=False)
                with gr.Tab("LLM Calls"):
                    llm_calls = gr.Code(language="json", show_label=False)
                with gr.Tab("LLM Tools"):
                    llm_tools = gr.Code(language="json", show_label=False)

        # Event handlers
        refresh_button.click(fn=refresh_exp_dir_choices, inputs=exp_dir_choice, outputs=exp_dir_choice)

        exp_dir_choice.change(
            fn=on_select_experiment,
            inputs=exp_dir_choice,
            outputs=[experiment_stats, trajectory_table, progress_bar, traj_id],
        )

        trajectory_table.select(
            fn=on_select_trajectory,
            inputs=trajectory_table,
            outputs=step_id,
        )

        traj_id.change(fn=new_trajectory, inputs=traj_id, outputs=step_id)

        timeline_click_input.change(
            fn=handle_timeline_click,
            inputs=[timeline_click_input, traj_id],
            outputs=step_id,
        )

        prev_btn.click(navigate_prev, inputs=[step_id], outputs=[step_id])
        next_btn.click(navigate_next, inputs=[step_id], outputs=[step_id])

        step_id.change(fn=get_compact_header_info, outputs=header_info)
        step_id.change(fn=get_step_counter, outputs=step_counter)
        step_id.change(fn=generate_timeline_html, outputs=timeline_html)
        step_id.change(fn=update_screenshot, outputs=screenshot)
        step_id.change(fn=get_prev_screenshot, outputs=prev_screenshot)
        step_id.change(fn=get_step_details, outputs=step_details)
        step_id.change(fn=update_trajectory_stats, outputs=stats_display)
        step_id.change(fn=update_raw_json, outputs=raw_json)
        step_id.change(fn=update_llm_calls, outputs=llm_calls)
        step_id.change(fn=update_llm_tools, outputs=llm_tools)

        live_toggle.change(
            fn=on_live_toggle,
            inputs=[live_toggle],
            outputs=[timer, live_progress_row],
        )

        timer.tick(
            fn=on_timer_tick,
            outputs=[experiment_stats, trajectory_table, progress_bar, timer],
        )

        demo.load(fn=refresh_exp_dir_choices, inputs=exp_dir_choice, outputs=exp_dir_choice)

    demo.queue()
    demo.launch(server_port=port, share=share, debug=debug)


def main() -> None:
    results_dir = expanduser("~/agentlab_results/al2")
    parser = argparse.ArgumentParser(description="AgentLab2 X-Ray 2 Viewer")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=results_dir,
        help="Path to results directory containing experiments",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with hot reloading on source changes",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port number (default: auto-select available port)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio share link for remote access",
    )
    args = parser.parse_args()

    run_viewer(Path(args.results_dir), debug=args.debug, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
