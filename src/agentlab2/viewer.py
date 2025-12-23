"""
Experiment viewer for AgentLab2.

A Gradio-based UI for exploring experiment results, trajectories, and agent outputs.
Mimics the UI of the original agentlab viewer but works with the new data structures.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from posixpath import expanduser
from typing import Any

import gradio as gr
from PIL import Image

from agentlab2.core import AgentOutput, EnvironmentOutput, Trajectory, TrajectoryStep
from agentlab2.storage import FileStorage


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
    """Global state for the viewer application."""

    results_dir: Path
    exp_dirs: list[Path] = field(default_factory=list)
    trajectories: dict[str, Trajectory] = field(default_factory=dict)
    current_trajectory: Trajectory | None = None
    current_exp_dir: Path | None = None
    step: int = 0

    def load_experiment(self, exp_dir: Path) -> dict[str, Any]:
        """Load all trajectories from an experiment directory."""
        self.exp_dirs = [exp_dir]
        self.current_exp_dir = exp_dir
        self.trajectories = {}

        traj_dir = exp_dir / "trajectories"
        if not traj_dir.exists():
            return {"error": f"No trajectories directory found in {exp_dir}"}

        storage = FileStorage(exp_dir)
        trajectories = storage.load_all_trajectories()

        for traj in trajectories:
            self.trajectories[traj.id] = traj

        return {"loaded": len(self.trajectories)}

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


# Global state instance
state: ViewerState


# CSS styling
css = """
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
shortcut_js = """
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

    try:
        subdirs = list(results_dir.iterdir())
    except PermissionError:
        return ["Select experiment directory"]

    for dir_path in subdirs:
        if not dir_path.is_dir():
            continue

        traj_dir = dir_path / "trajectories"
        try:
            if not traj_dir.is_dir():
                continue
            n_trajs = sum(1 for _ in traj_dir.glob("*.jsonl"))
        except PermissionError:
            continue

        exp_descriptions.append(f"{dir_path.name} ({n_trajs} trajectories)")

    return ["Select experiment directory"] + sorted(exp_descriptions, reverse=True)


def refresh_exp_dir_choices(current_choice):
    """Refresh the experiment directory dropdown."""
    global state
    return gr.Dropdown(choices=get_directory_contents(state.results_dir), value=current_choice)


def on_select_experiment(exp_name: str):
    """Handle experiment selection."""
    global state
    if exp_name == "Select experiment directory" or not exp_name:
        return None, None

    # Extract directory name (remove trajectory count suffix)
    dir_name = exp_name.split(" (")[0]
    exp_dir = state.results_dir / dir_name

    result = state.load_experiment(exp_dir)
    if "error" in result:
        return None, None

    # Create trajectory table
    traj_data = []
    for traj in state.trajectories.values():
        n_steps = len(traj.steps)
        final_reward = 0.0
        for step in reversed(traj.steps):
            if isinstance(step.output, EnvironmentOutput):
                final_reward = step.output.reward
                break
        task_id = traj.metadata.get("task_id", "unknown")
        duration_str = "-"
        if traj.start_time is not None and traj.end_time is not None:
            duration_str = format_duration(traj.end_time - traj.start_time)
        traj_data.append([traj.id, task_id, n_steps, f"{final_reward:.2f}", duration_str])

    # Select first trajectory by default
    first_traj = None
    if state.trajectories:
        first_id = list(state.trajectories.keys())[0]
        state.select_trajectory(first_id)
        first_traj = TrajectoryId(exp_dir=str(exp_dir), trajectory_name=first_id)

    return traj_data, first_traj


def on_select_trajectory(evt: gr.SelectData, traj_table):
    """Handle trajectory selection from table."""
    global state
    if traj_table is None or len(traj_table) == 0:
        return None

    row = evt.index[0]
    traj_id = traj_table.iloc[row, 0]
    state.select_trajectory(traj_id)
    return StepId(trajectory_id=TrajectoryId(trajectory_name=traj_id), step=0)


def new_trajectory(traj_id: TrajectoryId):
    """Handle new trajectory selection."""
    global state
    if traj_id and traj_id.trajectory_name:
        state.select_trajectory(traj_id.trajectory_name)
    return StepId(trajectory_id=traj_id, step=0)


def navigate_prev(step_id: StepId):
    """Navigate to previous step."""
    global state
    if step_id and step_id.step is not None:
        step = max(0, step_id.step - 1)
        state.step = step
        return StepId(trajectory_id=step_id.trajectory_id, step=step)
    return step_id


def navigate_next(step_id: StepId):
    """Navigate to next step."""
    global state
    if step_id and step_id.step is not None and state.current_trajectory:
        step = min(state.total_steps() - 1, step_id.step + 1)
        state.step = step
        return StepId(trajectory_id=step_id.trajectory_id, step=step)
    return step_id


def get_screenshot_from_step(step: EnvironmentOutput | AgentOutput | None) -> Image.Image | None:
    """Extract screenshot image from a step if available."""
    if not step:
        return None

    if isinstance(step, EnvironmentOutput):
        for content in step.obs.contents:
            if isinstance(content.data, Image.Image):
                return content.data
    return None


def update_screenshot():
    """Update screenshot display for current step."""
    global state
    step = state.get_step_at(state.step)
    img = get_screenshot_from_step(step)

    # If current step is AgentOutput, try to get screenshot from previous env step
    if img is None and state.step > 0:
        prev_step = state.get_step_at(state.step - 1)
        img = get_screenshot_from_step(prev_step)

    return img


def update_llm_tools():
    """Get tools configuration from LLM calls."""
    global state
    step = state.get_step_at(state.step)

    if isinstance(step, AgentOutput) and step.llm_calls:
        llm_call = step.llm_calls[0]
        if llm_call.prompt.tools:
            return json.dumps(llm_call.prompt.tools, indent=2)
    return "No tools in current step"


def update_raw_json():
    """Get raw JSON of current step."""
    global state
    step = state.get_step_at(state.step)
    if step:
        return step.model_dump_json(indent=2)
    return "No step selected"


def update_logs():
    global state
    if not state.current_trajectory or not state.current_exp_dir:
        return "No trajectory selected"

    storage = FileStorage(state.current_exp_dir)
    trajectory_id = state.current_trajectory.id

    if not storage.has_logs(trajectory_id):
        return f"No logs found for trajectory: {trajectory_id}\n\nLogs are stored in: {storage.get_log_path(trajectory_id)}"

    logs = storage.load_logs(trajectory_id)
    return logs if logs else "Log file is empty"


def update_llm_calls():
    """Get LLM calls from current step."""
    global state
    step = state.get_step_at(state.step)

    if isinstance(step, AgentOutput) and step.llm_calls:
        # Serialize all LLM calls
        calls_data = [call.model_dump() for call in step.llm_calls]
        return json.dumps(calls_data, indent=2, default=str)
    return "No LLM calls in current step"


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


def get_compact_header_info():
    """Get compact header info string."""
    global state
    if not state.current_trajectory:
        return "No trajectory selected"

    task_id = state.current_trajectory.metadata.get("task_id", "unknown")

    # Calculate final reward
    final_reward = 0.0
    is_done = False
    for step in reversed(state.current_trajectory.steps):
        if isinstance(step.output, EnvironmentOutput):
            final_reward = step.output.reward
            is_done = step.output.done
            break

    reward_emoji = "✅" if final_reward > 0 and is_done else "❌" if is_done else "⏳"

    return f"**{task_id}** │ {reward_emoji} Reward: {final_reward:.2f}"


def get_step_counter():
    """Get current step counter string."""
    global state
    if not state.current_trajectory:
        return "Step 0/0"
    current_step = state.step + 1
    total = state.total_steps()
    return f"Step {current_step}/{total}"


def get_step_details():
    """Get context-aware step details based on step type."""
    global state
    step = state.get_step_at(state.step)
    traj_step = state.get_trajectory_step_at(state.step)

    if step is None:
        return "No step selected"

    # Get step duration if available
    duration_info = ""
    if traj_step and traj_step.start_time is not None and traj_step.end_time is not None:
        duration = traj_step.end_time - traj_step.start_time
        duration_info = f" │ ⏱️ {format_duration(duration)}"

    if isinstance(step, EnvironmentOutput):
        # Environment step: show observation details
        sections = []
        sections.append(f"## 🌍 Environment Output{duration_info}\n")

        # Status
        if step.done:
            status = "✅ **Success**" if step.reward > 0 else "❌ **Failed**"
            sections.append(f"**Status:** {status} │ **Reward:** {step.reward:.2f}\n")
        else:
            sections.append(f"**Reward:** {step.reward:.2f} │ **Done:** No\n")

        # Observation contents
        for content in step.obs.contents:
            if isinstance(content.data, str):
                name = content.name or "Content"
                # Truncate long content
                data = content.data[:2000] + "..." if len(content.data) > 2000 else content.data
                sections.append(f"### {name}\n```\n{data}\n```\n")
            elif isinstance(content.data, Image.Image):
                sections.append(f"**{content.name or 'Screenshot'}:** {content.data.size[0]}x{content.data.size[1]}\n")
            elif isinstance(content.data, (dict, list)):
                name = content.name or "Data"
                data_str = json.dumps(content.data, indent=2)
                data_str = data_str[:1000] + "..." if len(data_str) > 1000 else data_str
                sections.append(f"### {name}\n```json\n{data_str}\n```\n")

        # Error from info
        if step.info.get("error"):
            sections.append(f"\n### ⚠️ Error\n```\n{step.info['error']}\n```\n")

        return "\n".join(sections)

    elif isinstance(step, AgentOutput):
        # Agent step: show actions and reasoning
        sections = []
        sections.append(f"## 🤖 Agent Output{duration_info}\n")

        # Actions
        if step.actions:
            sections.append("### Actions\n")
            for i, action in enumerate(step.actions):
                args_str = json.dumps(action.arguments, indent=2)
                sections.append(f"**{i + 1}. {action.name}**\n```json\n{args_str}\n```\n")
        else:
            sections.append("*No actions taken*\n")

        # LLM reasoning (if available)
        if step.llm_calls:
            llm_call = step.llm_calls[0]
            if llm_call.output:
                msg = llm_call.output
                if hasattr(msg, "content") and msg.content:
                    reasoning = msg.content[:1500] + "..." if len(msg.content) > 1500 else msg.content
                    sections.append(f"### Agent Reasoning\n{reasoning}\n")

        return "\n".join(sections)

    return "Unknown step type"


def get_prev_screenshot():
    """Get the previous environment screenshot."""
    global state
    # Find previous EnvironmentOutput with screenshot
    for i in range(state.step - 1, -1, -1):
        step = state.get_step_at(i)
        if isinstance(step, EnvironmentOutput):
            return get_screenshot_from_step(step)
    return None


def update_trajectory_stats():
    """Update trajectory statistics as compact markdown."""
    global state
    if not state.current_trajectory:
        return ""

    traj = state.current_trajectory
    n_env = len(state.get_env_steps())
    n_agent = len(state.get_agent_steps())

    # Count total actions
    total_actions = sum(len(step.actions) for _, step in state.get_agent_steps())

    # Count total LLM calls
    total_llm_calls = sum(len(step.llm_calls) for _, step in state.get_agent_steps())

    stats = f"🌍 Env Steps: **{n_env}** │ 🤖 Agent Steps: **{n_agent}** │ ⚡ Actions: **{total_actions}** │ 💬 LLM Calls: **{total_llm_calls}**"

    # Add timing info
    if traj.start_time is not None:
        start_dt = datetime.fromtimestamp(traj.start_time)
        start_str = start_dt.strftime("%H:%M:%S")
        stats += f" │ 🕐 Start: **{start_str}**"

        if traj.end_time is not None:
            duration = traj.end_time - traj.start_time
            stats += f" │ ⏱️ Duration: **{format_duration(duration)}**"
    return stats


def generate_timeline_html():
    """Generate an HTML timeline visualization of the trajectory."""
    global state

    if not state.current_trajectory or not state.current_trajectory.steps:
        return "<div style='padding: 10px; color: #666;'>No trajectory loaded</div>"

    # Colors matching the original
    env_color = "#a1c9f4"  # Light blue for env
    agent_color = "#8de5a1"  # Light green for agent
    current_highlight = "#ffd700"  # Gold for current step

    # Calculate durations for all steps
    durations = []
    for traj_step in state.current_trajectory.steps:
        if traj_step.start_time is not None and traj_step.end_time is not None:
            durations.append(traj_step.end_time - traj_step.start_time)
        else:
            durations.append(None)

    # Calculate width scaling based on durations
    valid_durations = [d for d in durations if d is not None and d > 0]
    if valid_durations:
        max_duration = max(valid_durations)
        min_duration = min(valid_durations)
    else:
        max_duration = min_duration = 1.0

    # Width range
    min_width = 12
    max_width = 240

    steps_html = []
    for i, traj_step in enumerate(state.current_trajectory.steps):
        step = traj_step.output
        is_current = i == state.step
        is_env = isinstance(step, EnvironmentOutput)

        # Calculate width based on duration
        duration = durations[i]
        if duration is not None and max_duration > min_duration:
            # Normalize to 0-1 range, then scale to width range
            normalized = (duration - min_duration) / (max_duration - min_duration)
            width = int(min_width + normalized * (max_width - min_width))
        else:
            width = min_width

        # Base styling
        bg_color = env_color if is_env else agent_color
        border = f"3px solid {current_highlight}" if is_current else "1px solid #ccc"
        box_shadow = "0 0 8px rgba(255, 215, 0, 0.8)" if is_current else "none"

        # Done state border
        done_border = ""
        if is_env and step.done:
            done_color = "#32cd32" if step.reward > 0 else "#dc3545"
            done_border = f"border-bottom: 4px solid {done_color};"

        step_num = i + 1  # 1-based display
        tooltip = f"Step {step_num}: {'Environment' if is_env else 'Agent'}"
        if duration is not None:
            tooltip += f" ({format_duration(duration)})"
        if is_env and step.done:
            tooltip += f" - Done, reward: {step.reward:.2f}"

        # onclick handler uses 0-based index internally
        # Use native setter to properly trigger framework change detection, then dispatch events
        onclick_handler = f"const inp = document.querySelector('#timeline_click_input input'); if(inp) {{ const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set; nativeSetter.call(inp, {i}); inp.dispatchEvent(new Event('input', {{ bubbles: true }})); inp.dispatchEvent(new Event('change', {{ bubbles: true }})); }}"

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

    # Legend
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


def handle_timeline_click(clicked_step: int, traj_id: TrajectoryId):
    """Handle click on timeline step."""
    global state
    if clicked_step is not None and state.current_trajectory:
        clicked_step = int(clicked_step)
        clicked_step = max(0, min(clicked_step, state.total_steps() - 1))
        state.step = clicked_step
        return StepId(trajectory_id=traj_id, step=clicked_step)
    return StepId(trajectory_id=traj_id, step=0)


def run_viewer(results_dir: Path, debug: bool = False, port: int | None = None, share: bool = False):
    """Run the Gradio viewer application.

    Args:
        results_dir: Path to results directory containing experiments.
        debug: Enable debug mode with hot reloading.
        port: Server port number. If None, Gradio picks an available port.
        share: Enable Gradio share link for remote access.
    """
    global state

    if isinstance(results_dir, str):
        results_dir = Path(results_dir)

    state = ViewerState(results_dir=results_dir)

    with gr.Blocks(theme=gr.themes.Soft(), css=css, head=shortcut_js) as demo:  # type: ignore
        traj_id = gr.State(value=TrajectoryId())
        step_id = gr.State(value=StepId())

        # Help section
        with gr.Accordion("Help", open=False):
            gr.Markdown(
                """\
# AgentLab2 Experiment Viewer

1. **Select your experiment directory** from the dropdown.
2. **Select a trajectory** from the table to view its steps.
3. **Navigate steps** using the Previous/Next buttons or Ctrl/Cmd + Arrow keys.
4. **View different data** by selecting tabs below.
"""
            )

        # Experiment selection
        with gr.Row():
            exp_dir_choice = gr.Dropdown(
                choices=get_directory_contents(results_dir),
                value="Select experiment directory",
                label="Experiment Directory",
                show_label=False,
                scale=6,
            )
            refresh_button = gr.Button("Refresh", scale=0, size="sm")

        # Trajectory selection (collapsible after selection)
        with gr.Accordion("📂 Trajectories", open=True):
            trajectory_table = gr.DataFrame(
                headers=["Name", "Task ID", "Steps", "Reward", "Duration"],
                max_height=300,
                show_label=False,
                interactive=False,
            )

        # Compact header bar with episode info + stats
        with gr.Row(variant="panel", elem_classes="compact-header"):
            with gr.Column(scale=1, min_width=200):
                header_info = gr.Markdown("**Select a trajectory**")
            with gr.Column(scale=3):
                stats_display = gr.Markdown("")

        # Timeline with navigation
        with gr.Row():
            with gr.Column(scale=0, min_width=80):
                prev_btn = gr.Button("◀ Prev", size="sm", elem_id="prev_btn", min_width=70)
            with gr.Column(scale=0, min_width=100):
                step_counter = gr.Markdown("Step 0/0")
            with gr.Column(scale=1):
                timeline_html = gr.HTML(label="Timeline")
            with gr.Column(scale=0, min_width=80):
                next_btn = gr.Button("Next ▶", size="sm", elem_id="next_btn", min_width=70)

        # Hidden input for timeline click handling (visible but height 0 so it renders in DOM)
        with gr.Row(visible=True, elem_id="timeline_click_input"):
            timeline_click_input = gr.Number(show_label=False, container=False)

        # Main two-column view
        with gr.Row():
            # Left column: Screenshots
            with gr.Column(scale=1):
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

            # Right column: Step details
            with gr.Column(scale=1):
                step_details = gr.Markdown(
                    value="Select a trajectory to view step details",
                    elem_classes="step-details",
                )

        # Debug section (collapsed)
        with gr.Accordion("🔧 Debug / Raw Data", open=False):
            with gr.Tabs():
                with gr.Tab("Raw JSON"):
                    raw_json = gr.Code(language="json", show_label=False)
                with gr.Tab("LLM Calls"):
                    llm_calls = gr.Code(language="json", show_label=False)
                with gr.Tab("LLM Tools"):
                    llm_tools = gr.Code(language="json", show_label=False)
                with gr.Tab("Episode Logs"):
                    episode_logs = gr.Textbox(
                        show_label=False,
                        lines=25,
                        max_lines=50,
                        interactive=False,
                    )

        # Event handlers
        refresh_button.click(fn=refresh_exp_dir_choices, inputs=exp_dir_choice, outputs=exp_dir_choice)

        exp_dir_choice.change(
            fn=on_select_experiment,
            inputs=exp_dir_choice,
            outputs=[trajectory_table, traj_id],
        )

        trajectory_table.select(
            fn=on_select_trajectory,
            inputs=trajectory_table,
            outputs=step_id,
        )

        traj_id.change(fn=new_trajectory, inputs=traj_id, outputs=step_id)
        traj_id.change(fn=update_logs, outputs=episode_logs)

        # Timeline click handler
        timeline_click_input.change(
            fn=handle_timeline_click,
            inputs=[timeline_click_input, traj_id],
            outputs=step_id,
        )

        # Navigation
        prev_btn.click(navigate_prev, inputs=[step_id], outputs=[step_id])
        next_btn.click(navigate_next, inputs=[step_id], outputs=[step_id])

        # Step change updates - new simplified handlers
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

        # Initial load
        demo.load(fn=refresh_exp_dir_choices, inputs=exp_dir_choice, outputs=exp_dir_choice)

    demo.queue()
    demo.launch(server_port=port, share=share, debug=debug)


def main():
    """Main entry point for the viewer."""
    import argparse

    results_dir = expanduser("~/agentlab_results/al2")
    parser = argparse.ArgumentParser(description="AgentLab2 Experiment Viewer")
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
