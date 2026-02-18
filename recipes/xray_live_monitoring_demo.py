import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

from litellm import Message
from PIL import Image, ImageDraw

from agentlab2.core import (
    Action,
    AgentOutput,
    Content,
    EnvironmentOutput,
    Observation,
    StepError,
    Trajectory,
    TrajectoryStep,
)
from agentlab2.llm import LLMCall, LLMConfig, Prompt, Usage
from agentlab2.storage import FileStorage
from agentlab2.viewer import run_viewer

DEFAULT_RESULTS_DIR = Path.home() / "agentlab_results" / "al2"


def _make_screenshot(label: str, color: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", (980, 520), color=color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 25, 955, 495), outline=(255, 255, 255), width=3)
    draw.text((40, 40), "X-Ray Live Demo", fill=(255, 255, 255))
    draw.text((40, 80), label, fill=(255, 255, 255))
    draw.text((40, 120), time.strftime("%H:%M:%S"), fill=(225, 225, 225))
    return image


def _make_observation(status: str, detail: str, color: tuple[int, int, int]) -> Observation:
    ui_state = json.dumps({"status": status, "detail": detail}, indent=2)
    return Observation(
        contents=[
            Content(name="status", data=status),
            Content(name="detail", data=detail),
            Content(name="ui_state", data=ui_state),
            Content(name="screenshot", data=_make_screenshot(f"{status} | {detail}", color)),
        ]
    )


def _tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Click an element on the page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector"},
                },
                "required": ["selector"],
            },
        },
    }


def _make_llm_call(
    reasoning: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    cache_creation_tokens: int,
    cost: float,
) -> LLMCall:
    return LLMCall(
        llm_config=LLMConfig(model_name="demo/mock-model", temperature=0.2),
        prompt=Prompt(
            messages=[
                {"role": "system", "content": "You are an autonomous web agent."},
                {"role": "user", "content": "Read state and choose the next action."},
            ],
            tools=[_tool_schema()],
        ),
        output=Message(role="assistant", content=reasoning),
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cost=cost,
        ),
    )


def _env_step(
    status: str,
    detail: str,
    reward: float,
    done: bool,
    start_time: float,
    end_time: float,
    color: tuple[int, int, int],
    message: str,
) -> TrajectoryStep:
    return TrajectoryStep(
        output=EnvironmentOutput(
            obs=_make_observation(status=status, detail=detail, color=color),
            reward=reward,
            done=done,
            info={"message": message},
        ),
        start_time=start_time,
        end_time=end_time,
    )


def _agent_step(
    action_name: str,
    arguments: dict[str, Any],
    reasoning: str,
    start_time: float,
    end_time: float,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cost: float = 0.0,
    error: StepError | None = None,
) -> TrajectoryStep:
    return TrajectoryStep(
        output=AgentOutput(
            actions=[Action(id=f"call_{int(start_time * 1000)}", name=action_name, arguments=arguments)],
            llm_calls=[
                _make_llm_call(
                    reasoning=reasoning,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                    cost=cost,
                )
            ],
            error=error,
        ),
        start_time=start_time,
        end_time=end_time,
    )


def _save_trajectory(storage: FileStorage, trajectory: Trajectory) -> None:
    storage.save_trajectory(trajectory)


def _write_episode_configs(exp_dir: Path, total: int) -> None:
    config_dir = exp_dir / "episode_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(total):
        path = config_dir / f"episode_{idx}_task_demo_{idx}.json"
        payload = {"id": idx, "task_id": f"demo_{idx}", "note": "demo config for progress bar"}
        path.write_text(json.dumps(payload, indent=2))


def _update_trajectory_metadata(storage: FileStorage, trajectory: Trajectory) -> None:
    metadata_path = storage.output_dir / "trajectories" / f"{trajectory.id}.metadata.json"
    metadata_path.write_text(json.dumps(trajectory.model_dump(exclude={"steps"}), indent=2))


def _seed_demo_trajectories(storage: FileStorage) -> Trajectory:
    now = time.time() - 120

    success = Trajectory(
        id="traj_success",
        metadata={"task_id": "demo_success"},
        start_time=now,
        end_time=now + 6.2,
        reward_info={"reward": 1.0, "done": True, "message": "Solved"},
        steps=[
            _env_step("success", "loaded", 0.0, False, now, now + 0.9, (24, 108, 71), "Page loaded"),
            _agent_step(
                action_name="browser_click",
                arguments={"selector": "#submit"},
                reasoning="Click submit to finish.",
                start_time=now + 1.2,
                end_time=now + 2.0,
                prompt_tokens=720,
                completion_tokens=98,
                cached_tokens=140,
                cost=0.0032,
            ),
            _env_step(
                "success",
                "validated",
                1.0,
                True,
                now + 2.3,
                now + 3.1,
                (31, 148, 93),
                "Validation successful",
            ),
        ],
    )
    _save_trajectory(storage, success)

    completed = Trajectory(
        id="traj_completed",
        metadata={"task_id": "demo_completed"},
        start_time=now + 10,
        end_time=now + 17.3,
        reward_info={"reward": 0.0, "done": True, "message": "Completed with no reward"},
        steps=[
            _env_step(
                "completed",
                "loaded",
                0.0,
                False,
                now + 10.0,
                now + 10.7,
                (65, 98, 120),
                "Page loaded",
            ),
            _agent_step(
                action_name="browser_click",
                arguments={"selector": "#skip"},
                reasoning="No high-value action remains, finalize.",
                start_time=now + 11.0,
                end_time=now + 11.8,
                prompt_tokens=540,
                completion_tokens=64,
                cost=0.0021,
            ),
            _env_step(
                "completed",
                "done",
                0.0,
                True,
                now + 12.1,
                now + 12.8,
                (72, 110, 136),
                "Done but no reward",
            ),
        ],
    )
    _save_trajectory(storage, completed)

    error = Trajectory(
        id="traj_error",
        metadata={"task_id": "demo_error"},
        start_time=now + 20,
        end_time=now + 24.5,
        reward_info={"reward": 0.0, "done": True, "message": "Execution failed"},
        steps=[
            _env_step("error", "loaded", 0.0, False, now + 20.0, now + 20.6, (145, 52, 52), "Page loaded"),
            _agent_step(
                action_name="browser_type",
                arguments={"selector": "#amount", "text": "abc"},
                reasoning="Attempt to enter amount.",
                start_time=now + 21.0,
                end_time=now + 21.7,
                prompt_tokens=680,
                completion_tokens=92,
                cached_tokens=120,
                cache_creation_tokens=40,
                cost=0.0036,
                error=StepError(
                    error_type="ValueError",
                    exception_str="Failed to parse numeric amount",
                    stack_trace="Traceback (most recent call last): ...",
                ),
            ),
            _env_step(
                "error",
                "failed",
                0.0,
                True,
                now + 22.0,
                now + 22.8,
                (167, 68, 68),
                "Validation failed",
            ),
        ],
    )
    _save_trajectory(storage, error)

    stuck_running = Trajectory(
        id="traj_stuck_running",
        metadata={"task_id": "demo_stuck"},
        start_time=now + 30,
        end_time=None,
        reward_info={},
        steps=[
            _env_step(
                "running",
                "waiting for external event",
                0.0,
                False,
                now + 30.0,
                now + 30.6,
                (122, 90, 29),
                "Pending dependency",
            ),
            _agent_step(
                action_name="browser_wait",
                arguments={"seconds": 2},
                reasoning="Polling for state change.",
                start_time=now + 31.0,
                end_time=now + 31.4,
                prompt_tokens=410,
                completion_tokens=58,
                cost=0.0013,
            ),
            _env_step(
                "running",
                "still pending",
                0.0,
                False,
                now + 31.9,
                now + 32.5,
                (133, 99, 33),
                "No state change yet",
            ),
        ],
    )
    _save_trajectory(storage, stuck_running)

    live = Trajectory(
        id="traj_live_stream",
        metadata={"task_id": "demo_live"},
        start_time=now + 40,
        end_time=None,
        reward_info={},
        steps=[
            _env_step(
                "running",
                "stream start",
                0.0,
                False,
                now + 40.0,
                now + 40.8,
                (24, 82, 120),
                "Live trajectory started",
            ),
            _agent_step(
                action_name="browser_click",
                arguments={"selector": "#begin"},
                reasoning="Kick off the streaming run.",
                start_time=now + 41.1,
                end_time=now + 41.8,
                prompt_tokens=620,
                completion_tokens=84,
                cost=0.0029,
            ),
        ],
    )
    _save_trajectory(storage, live)
    return live


def _stream_live_updates(storage: FileStorage, trajectory: Trajectory, interval_s: float) -> None:
    next_step_num = len(trajectory.steps)

    for idx in range(2):
        start_time = time.time()
        env = _env_step(
            status="running",
            detail=f"live pass {idx + 1}",
            reward=0.0,
            done=False,
            start_time=start_time,
            end_time=start_time + 0.4,
            color=(28, 94, 138),
            message=f"Live update {idx + 1}",
        )
        trajectory.steps.append(env)
        storage.save_step(env, trajectory.id, next_step_num)
        next_step_num += 1
        time.sleep(interval_s)

        agent_start = time.time()
        agent = _agent_step(
            action_name="browser_click",
            arguments={"selector": f"#step-{idx + 1}"},
            reasoning=f"Advance live trajectory at pass {idx + 1}.",
            start_time=agent_start,
            end_time=agent_start + 0.5,
            prompt_tokens=500 + idx * 40,
            completion_tokens=70 + idx * 10,
            cached_tokens=80,
            cost=0.0024 + idx * 0.0003,
        )
        trajectory.steps.append(agent)
        storage.save_step(agent, trajectory.id, next_step_num)
        next_step_num += 1
        time.sleep(interval_s)

    final_start = time.time()
    final_env = _env_step(
        status="success",
        detail="live completed",
        reward=1.0,
        done=True,
        start_time=final_start,
        end_time=final_start + 0.5,
        color=(31, 148, 93),
        message="Live trajectory finished successfully",
    )
    trajectory.steps.append(final_env)
    storage.save_step(final_env, trajectory.id, next_step_num)

    trajectory.end_time = final_env.end_time
    trajectory.reward_info = {"reward": 1.0, "done": True, "message": "Live run completed"}
    _update_trajectory_metadata(storage, trajectory)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and visualize a live X-Ray experiment demo.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root directory containing experiment folders (default: ~/agentlab_results/al2).",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=f"xray_live_demo_{time.strftime('%Y%m%d_%H%M%S')}",
        help="Experiment directory name to create under --results-dir.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Viewer port when launching the UI.",
    )
    parser.add_argument(
        "--live-interval",
        type=float,
        default=3.0,
        help="Seconds between live updates for traj_live_stream.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Generate demo data and live updates without launching the viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir.expanduser()
    exp_dir = results_dir / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=False)

    storage = FileStorage(exp_dir)
    _write_episode_configs(exp_dir, total=6)
    live_traj = _seed_demo_trajectories(storage)

    updater = threading.Thread(
        target=_stream_live_updates,
        args=(storage, live_traj, float(args.live_interval)),
        daemon=not args.no_viewer,
    )
    updater.start()

    print(f"Demo experiment created: {exp_dir}")
    print("Trajectory states included: stuck running, success, error, completed, live-streaming updates.")
    print(f"Run viewer manually with: uv run python -m agentlab2.viewer --results-dir {results_dir}")
    print(f"In the dropdown select: {args.exp_name}")

    if args.no_viewer:
        updater.join()
        print("Live updates finished.")
        return

    run_viewer(results_dir=results_dir, port=int(args.port))


if __name__ == "__main__":
    main()
