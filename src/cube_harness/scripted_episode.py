"""Scripted episode runner — execute a fixed action sequence against a task without an LLM.

Useful for verifying that a mechanical fix (e.g. a new tool action) works end-to-end
before burning LLM tokens on a full agent run.

Usage in a script:
    from cube_harness.scripted_episode import run_scripted_episode

    result = run_scripted_episode(
        task_config=my_task_config,
        tool_config=my_tool_config,
        actions=[
            {"name": "keyboard_type_into", "arguments": {"bid": "a182", "text": "System Administrator"}},
            {"name": "noop", "arguments": {}},
            {"name": "submit_form", "arguments": {}},
        ],
    )
    print(result)  # ScriptedResult(reward=1.0, done=True, steps=[...])

CLI:
    uv run -m cube_harness.scripted_episode  (see --help)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from cube.core import Action, EnvironmentOutput, Observation

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """Result of a single scripted step."""

    turn: int
    action_name: str
    action_args: dict[str, Any]
    result: str  # text returned by the tool
    reward: float
    done: bool
    message: str  # validation message from task info


@dataclass
class ScriptedResult:
    """Final result of a scripted episode."""

    reward: float
    done: bool
    message: str
    steps: list[StepRecord] = field(default_factory=list)

    def __str__(self) -> str:
        status = "SOLVED" if self.reward == 1.0 else "FAILED"
        lines = [f"[{status}] reward={self.reward} done={self.done} msg={self.message!r}", ""]
        for s in self.steps:
            args_str = ", ".join(f"{k}={v!r}" for k, v in s.action_args.items())
            result_short = s.result[:60]
            lines.append(f"  T{s.turn:02d}  {s.action_name}({args_str}) → {result_short}  | r={s.reward} | {s.message!r}")
        return "\n".join(lines)


def run_scripted_episode(
    task_config: Any,
    tool_config: Any,
    actions: list[dict[str, Any]],
    verbose: bool = True,
) -> ScriptedResult:
    """Run a fixed action sequence against a task and return per-step results.

    Args:
        task_config: A TaskConfig (new cube path) or similar task specification.
        tool_config: A ToolConfig to create the tool.
        actions: List of action dicts with "name" and "arguments" keys.
        verbose: Print each step to stdout.

    Returns:
        ScriptedResult with per-step records and final reward.
    """
    # Import here to avoid circular imports
    from cube_harness.legacy import EnvConfig

    tool = tool_config.make()
    task = task_config.make() if hasattr(task_config, "make") else task_config
    task.setup(tool)

    env_output: EnvironmentOutput = EnvironmentOutput(obs=tool.page_obs() if hasattr(tool, "page_obs") else Observation(), reward=0.0, done=False, info={})

    records: list[StepRecord] = []

    for turn, action_dict in enumerate(actions):
        name = action_dict["name"]
        args = action_dict.get("arguments", {})
        action = Action(id=f"scripted_{turn}", name=name, arguments=args)

        if verbose:
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            print(f"T{turn:02d}  {name}({args_str})")

        try:
            obs_or_error = tool._execute_action(action)
            result_str = "Success"
            if hasattr(obs_or_error, "contents"):
                # Extract result text from first tool_call_id content
                for c in obs_or_error.contents:
                    if hasattr(c, "tool_call_id") and c.tool_call_id:
                        result_str = str(c.data)[:100]
                        break
        except Exception as e:
            result_str = f"Failed: {e}"

        # Validate task
        try:
            reward, info = task.validate_task(obs_or_error if hasattr(obs_or_error, "contents") else Observation())
            done = info.get("done", reward == 1.0)
            message = info.get("message", "")
        except Exception as e:
            reward, done, message = 0.0, False, f"validate error: {e}"

        record = StepRecord(turn=turn, action_name=name, action_args=args, result=result_str, reward=reward, done=done, message=message)
        records.append(record)

        if verbose:
            print(f"       → {result_str[:60]}  | reward={reward}  | {message!r}")

        if done:
            break

    final = records[-1] if records else StepRecord(0, "none", {}, "", 0.0, False, "")
    result = ScriptedResult(reward=final.reward, done=final.done, message=final.message, steps=records)

    if verbose:
        print(f"\n{'SOLVED' if result.reward == 1.0 else 'FAILED'}  reward={result.reward}\n")

    try:
        tool.close()
    except Exception:
        pass

    return result


def _load_actions_from_json(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run a scripted action sequence against a WorkArena task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("actions_json", help="JSON file with list of {name, arguments} dicts")
    parser.add_argument("--task", default="workarena.servicenow.create-incident", help="Task ID")
    parser.add_argument("--seed", type=int, default=0, help="Task seed")
    parser.add_argument("--headless", action="store_true", default=False, help="Run headless")
    args = parser.parse_args()

    actions = _load_actions_from_json(args.actions_json)

    try:
        from cube_browser_playwright.playwright_session import PlaywrightSessionConfig
        from workarena_cube.benchmark import WorkArenaBenchmark

        from cube_harness.tools.browsergym import BrowsergymConfig
        from cube_harness.tools.toolbox import ToolboxConfig

        print(f"Task: {args.task}  seed={args.seed}  headless={args.headless}")
        print(f"Actions: {len(actions)}\n")

        tool_config = ToolboxConfig(
            tool_configs=[
                BrowsergymConfig(
                    browser=PlaywrightSessionConfig(headless=args.headless, timeout=30000),
                    use_screenshot=False,
                    use_axtree=True,
                    use_html=False,
                ),
            ]
        )
        benchmark = WorkArenaBenchmark(level="l1", n_seeds_l1=1, default_tool_config=tool_config)
        benchmark.setup()
        tasks = [t for t in benchmark.load_tasks() if t.id == args.task]
        if not tasks:
            print(f"Task {args.task!r} not found", file=sys.stderr)
            sys.exit(1)

        result = run_scripted_episode(task_config=tasks[0], tool_config=tool_config, actions=actions)
        print(result)
        benchmark.close()
    except ImportError as e:
        print(f"Import error: {e}\nRun from the cube-harness directory with `uv run`.", file=sys.stderr)
        sys.exit(1)
