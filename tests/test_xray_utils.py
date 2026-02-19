"""Tests for agentlab2.analyze.xray_utils module."""

import json
from pathlib import Path

import pytest
from PIL import Image

from agentlab2.analyze import xray_utils
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
from agentlab2.llm import LLMCall, LLMConfig, Message, Prompt, Usage


# ---------------------------------------------------------------------------
# Additional fixtures (complement conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def red_image() -> Image.Image:
    return Image.new("RGB", (100, 100), color="red")


@pytest.fixture
def env_step_with_screenshot(red_image: Image.Image) -> EnvironmentOutput:
    obs = Observation(contents=[Content(data=red_image, name="screenshot")])
    return EnvironmentOutput(obs=obs, reward=0.0, done=False)


@pytest.fixture
def env_step_with_axtree() -> EnvironmentOutput:
    axtree_text = "RootWebArea 'Example'\n  button 'Submit'"
    obs = Observation(contents=[Content(data=axtree_text, name="axtree_txt")])
    return EnvironmentOutput(obs=obs, reward=0.0, done=False)


@pytest.fixture
def env_step_done_success() -> EnvironmentOutput:
    obs = Observation(contents=[Content(data="Done!", name="goal")])
    return EnvironmentOutput(obs=obs, reward=1.0, done=True)


@pytest.fixture
def env_step_done_failure() -> EnvironmentOutput:
    obs = Observation(contents=[Content(data="Failed", name="goal")])
    return EnvironmentOutput(obs=obs, reward=0.0, done=True)


@pytest.fixture
def sample_llm_call() -> LLMCall:
    config = LLMConfig(model_name="gpt-test")
    prompt = Prompt(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Click the button."},
        ],
        tools=[],
    )
    msg = Message(role="assistant", content="I will click the button.")
    usage = Usage(prompt_tokens=100, completion_tokens=20, cost=0.001)
    return LLMCall(id="test_call", llm_config=config, prompt=prompt, output=msg, usage=usage)


@pytest.fixture
def agent_step_with_llm_call(sample_llm_call: LLMCall) -> AgentOutput:
    return AgentOutput(
        actions=[Action(id="a1", name="click", arguments={"element_id": "btn"})],
        llm_calls=[sample_llm_call],
    )


@pytest.fixture
def timed_trajectory(env_step_with_screenshot: EnvironmentOutput, agent_step_with_llm_call: AgentOutput) -> Trajectory:
    """Trajectory with timing info and metadata."""
    traj = Trajectory(
        id="test_traj",
        metadata={"task_id": "task_1", "agent_name": "agent_a"},
        start_time=0.0,
        end_time=5.0,
        reward_info={"reward": 1.0},
    )
    traj.steps.append(TrajectoryStep(output=env_step_with_screenshot, start_time=0.0, end_time=1.0))
    traj.steps.append(TrajectoryStep(output=agent_step_with_llm_call, start_time=1.0, end_time=2.0))
    traj.steps.append(
        TrajectoryStep(output=EnvironmentOutput(obs=Observation.from_text("done"), reward=1.0, done=True))
    )
    return traj


@pytest.fixture
def multi_agent_trajectories() -> list[Trajectory]:
    """Multiple trajectories with different agents and tasks."""
    trajs = []
    for agent in ["agent_a", "agent_b"]:
        for task in ["task_1", "task_2"]:
            for seed in range(2):
                traj = Trajectory(
                    id=f"{agent}_{task}_{seed}",
                    metadata={"agent_name": agent, "task_id": task, "seed": seed},
                    start_time=float(seed),
                    end_time=float(seed + 1),
                    reward_info={"reward": 1.0 if seed == 0 else 0.0},
                )
                trajs.append(traj)
    return trajs


# ---------------------------------------------------------------------------
# TestFormatDuration
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_milliseconds(self) -> None:
        result = xray_utils.format_duration(0.5)
        assert result == "500ms"

    def test_seconds(self) -> None:
        result = xray_utils.format_duration(4.2)
        assert result == "4.2s"

    def test_minutes_and_seconds(self) -> None:
        result = xray_utils.format_duration(3 * 60 + 12)
        assert result == "3m 12s"

    def test_hours_and_minutes(self) -> None:
        result = xray_utils.format_duration(3600 + 5 * 60)
        assert result == "1h 5m"

    def test_boundary_at_one_second(self) -> None:
        # Exactly 1 second goes to the seconds branch, not ms
        result = xray_utils.format_duration(1.0)
        assert result == "1.0s"

    def test_just_below_one_second(self) -> None:
        result = xray_utils.format_duration(0.999)
        assert result == "999ms"


# ---------------------------------------------------------------------------
# TestGetDirectoryContents
# ---------------------------------------------------------------------------


class TestGetDirectoryContents:
    def test_returns_sentinel_for_missing_dir(self, tmp_path: Path) -> None:
        result = xray_utils.get_directory_contents(tmp_path / "nonexistent")
        assert result == ["Select experiment directory"]

    def test_returns_dirs_with_trajectories_subdir(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "my_exp"
        (exp_dir / "trajectories").mkdir(parents=True)
        result = xray_utils.get_directory_contents(tmp_path)
        assert any("my_exp" in entry for entry in result)

    def test_ignores_dirs_without_trajectories(self, tmp_path: Path) -> None:
        (tmp_path / "no_traj_dir").mkdir()
        result = xray_utils.get_directory_contents(tmp_path)
        assert not any("no_traj_dir" in entry for entry in result)

    def test_count_is_based_on_jsonl_files(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        traj_dir = exp_dir / "trajectories"
        traj_dir.mkdir(parents=True)
        (traj_dir / "a.jsonl").write_text("{}")
        (traj_dir / "b.jsonl").write_text("{}")
        result = xray_utils.get_directory_contents(tmp_path)
        assert any("2 trajectories" in entry for entry in result)

    def test_sentinel_is_first_entry(self, tmp_path: Path) -> None:
        result = xray_utils.get_directory_contents(tmp_path)
        assert result[0] == "Select experiment directory"

    def test_sorted_reverse_order(self, tmp_path: Path) -> None:
        for name in ["aaa_exp", "zzz_exp"]:
            (tmp_path / name / "trajectories").mkdir(parents=True)
        result = xray_utils.get_directory_contents(tmp_path)
        names = [e.split(" ")[0] for e in result[1:]]
        assert names == sorted(names, reverse=True)


# ---------------------------------------------------------------------------
# TestGetScreenshotFromStep
# ---------------------------------------------------------------------------


class TestGetScreenshotFromStep:
    def test_returns_none_for_none_input(self) -> None:
        assert xray_utils.get_screenshot_from_step(None) is None

    def test_returns_none_for_agent_output(self, agent_step_with_llm_call: AgentOutput) -> None:
        assert xray_utils.get_screenshot_from_step(agent_step_with_llm_call) is None

    def test_extracts_image_from_env_output(self, env_step_with_screenshot: EnvironmentOutput) -> None:
        img = xray_utils.get_screenshot_from_step(env_step_with_screenshot)
        assert isinstance(img, Image.Image)

    def test_returns_none_when_no_image_content(self) -> None:
        env_step = EnvironmentOutput(obs=Observation.from_text("no image here"))
        assert xray_utils.get_screenshot_from_step(env_step) is None


# ---------------------------------------------------------------------------
# TestGetCurrentScreenshot
# ---------------------------------------------------------------------------


class TestGetCurrentScreenshot:
    def test_returns_current_screenshot_for_env_step(self, env_step_with_screenshot: EnvironmentOutput) -> None:
        img = xray_utils.get_current_screenshot(env_step_with_screenshot, None)
        assert isinstance(img, Image.Image)

    def test_falls_back_to_prev_env_step_for_agent_step(
        self, agent_step_with_llm_call: AgentOutput, env_step_with_screenshot: EnvironmentOutput
    ) -> None:
        img = xray_utils.get_current_screenshot(agent_step_with_llm_call, env_step_with_screenshot)
        assert isinstance(img, Image.Image)

    def test_returns_none_when_no_screenshots(self, agent_step_with_llm_call: AgentOutput) -> None:
        env_no_img = EnvironmentOutput(obs=Observation.from_text("no image"))
        img = xray_utils.get_current_screenshot(agent_step_with_llm_call, env_no_img)
        assert img is None

    def test_returns_none_for_none_inputs(self) -> None:
        assert xray_utils.get_current_screenshot(None, None) is None


# ---------------------------------------------------------------------------
# TestExtractObsContent
# ---------------------------------------------------------------------------


class TestExtractObsContent:
    def test_finds_axtree_by_name(self, env_step_with_axtree: EnvironmentOutput) -> None:
        result = xray_utils.extract_obs_content(env_step_with_axtree, "axtree")
        assert result == "RootWebArea 'Example'\n  button 'Submit'"

    def test_case_insensitive_match(self, env_step_with_axtree: EnvironmentOutput) -> None:
        result = xray_utils.extract_obs_content(env_step_with_axtree, "AXTREE")
        assert result is not None

    def test_returns_none_for_agent_output(self, agent_step_with_llm_call: AgentOutput) -> None:
        result = xray_utils.extract_obs_content(agent_step_with_llm_call, "axtree")  # type: ignore[arg-type]
        assert result is None

    def test_returns_none_when_no_match(self, env_step_with_axtree: EnvironmentOutput) -> None:
        result = xray_utils.extract_obs_content(env_step_with_axtree, "nonexistent_key")
        assert result is None

    def test_returns_none_for_image_content(self, env_step_with_screenshot: EnvironmentOutput) -> None:
        # screenshot content is not str, should not be returned
        result = xray_utils.extract_obs_content(env_step_with_screenshot, "screenshot")
        assert result is None

    def test_returns_none_for_none_input(self) -> None:
        assert xray_utils.extract_obs_content(None, "axtree") is None


# ---------------------------------------------------------------------------
# TestGetChatMessagesMarkdown
# ---------------------------------------------------------------------------


class TestGetChatMessagesMarkdown:
    def test_returns_empty_for_env_step(self, env_step_with_axtree: EnvironmentOutput) -> None:
        assert xray_utils.get_chat_messages_markdown(env_step_with_axtree) == ""

    def test_returns_empty_for_none(self) -> None:
        assert xray_utils.get_chat_messages_markdown(None) == ""

    def test_returns_empty_when_no_llm_calls(self) -> None:
        step = AgentOutput(actions=[], llm_calls=[])
        assert xray_utils.get_chat_messages_markdown(step) == ""

    def test_contains_role_headers(self, agent_step_with_llm_call: AgentOutput) -> None:
        result = xray_utils.get_chat_messages_markdown(agent_step_with_llm_call)
        assert "system" in result
        assert "user" in result

    def test_contains_message_content(self, agent_step_with_llm_call: AgentOutput) -> None:
        result = xray_utils.get_chat_messages_markdown(agent_step_with_llm_call)
        assert "You are a helpful assistant." in result
        assert "Click the button." in result

    def test_truncates_long_content(self, sample_llm_call: LLMCall) -> None:
        long_content = "x" * 400000
        sample_llm_call.prompt.messages[1] = {"role": "user", "content": long_content}
        step = AgentOutput(llm_calls=[sample_llm_call])
        result = xray_utils.get_chat_messages_markdown(step)
        assert "[truncated]" in result

    def test_handles_list_content_with_image(self, sample_llm_call: LLMCall) -> None:
        sample_llm_call.prompt.messages[1] = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is a screenshot:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
        step = AgentOutput(llm_calls=[sample_llm_call])
        result = xray_utils.get_chat_messages_markdown(step)
        assert "Here is a screenshot:" in result
        assert "[Image]" in result


# ---------------------------------------------------------------------------
# TestGetStepErrorMarkdown
# ---------------------------------------------------------------------------


class TestGetStepErrorMarkdown:
    def test_no_error_returns_message(self) -> None:
        env_step = EnvironmentOutput(obs=Observation.from_text("ok"))
        result = xray_utils.get_step_error_markdown(env_step)
        assert "No errors" in result

    def test_none_returns_message(self) -> None:
        result = xray_utils.get_step_error_markdown(None)
        assert "No errors" in result

    def test_agent_output_error(self) -> None:
        err = StepError(error_type="ValueError", exception_str="bad value", stack_trace="traceback here")
        step = AgentOutput(error=err)
        result = xray_utils.get_step_error_markdown(step)
        assert "ValueError" in result
        assert "bad value" in result
        assert "traceback here" in result

    def test_env_output_error(self) -> None:
        err = StepError(error_type="TimeoutError", exception_str="timed out", stack_trace="...")
        env_step = EnvironmentOutput(obs=Observation.from_text("failed"), error=err)
        result = xray_utils.get_step_error_markdown(env_step)
        assert "TimeoutError" in result

    def test_env_info_error_fallback(self) -> None:
        env_step = EnvironmentOutput(obs=Observation.from_text("ok"), info={"error": "Something went wrong"})
        result = xray_utils.get_step_error_markdown(env_step)
        assert "Something went wrong" in result


# ---------------------------------------------------------------------------
# TestGetStepLogsMarkdown
# ---------------------------------------------------------------------------


class TestGetStepLogsMarkdown:
    def test_returns_no_log_message_when_empty(self) -> None:
        env_step = EnvironmentOutput(obs=Observation.from_text("ok"))
        result = xray_utils.get_step_logs_markdown(env_step, None)
        assert "No log information" in result

    def test_shows_info_keys(self) -> None:
        env_step = EnvironmentOutput(obs=Observation.from_text("ok"), info={"url": "https://example.com"})
        result = xray_utils.get_step_logs_markdown(env_step, None)
        assert "url" in result
        assert "example.com" in result

    def test_excludes_error_and_message_keys(self) -> None:
        env_step = EnvironmentOutput(
            obs=Observation.from_text("ok"),
            info={"error": "bad", "message": "task done", "url": "https://x.com"},
        )
        result = xray_utils.get_step_logs_markdown(env_step, None)
        assert "error" not in result
        assert "message" not in result or "url" in result

    def test_shows_trajectory_metadata(self) -> None:
        env_step = EnvironmentOutput(obs=Observation.from_text("ok"))
        traj = Trajectory(id="t1", metadata={"agent_name": "test_agent", "task_id": "task_1"})
        result = xray_utils.get_step_logs_markdown(env_step, traj)
        assert "test_agent" in result
        assert "task_1" in result

    def test_agent_output_shows_only_metadata(self, agent_step_with_llm_call: AgentOutput) -> None:
        traj = Trajectory(id="t1", metadata={"agent_name": "agent_a"})
        result = xray_utils.get_step_logs_markdown(agent_step_with_llm_call, traj)
        assert "agent_a" in result


# ---------------------------------------------------------------------------
# TestComputeTrajectoryStats
# ---------------------------------------------------------------------------


class TestComputeTrajectoryStats:
    def test_empty_trajectory(self) -> None:
        traj = Trajectory(id="empty")
        stats = xray_utils.compute_trajectory_stats(traj)
        assert stats["n_env_steps"] == 0
        assert stats["n_agent_steps"] == 0
        assert stats["total_actions"] == 0
        assert stats["final_reward"] == 0.0
        assert stats["duration"] is None

    def test_counts_env_and_agent_steps(self, timed_trajectory: Trajectory) -> None:
        stats = xray_utils.compute_trajectory_stats(timed_trajectory)
        assert stats["n_env_steps"] == 2
        assert stats["n_agent_steps"] == 1

    def test_duration_from_timing(self, timed_trajectory: Trajectory) -> None:
        stats = xray_utils.compute_trajectory_stats(timed_trajectory)
        assert stats["duration"] == pytest.approx(5.0)

    def test_final_reward_from_reward_info(self, timed_trajectory: Trajectory) -> None:
        stats = xray_utils.compute_trajectory_stats(timed_trajectory)
        assert stats["final_reward"] == 1.0

    def test_aggregates_token_usage(self, timed_trajectory: Trajectory) -> None:
        stats = xray_utils.compute_trajectory_stats(timed_trajectory)
        # The agent step has usage: prompt=100, completion=20
        assert stats["prompt_tokens"] == 100
        assert stats["completion_tokens"] == 20

    def test_total_llm_calls(self, timed_trajectory: Trajectory) -> None:
        stats = xray_utils.compute_trajectory_stats(timed_trajectory)
        assert stats["total_llm_calls"] == 1


# ---------------------------------------------------------------------------
# TestComputeExperimentStats
# ---------------------------------------------------------------------------


class TestComputeExperimentStats:
    def test_empty_list_returns_empty_string(self) -> None:
        assert xray_utils.compute_experiment_stats([]) == ""

    def test_single_finished_trajectory(self, timed_trajectory: Trajectory) -> None:
        result = xray_utils.compute_experiment_stats([timed_trajectory])
        assert "1" in result
        assert "Finished" in result

    def test_counts_failed_trajectories(self) -> None:
        # Trajectory without start/end time is considered "failed"
        failed_traj = Trajectory(id="failed")
        result = xray_utils.compute_experiment_stats([failed_traj])
        assert "Failed" in result

    def test_computes_success_rate(self, timed_trajectory: Trajectory) -> None:
        result = xray_utils.compute_experiment_stats([timed_trajectory])
        assert "Success Rate" in result

    def test_shows_token_totals(self, timed_trajectory: Trajectory) -> None:
        result = xray_utils.compute_experiment_stats([timed_trajectory])
        assert "prompt" in result


# ---------------------------------------------------------------------------
# TestBuildAgentTable
# ---------------------------------------------------------------------------


class TestBuildAgentTable:
    def test_empty_list(self) -> None:
        assert xray_utils.build_agent_table([]) == []

    def test_single_agent(self, timed_trajectory: Trajectory) -> None:
        rows = xray_utils.build_agent_table([timed_trajectory])
        assert len(rows) == 1
        assert rows[0]["agent_name"] == "agent_a"

    def test_groups_by_agent_name(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_agent_table(multi_agent_trajectories)
        names = [r["agent_name"] for r in rows]
        assert "agent_a" in names
        assert "agent_b" in names
        assert len(rows) == 2

    def test_unknown_fallback_for_missing_agent_name(self) -> None:
        traj = Trajectory(id="no_agent", metadata={"task_id": "t1"})
        rows = xray_utils.build_agent_table([traj])
        assert rows[0]["agent_name"] == "unknown"

    def test_counts_tasks(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_agent_table(multi_agent_trajectories)
        agent_a_row = next(r for r in rows if r["agent_name"] == "agent_a")
        assert agent_a_row["n_tasks"] == 2

    def test_total_cost_dash_for_unloaded_stubs(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """total_cost shows '-' when no cost data is available (metadata stubs have steps=[])."""
        rows = xray_utils.build_agent_table(multi_agent_trajectories)
        agent_a_row = next(r for r in rows if r["agent_name"] == "agent_a")
        assert agent_a_row["total_cost"] == "-"

    def test_no_success_rate_column(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """success_rate was removed from the agent table."""
        rows = xray_utils.build_agent_table(multi_agent_trajectories)
        assert len(rows) > 0
        assert "success_rate" not in rows[0]


# ---------------------------------------------------------------------------
# TestBuildTaskTable
# ---------------------------------------------------------------------------


class TestBuildTaskTable:
    def test_filters_by_agent_key(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_task_table(multi_agent_trajectories, "agent_a")
        task_ids = [r["task_id"] for r in rows]
        assert "task_1" in task_ids
        assert "task_2" in task_ids
        assert len(rows) == 2

    def test_groups_by_task_id(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_task_table(multi_agent_trajectories, "agent_a")
        task_1_row = next(r for r in rows if r["task_id"] == "task_1")
        assert task_1_row["n_seeds"] == 2

    def test_returns_empty_for_unknown_agent(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_task_table(multi_agent_trajectories, "nonexistent_agent")
        assert rows == []

    def test_has_no_success_rate_column(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """success_rate was removed; only avg_reward is shown."""
        rows = xray_utils.build_task_table(multi_agent_trajectories, "agent_a")
        assert len(rows) > 0
        assert "success_rate" not in rows[0]
        assert "avg_reward" in rows[0]

    def test_avg_reward_value(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """avg_reward is mean numeric reward (0.5 for 1.0 + 0.0)."""
        rows = xray_utils.build_task_table(multi_agent_trajectories, "agent_a")
        task_1_row = next(r for r in rows if r["task_id"] == "task_1")
        assert task_1_row["avg_reward"] == 0.5

    def test_avg_duration_present(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """avg_duration is present and shows a formatted string."""
        rows = xray_utils.build_task_table(multi_agent_trajectories, "agent_a")
        task_1_row = next(r for r in rows if r["task_id"] == "task_1")
        assert "avg_duration" in task_1_row
        # Each seed is 1.0s long → avg = 1.0s
        assert task_1_row["avg_duration"] == "1.0s"

    def test_avg_duration_missing_when_no_timing(self) -> None:
        """avg_duration falls back to '-' when trajectories have no timing info."""
        trajs = [
            Trajectory(
                id="t1",
                metadata={"agent_name": "agent_a", "task_id": "task_x"},
                reward_info={"reward": 1.0},
            )
        ]
        rows = xray_utils.build_task_table(trajs, "agent_a")
        assert rows[0]["avg_duration"] == "-"

    def test_avg_steps_present(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """avg_steps shows '-' for metadata stubs (no loaded steps)."""
        rows = xray_utils.build_task_table(multi_agent_trajectories, "agent_a")
        task_1_row = next(r for r in rows if r["task_id"] == "task_1")
        assert "avg_steps" in task_1_row
        # Stubs have steps=[] → shows "-"
        assert task_1_row["avg_steps"] == "-"

    def test_avg_tokens_and_cost_present(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """avg_tokens and avg_cost show '-' for unloaded metadata stubs."""
        rows = xray_utils.build_task_table(multi_agent_trajectories, "agent_a")
        task_1_row = next(r for r in rows if r["task_id"] == "task_1")
        assert task_1_row["avg_tokens"] == "-"
        assert task_1_row["avg_cost"] == "-"


# ---------------------------------------------------------------------------
# TestBuildSeedTable
# ---------------------------------------------------------------------------


class TestBuildSeedTable:
    def test_filters_by_agent_and_task(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_seed_table(multi_agent_trajectories, "agent_a", "task_1")
        assert len(rows) == 2
        traj_ids = [r["traj_id"] for r in rows]
        assert "agent_a_task_1_0" in traj_ids
        assert "agent_a_task_1_1" in traj_ids

    def test_one_row_per_trajectory(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_seed_table(multi_agent_trajectories, "agent_b", "task_2")
        assert len(rows) == 2

    def test_includes_reward(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_seed_table(multi_agent_trajectories, "agent_a", "task_1")
        rewards = [r["reward"] for r in rows]
        assert 1.0 in rewards
        assert 0.0 in rewards

    def test_returns_empty_for_unknown_combination(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_seed_table(multi_agent_trajectories, "agent_a", "nonexistent_task")
        assert rows == []


# ---------------------------------------------------------------------------
# TestComputeStepWidth
# ---------------------------------------------------------------------------


class TestComputeStepWidth:
    def test_returns_min_width_for_none_duration(self) -> None:
        result = xray_utils._compute_step_width(None, 0.0, 1.0)
        assert result == xray_utils._MIN_WIDTH

    def test_returns_min_width_when_min_equals_max(self) -> None:
        result = xray_utils._compute_step_width(0.5, 0.5, 0.5)
        assert result == xray_utils._MIN_WIDTH

    def test_returns_min_width_for_min_duration(self) -> None:
        result = xray_utils._compute_step_width(0.0, 0.0, 10.0)
        assert result == xray_utils._MIN_WIDTH

    def test_returns_max_width_for_max_duration(self) -> None:
        result = xray_utils._compute_step_width(10.0, 0.0, 10.0)
        assert result == xray_utils._MAX_WIDTH

    def test_scales_linearly_between_min_and_max(self) -> None:
        min_w = xray_utils._MIN_WIDTH
        max_w = xray_utils._MAX_WIDTH
        mid = xray_utils._compute_step_width(5.0, 0.0, 10.0)
        assert min_w < mid < max_w
        expected = int(min_w + 0.5 * (max_w - min_w))
        assert mid == expected


# ---------------------------------------------------------------------------
# TestGenerateTimelineHtml
# ---------------------------------------------------------------------------


class TestGenerateTimelineHtml:
    def test_empty_trajectory_returns_placeholder(self) -> None:
        traj = Trajectory(id="empty")
        result = xray_utils.generate_timeline_html(traj, 0)
        assert "No trajectory loaded" in result

    def test_none_trajectory_returns_placeholder(self) -> None:
        result = xray_utils.generate_timeline_html(None, 0)
        assert "No trajectory loaded" in result

    def test_contains_step_numbers(self, timed_trajectory: Trajectory) -> None:
        result = xray_utils.generate_timeline_html(timed_trajectory, 0)
        # Step 1, 2, 3 should appear as text in segments
        assert ">1<" in result
        assert ">2<" in result

    def test_current_step_has_gold_border(self, timed_trajectory: Trajectory) -> None:
        result = xray_utils.generate_timeline_html(timed_trajectory, 0)
        assert xray_utils._CURRENT_BORDER_COLOR in result

    def test_different_colors_for_env_and_agent(self, timed_trajectory: Trajectory) -> None:
        result = xray_utils.generate_timeline_html(timed_trajectory, 0)
        assert xray_utils._ENV_COLOR in result
        assert xray_utils._AGENT_COLOR in result

    def test_done_step_has_success_border(self) -> None:
        env_done = EnvironmentOutput(obs=Observation.from_text("done"), reward=1.0, done=True)
        traj = Trajectory(id="t")
        traj.steps.append(TrajectoryStep(output=env_done))
        result = xray_utils.generate_timeline_html(traj, 0)
        assert xray_utils._SUCCESS_BORDER_COLOR in result

    def test_failed_step_has_failure_border(self) -> None:
        env_failed = EnvironmentOutput(obs=Observation.from_text("failed"), reward=0.0, done=True)
        traj = Trajectory(id="t")
        traj.steps.append(TrajectoryStep(output=env_failed))
        result = xray_utils.generate_timeline_html(traj, 0)
        assert xray_utils._FAILURE_BORDER_COLOR in result

    def test_legend_present(self, timed_trajectory: Trajectory) -> None:
        result = xray_utils.generate_timeline_html(timed_trajectory, 0)
        assert "Env" in result
        assert "Agent" in result
        assert "Current" in result


# ---------------------------------------------------------------------------
# TestGetStepDetailsMarkdown
# ---------------------------------------------------------------------------


class TestGetStepDetailsMarkdown:
    def test_none_step_returns_placeholder(self) -> None:
        result = xray_utils.get_step_details_markdown(None, None)
        assert "No step selected" in result

    def test_env_step_shows_reward(self, env_step_done_success: EnvironmentOutput) -> None:
        result = xray_utils.get_step_details_markdown(env_step_done_success, None)
        assert "Reward" in result
        assert "1.00" in result

    def test_env_step_shows_done_status(self, env_step_done_success: EnvironmentOutput) -> None:
        result = xray_utils.get_step_details_markdown(env_step_done_success, None)
        assert "Success" in result

    def test_agent_step_shows_actions(self, agent_step_with_llm_call: AgentOutput) -> None:
        result = xray_utils.get_step_details_markdown(agent_step_with_llm_call, None)
        assert "click" in result
        assert "btn" in result

    def test_agent_step_shows_token_usage(self, agent_step_with_llm_call: AgentOutput) -> None:
        result = xray_utils.get_step_details_markdown(agent_step_with_llm_call, None)
        assert "100" in result  # prompt_tokens

    def test_includes_duration_when_timing_available(self, agent_step_with_llm_call: AgentOutput) -> None:
        traj_step = TrajectoryStep(output=agent_step_with_llm_call, start_time=0.0, end_time=2.5)
        result = xray_utils.get_step_details_markdown(agent_step_with_llm_call, traj_step)
        assert "2.5s" in result

    def test_env_step_shows_content_names(self) -> None:
        obs = Observation(
            contents=[
                Content(data="goal text", name="goal"),
                Content(data="some html", name="html_content"),
            ]
        )
        env_step = EnvironmentOutput(obs=obs)
        result = xray_utils.get_step_details_markdown(env_step, None)
        assert "goal" in result
        assert "html_content" in result


# ---------------------------------------------------------------------------
# TestGetTaskGoal
# ---------------------------------------------------------------------------


class TestGetTaskGoal:
    def test_returns_placeholder_for_none(self) -> None:
        result = xray_utils.get_task_goal(None)
        assert "No trajectory" in result

    def test_returns_placeholder_for_empty_trajectory(self) -> None:
        traj = Trajectory(id="empty")
        result = xray_utils.get_task_goal(traj)
        assert "No goal text found" in result

    def test_extracts_first_text_from_first_env_step(self) -> None:
        obs = Observation(contents=[Content(data="Buy 3 apples", name="goal")])
        env_out = EnvironmentOutput(obs=obs)
        traj = Trajectory(id="t1")
        traj.steps.append(TrajectoryStep(output=env_out))
        result = xray_utils.get_task_goal(traj)
        assert result == "Buy 3 apples"

    def test_skips_agent_steps_at_start(self) -> None:
        agent_out = AgentOutput(actions=[])
        obs = Observation(contents=[Content(data="Real goal")])
        env_out = EnvironmentOutput(obs=obs)
        traj = Trajectory(id="t1")
        traj.steps.append(TrajectoryStep(output=agent_out))
        traj.steps.append(TrajectoryStep(output=env_out))
        result = xray_utils.get_task_goal(traj)
        assert result == "Real goal"

    def test_skips_image_contents(self, red_image: Image.Image) -> None:
        obs = Observation(
            contents=[
                Content(data=red_image, name="screenshot"),
                Content(data="Goal text after image"),
            ]
        )
        env_out = EnvironmentOutput(obs=obs)
        traj = Trajectory(id="t1")
        traj.steps.append(TrajectoryStep(output=env_out))
        result = xray_utils.get_task_goal(traj)
        assert result == "Goal text after image"

    def test_skips_whitespace_only_content(self) -> None:
        obs = Observation(
            contents=[
                Content(data="   \n  "),
                Content(data="Actual goal"),
            ]
        )
        env_out = EnvironmentOutput(obs=obs)
        traj = Trajectory(id="t1")
        traj.steps.append(TrajectoryStep(output=env_out))
        result = xray_utils.get_task_goal(traj)
        assert result == "Actual goal"


# ---------------------------------------------------------------------------
# TestGetAgentActionMarkdown
# ---------------------------------------------------------------------------


class TestGetAgentActionMarkdown:
    def test_returns_terminal_placeholder_for_none(self) -> None:
        result = xray_utils.get_agent_action_markdown(None)
        assert "Terminal" in result or "terminal" in result

    def test_returns_no_actions_placeholder_when_empty(self) -> None:
        agent_out = AgentOutput(actions=[])
        result = xray_utils.get_agent_action_markdown(agent_out)
        assert "No actions" in result

    def test_formats_as_function_call(self) -> None:
        action = Action(name="browser_click", arguments={"bid": "a42", "button": "left"})
        agent_out = AgentOutput(actions=[action])
        result = xray_utils.get_agent_action_markdown(agent_out)
        assert "browser_click" in result
        assert "bid" in result
        assert "a42" in result

    def test_uses_backtick_code_format(self) -> None:
        action = Action(name="noop", arguments={})
        agent_out = AgentOutput(actions=[action])
        result = xray_utils.get_agent_action_markdown(agent_out)
        assert "`noop()`" in result

    def test_truncates_long_string_arguments(self) -> None:
        long_text = "x" * 300
        action = Action(name="browser_type", arguments={"text": long_text})
        agent_out = AgentOutput(actions=[action])
        result = xray_utils.get_agent_action_markdown(agent_out)
        assert "…" in result
        assert "x" * 300 not in result

    def test_renders_multiple_actions(self) -> None:
        actions = [
            Action(name="browser_click", arguments={"bid": "a1"}),
            Action(name="browser_type", arguments={"bid": "b1", "text": "hello"}),
        ]
        agent_out = AgentOutput(actions=actions)
        result = xray_utils.get_agent_action_markdown(agent_out)
        assert "browser_click" in result
        assert "browser_type" in result

    def test_formats_non_string_arguments(self) -> None:
        action = Action(name="scroll", arguments={"delta_x": 0, "delta_y": 100, "relative": True})
        agent_out = AgentOutput(actions=[action])
        result = xray_utils.get_agent_action_markdown(agent_out)
        assert "delta_y=100" in result
        assert "relative=True" in result

    def test_empty_arguments_dict(self) -> None:
        action = Action(name="noop", arguments={})
        agent_out = AgentOutput(actions=[action])
        result = xray_utils.get_agent_action_markdown(agent_out)
        assert "noop()" in result


# ---------------------------------------------------------------------------
# TestGetPairedStepDetailsMarkdown
# ---------------------------------------------------------------------------


class TestGetPairedStepDetailsMarkdown:
    def test_returns_placeholder_when_env_none(self) -> None:
        result = xray_utils.get_paired_step_details_markdown(None, None, None, None)
        assert "No step selected" in result

    def test_contains_env_section(self, env_step_done_success: EnvironmentOutput) -> None:
        result = xray_utils.get_paired_step_details_markdown(env_step_done_success, None, None, None)
        assert "Environment" in result

    def test_contains_terminal_marker_when_no_agent(self, env_step_done_success: EnvironmentOutput) -> None:
        result = xray_utils.get_paired_step_details_markdown(env_step_done_success, None, None, None)
        assert "terminal" in result.lower() or "No agent action" in result

    def test_contains_agent_section_when_agent_present(
        self,
        env_step_done_success: EnvironmentOutput,
        agent_step_with_llm_call: AgentOutput,
    ) -> None:
        result = xray_utils.get_paired_step_details_markdown(
            env_step_done_success, agent_step_with_llm_call, None, None
        )
        assert "Agent" in result
        assert "click" in result

    def test_includes_env_duration_from_traj_step(
        self, env_step_done_success: EnvironmentOutput
    ) -> None:
        ts = TrajectoryStep(output=env_step_done_success, start_time=0.0, end_time=3.0)
        result = xray_utils.get_paired_step_details_markdown(env_step_done_success, None, ts, None)
        assert "3.0s" in result

    def test_includes_agent_duration_from_traj_step(
        self,
        env_step_done_success: EnvironmentOutput,
        agent_step_with_llm_call: AgentOutput,
    ) -> None:
        env_ts = TrajectoryStep(output=env_step_done_success, start_time=0.0, end_time=1.0)
        agent_ts = TrajectoryStep(output=agent_step_with_llm_call, start_time=1.0, end_time=4.5)
        result = xray_utils.get_paired_step_details_markdown(
            env_step_done_success, agent_step_with_llm_call, env_ts, agent_ts
        )
        assert "3.5s" in result

    def test_shows_reward_for_done_step(self, env_step_done_success: EnvironmentOutput) -> None:
        result = xray_utils.get_paired_step_details_markdown(env_step_done_success, None, None, None)
        assert "1.00" in result


# ---------------------------------------------------------------------------
# TestGetPairedErrorMarkdown
# ---------------------------------------------------------------------------


class TestGetPairedErrorMarkdown:
    def test_no_errors_returns_message(self) -> None:
        env_out = EnvironmentOutput(obs=Observation.from_text("ok"))
        result = xray_utils.get_paired_error_markdown(env_out, None)
        assert "No errors" in result

    def test_shows_env_error(self) -> None:
        err = StepError(error_type="TimeoutError", exception_str="timed out", stack_trace="...")
        env_out = EnvironmentOutput(obs=Observation.from_text("failed"), error=err)
        result = xray_utils.get_paired_error_markdown(env_out, None)
        assert "TimeoutError" in result
        assert "timed out" in result

    def test_shows_agent_error(self) -> None:
        err = StepError(error_type="ValueError", exception_str="bad input", stack_trace="...")
        agent_out = AgentOutput(error=err)
        env_out = EnvironmentOutput(obs=Observation.from_text("ok"))
        result = xray_utils.get_paired_error_markdown(env_out, agent_out)
        assert "ValueError" in result

    def test_shows_both_errors_when_present(self) -> None:
        env_err = StepError(error_type="EnvError", exception_str="env failed", stack_trace="a")
        agent_err = StepError(error_type="AgentError", exception_str="agent failed", stack_trace="b")
        env_out = EnvironmentOutput(obs=Observation.from_text("ok"), error=env_err)
        agent_out = AgentOutput(error=agent_err)
        result = xray_utils.get_paired_error_markdown(env_out, agent_out)
        assert "EnvError" in result
        assert "AgentError" in result

    def test_env_info_error_fallback(self) -> None:
        env_out = EnvironmentOutput(obs=Observation.from_text("ok"), info={"error": "page crashed"})
        result = xray_utils.get_paired_error_markdown(env_out, None)
        assert "page crashed" in result

    def test_no_errors_when_both_none(self) -> None:
        env_out = EnvironmentOutput(obs=Observation.from_text("ok"))
        agent_out = AgentOutput(actions=[])
        result = xray_utils.get_paired_error_markdown(env_out, agent_out)
        assert "No errors" in result


# ---------------------------------------------------------------------------
# TestGetChatMessagesMarkdownWithMessageObjects
# ---------------------------------------------------------------------------


class TestGetChatMessagesMarkdownWithMessageObjects:
    """Additional tests for Message object handling (not just dicts)."""

    def test_handles_litellm_message_objects(self) -> None:
        """LLMCall.output is a Message object; prompts can also contain Message objects."""
        config = LLMConfig(model_name="gpt-test")
        msg = Message(role="user", content="Use a Message object")
        prompt = Prompt(messages=[msg], tools=[])
        output_msg = Message(role="assistant", content="ok")
        llm_call = LLMCall(id="test", llm_config=config, prompt=prompt, output=output_msg, usage=Usage())
        step = AgentOutput(llm_calls=[llm_call])
        result = xray_utils.get_chat_messages_markdown(step)
        assert "user" in result
        assert "Use a Message object" in result

    def test_handles_mixed_dict_and_message_in_messages(self) -> None:
        config = LLMConfig(model_name="gpt-test")
        msg_obj = Message(role="system", content="System prompt from Message")
        msg_dict = {"role": "user", "content": "User dict message"}
        prompt = Prompt(messages=[msg_obj, msg_dict], tools=[])
        output_msg = Message(role="assistant", content="ok")
        llm_call = LLMCall(id="test2", llm_config=config, prompt=prompt, output=output_msg, usage=Usage())
        step = AgentOutput(llm_calls=[llm_call])
        result = xray_utils.get_chat_messages_markdown(step)
        assert "system" in result
        assert "System prompt from Message" in result
        assert "User dict message" in result
