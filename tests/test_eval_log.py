"""Tests for cube_harness.eval_log — Atlas EvalLog system."""

import json
import tempfile
from pathlib import Path

import pytest
from cube.core import Content, EnvironmentOutput, Observation, StepError

from cube_harness.core import AgentOutput, Trajectory, TrajectoryStep
from cube_harness.eval_log import (
    AgentInfo,
    EvalLog,
    TaskEvalRecord,
    TaskInfo,
    UsageSummary,
    _extract_error_type,
    _extract_first_observation_text,
    _extract_llm_model,
    _extract_tool_names,
    _to_github_url,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_output(reward: float = 0.0, done: bool = True, text: str = "Task: do it") -> EnvironmentOutput:
    obs = Observation(contents=[Content.from_data(text)])
    return EnvironmentOutput(obs=obs, reward=reward, done=done, info={})


def _trajectory(reward: float = 1.0, task_id: str = "t1", n_agent_steps: int = 1) -> Trajectory:
    """Build a minimal completed trajectory for testing."""
    traj = Trajectory(
        id=f"{task_id}_ep0",
        metadata={"task_id": task_id},
        start_time=100.0,
        end_time=110.0,
        reward_info={"reward": reward, "done": reward > 0},
        summary_stats={
            "n_agent_steps": n_agent_steps,
            "n_env_steps": n_agent_steps + 1,
            "total_llm_calls": n_agent_steps,
            "prompt_tokens": 100 * n_agent_steps,
            "completion_tokens": 50 * n_agent_steps,
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
            "cost": 0.01 * n_agent_steps,
        },
    )
    traj.steps.append(TrajectoryStep(output=_env_output(reward=0.0, done=False), start_time=100.0, end_time=101.0))
    for _ in range(n_agent_steps):
        traj.steps.append(TrajectoryStep(output=AgentOutput(actions=[]), start_time=101.0, end_time=102.0))
    traj.steps.append(TrajectoryStep(output=_env_output(reward=reward, done=reward > 0), start_time=102.0, end_time=103.0))
    return traj


# ---------------------------------------------------------------------------
# _extract_llm_model
# ---------------------------------------------------------------------------


def test_extract_llm_model_top_level_model_name() -> None:
    assert _extract_llm_model({"model_name": "gpt-4o"}) == "gpt-4o"


def test_extract_llm_model_top_level_model() -> None:
    assert _extract_llm_model({"model": "claude-3-5-sonnet"}) == "claude-3-5-sonnet"


def test_extract_llm_model_nested_llm_config() -> None:
    assert _extract_llm_model({"llm_config": {"model_name": "gpt-4o-mini"}}) == "gpt-4o-mini"


def test_extract_llm_model_nested_llm() -> None:
    assert _extract_llm_model({"llm": {"model": "o1"}}) == "o1"


def test_extract_llm_model_returns_none_when_absent() -> None:
    assert _extract_llm_model({"temperature": 0.7}) is None


def test_extract_llm_model_ignores_non_string_values() -> None:
    assert _extract_llm_model({"model_name": 42}) is None


# ---------------------------------------------------------------------------
# _extract_tool_names
# ---------------------------------------------------------------------------


def test_extract_tool_names_litellm_format() -> None:
    tools = [{"type": "function", "function": {"name": "click", "description": "Click"}}]
    assert _extract_tool_names(tools) == ["click"]


def test_extract_tool_names_flat_format() -> None:
    tools = [{"name": "type_text", "description": "Type"}]
    assert _extract_tool_names(tools) == ["type_text"]


def test_extract_tool_names_mixed_formats() -> None:
    tools = [
        {"type": "function", "function": {"name": "click"}},
        {"name": "scroll"},
    ]
    assert _extract_tool_names(tools) == ["click", "scroll"]


def test_extract_tool_names_empty_list() -> None:
    assert _extract_tool_names([]) == []


def test_extract_tool_names_skips_tools_without_name() -> None:
    tools = [{"type": "function", "function": {"description": "No name here"}}]
    assert _extract_tool_names(tools) == []


# ---------------------------------------------------------------------------
# _extract_error_type
# ---------------------------------------------------------------------------


def test_extract_error_type_clean_trajectory() -> None:
    traj = _trajectory(reward=1.0)
    assert _extract_error_type(traj) is None


def test_extract_error_type_from_agent_output() -> None:
    traj = _trajectory(reward=0.0)
    err = StepError(error_type="ValueError", exception_str="bad value", stack_trace="")
    traj.steps.insert(1, TrajectoryStep(output=AgentOutput(error=err), start_time=101.0, end_time=101.5))
    assert _extract_error_type(traj) == "ValueError"


def test_extract_error_type_returns_first_error() -> None:
    traj = _trajectory(reward=0.0)
    err1 = StepError(error_type="TimeoutError", exception_str="timeout", stack_trace="")
    err2 = StepError(error_type="ValueError", exception_str="bad", stack_trace="")
    traj.steps.insert(1, TrajectoryStep(output=AgentOutput(error=err1), start_time=101.0, end_time=101.5))
    traj.steps.insert(2, TrajectoryStep(output=AgentOutput(error=err2), start_time=101.5, end_time=102.0))
    assert _extract_error_type(traj) == "TimeoutError"


# ---------------------------------------------------------------------------
# _extract_first_observation_text
# ---------------------------------------------------------------------------


def test_extract_first_observation_text_returns_text() -> None:
    traj = _trajectory(reward=1.0)
    text = _extract_first_observation_text(traj)
    assert text == "Task: do it"


def test_extract_first_observation_text_empty_trajectory() -> None:
    traj = Trajectory(id="empty")
    assert _extract_first_observation_text(traj) is None


def test_extract_first_observation_text_joins_multiple_contents() -> None:
    obs = Observation(contents=[Content.from_data("Goal:"), Content.from_data("Click the button")])
    env_out = EnvironmentOutput(obs=obs, reward=0.0, done=False, info={})
    traj = Trajectory(id="t1")
    traj.steps.append(TrajectoryStep(output=env_out, start_time=0.0, end_time=1.0))
    text = _extract_first_observation_text(traj)
    assert "Goal:" in text
    assert "Click the button" in text


# ---------------------------------------------------------------------------
# _to_github_url
# ---------------------------------------------------------------------------


def test_to_github_url_ssh() -> None:
    url = _to_github_url("git@github.com:org/repo.git", "abc123")
    assert url == "https://github.com/org/repo/tree/abc123"


def test_to_github_url_https() -> None:
    url = _to_github_url("https://github.com/org/repo", "abc123")
    assert url == "https://github.com/org/repo/tree/abc123"


def test_to_github_url_non_github_returns_none() -> None:
    url = _to_github_url("https://gitlab.com/org/repo.git", "abc123")
    assert url is None


def test_to_github_url_strips_git_suffix() -> None:
    url = _to_github_url("https://github.com/org/repo.git", "sha1")
    assert url == "https://github.com/org/repo/tree/sha1"


# ---------------------------------------------------------------------------
# UsageSummary
# ---------------------------------------------------------------------------


def test_usage_summary_from_stats() -> None:
    stats = {
        "prompt_tokens": 500,
        "completion_tokens": 200,
        "cached_tokens": 50,
        "cache_creation_tokens": 10,
        "cost": 0.05,
        "total_llm_calls": 3,
    }
    usage = UsageSummary.from_summary_stats(stats)
    assert usage.prompt_tokens == 500
    assert usage.completion_tokens == 200
    assert usage.total_tokens == 700
    assert usage.total_cost_usd == 0.05
    assert usage.n_llm_calls == 3


def test_usage_summary_from_none() -> None:
    usage = UsageSummary.from_summary_stats(None)
    assert usage.prompt_tokens == 0
    assert usage.total_cost_usd == 0.0


def test_usage_summary_total_tokens_is_sum() -> None:
    stats = {"prompt_tokens": 100, "completion_tokens": 40}
    usage = UsageSummary.from_summary_stats(stats)
    assert usage.total_tokens == 140


# ---------------------------------------------------------------------------
# AgentInfo
# ---------------------------------------------------------------------------


def test_agent_info_from_agent_config_basic(mock_agent_config) -> None:
    info = AgentInfo.from_agent_config(mock_agent_config)
    assert len(info.agent_id) == 64  # SHA-256 hex
    assert "MockAgentConfig" in info.config_type
    assert isinstance(info.config, dict)
    assert isinstance(info.dependency_versions, dict)
    assert isinstance(info.framework_version, str)


def test_agent_info_from_agent_config_with_schemas(mock_agent_config) -> None:
    schemas = [{"type": "function", "function": {"name": "click"}}]
    info = AgentInfo.from_agent_config(mock_agent_config, action_schemas=schemas)
    assert info.tools == schemas
    assert info.tool_names == ["click"]


def test_agent_info_agent_id_is_stable(mock_agent_config) -> None:
    info1 = AgentInfo.from_agent_config(mock_agent_config)
    info2 = AgentInfo.from_agent_config(mock_agent_config)
    assert info1.agent_id == info2.agent_id


def test_agent_info_with_action_schemas_returns_copy(mock_agent_config) -> None:
    info = AgentInfo.from_agent_config(mock_agent_config)
    schemas = [{"type": "function", "function": {"name": "scroll"}}]
    enriched = info.with_action_schemas(schemas)
    assert enriched.tool_names == ["scroll"]
    assert info.tool_names == []  # original unchanged


def test_agent_info_llm_model_extracted() -> None:
    """AgentConfig with llm_config sub-dict gets llm_model populated."""
    from cube_harness.agent import AgentConfig

    class LLMAgentConfig(AgentConfig):
        llm_config: dict = {"model_name": "gpt-4o"}

        def make(self, action_set=None, **kwargs):  # type: ignore[override]
            raise NotImplementedError

    cfg = LLMAgentConfig()
    info = AgentInfo.from_agent_config(cfg)
    assert info.llm_model == "gpt-4o"


# ---------------------------------------------------------------------------
# TaskInfo
# ---------------------------------------------------------------------------


def test_task_info_from_trajectory_minimal() -> None:
    traj = _trajectory(task_id="task-42")
    info = TaskInfo.from_trajectory_and_metadata(traj)
    assert info.task_id == "task-42"
    assert info.benchmark_name == "unknown"
    assert info.first_observation_text == "Task: do it"


def test_task_info_with_benchmark_metadata() -> None:
    from cube.benchmark import BenchmarkMetadata

    bm = BenchmarkMetadata(name="MiniWoB", version="1.0", description="Mini browser tasks", authors=["A"], tags=["browser"])
    traj = _trajectory(task_id="click-dialog")
    info = TaskInfo.from_trajectory_and_metadata(traj, benchmark_metadata=bm)
    assert info.benchmark_name == "MiniWoB"
    assert info.benchmark_id == "miniwob"
    assert info.benchmark_version == "1.0"
    assert "A" in info.benchmark_authors


def test_task_info_with_task_metadata() -> None:
    from cube.task import TaskMetadata

    tm = TaskMetadata(
        id="click-dialog",
        split="test",
        abstract_description="Click a dialog button",
        recommended_max_steps=10,
        extra_info={"difficulty": "easy"},
    )
    traj = _trajectory(task_id="click-dialog")
    info = TaskInfo.from_trajectory_and_metadata(traj, task_metadata=tm)
    assert info.split == "test"
    assert info.abstract_description == "Click a dialog button"
    assert info.recommended_max_steps == 10
    assert info.extra_info["difficulty"] == "easy"


def test_task_info_task_version_hash_from_task_config(mock_tool_config) -> None:
    from cube.task import TaskConfig, TaskMetadata

    class MockTaskConfig(TaskConfig):
        def make(self, runtime_context=None, container_backend=None):  # type: ignore[override]
            raise NotImplementedError

    tc = MockTaskConfig(task_id="t1", seed=42, tool_config=mock_tool_config)
    traj = _trajectory(task_id="t1")
    info = TaskInfo.from_trajectory_and_metadata(traj, task_config=tc)
    assert info.seed == 42
    assert info.task_version_hash is not None
    assert len(info.task_version_hash) == 64


# ---------------------------------------------------------------------------
# TaskEvalRecord
# ---------------------------------------------------------------------------


def test_task_eval_record_success(mock_agent_config) -> None:
    traj = _trajectory(reward=1.0)
    agent_info = AgentInfo.from_agent_config(mock_agent_config)
    task_info = TaskInfo.from_trajectory_and_metadata(traj)
    record = TaskEvalRecord.from_trajectory(traj, agent_info, task_info, exp_name="exp1")
    assert record.success is True
    assert record.reward == 1.0
    assert record.run_id == "exp1_t1_ep0"


def test_task_eval_record_failure(mock_agent_config) -> None:
    traj = _trajectory(reward=0.0)
    agent_info = AgentInfo.from_agent_config(mock_agent_config)
    task_info = TaskInfo.from_trajectory_and_metadata(traj)
    record = TaskEvalRecord.from_trajectory(traj, agent_info, task_info)
    assert record.success is False
    assert record.reward == 0.0


def test_task_eval_record_wall_time(mock_agent_config) -> None:
    traj = _trajectory(reward=1.0)
    agent_info = AgentInfo.from_agent_config(mock_agent_config)
    task_info = TaskInfo.from_trajectory_and_metadata(traj)
    record = TaskEvalRecord.from_trajectory(traj, agent_info, task_info)
    assert record.wall_time_s == pytest.approx(10.0)


def test_task_eval_record_n_steps(mock_agent_config) -> None:
    traj = _trajectory(reward=1.0, n_agent_steps=3)
    agent_info = AgentInfo.from_agent_config(mock_agent_config)
    task_info = TaskInfo.from_trajectory_and_metadata(traj)
    record = TaskEvalRecord.from_trajectory(traj, agent_info, task_info)
    assert record.n_steps == len(traj.steps)
    assert record.n_agent_steps == 3


# ---------------------------------------------------------------------------
# EvalLog JSONL round-trip
# ---------------------------------------------------------------------------


def test_eval_log_save_and_load(mock_agent_config) -> None:
    traj = _trajectory(reward=1.0)
    agent_info = AgentInfo.from_agent_config(mock_agent_config)
    task_info = TaskInfo.from_trajectory_and_metadata(traj)
    record = TaskEvalRecord.from_trajectory(traj, agent_info, task_info, exp_name="test_exp")
    log = EvalLog(records=[record])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "eval_log.jsonl"
        log.save_jsonl(path)
        loaded = EvalLog.load_jsonl(path)

    assert len(loaded.records) == 1
    assert loaded.records[0].trajectory_id == record.trajectory_id
    assert loaded.records[0].success == record.success
    assert loaded.records[0].reward == record.reward


def test_eval_log_jsonl_is_valid_json_per_line(mock_agent_config) -> None:
    traj = _trajectory(reward=0.5)
    agent_info = AgentInfo.from_agent_config(mock_agent_config)
    task_info = TaskInfo.from_trajectory_and_metadata(traj)
    record = TaskEvalRecord.from_trajectory(traj, agent_info, task_info)
    log = EvalLog(records=[record])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.jsonl"
        log.save_jsonl(path)
        lines = path.read_text().strip().splitlines()

    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert "task" in parsed
    assert "agent" in parsed
    assert "reward" in parsed


def test_eval_log_append_record(mock_agent_config) -> None:
    traj1 = _trajectory(reward=1.0, task_id="t1")
    traj2 = _trajectory(reward=0.0, task_id="t2")
    agent_info = AgentInfo.from_agent_config(mock_agent_config)
    task_info1 = TaskInfo.from_trajectory_and_metadata(traj1)
    task_info2 = TaskInfo.from_trajectory_and_metadata(traj2)
    rec1 = TaskEvalRecord.from_trajectory(traj1, agent_info, task_info1)
    rec2 = TaskEvalRecord.from_trajectory(traj2, agent_info, task_info2)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "streamed.jsonl"
        EvalLog.append_record(rec1, path)
        EvalLog.append_record(rec2, path)
        loaded = EvalLog.load_jsonl(path)

    assert len(loaded.records) == 2
    task_ids = {r.task.task_id for r in loaded.records}
    assert task_ids == {"t1", "t2"}
