"""Tests for cube_harness.eval_log — Atlas EvalLog system."""

import json
import tempfile
from pathlib import Path

import pytest
from cube.core import Content, EnvironmentOutput, Observation, StepError

from cube_harness.core import AgentOutput, Trajectory, TrajectoryStep
from cube_harness.eval_log import (
    AgentInfo,
    BenchmarkSubset,
    EpisodeRecord,
    EvalLog,
    ExperimentRecord,
    JudgeConfig,
    JudgeOutput,
    UsageSummary,
    Verifier,
    _extract_error_type,
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


def test_agent_info_agent_id_is_stable(mock_agent_config) -> None:
    info1 = AgentInfo.from_agent_config(mock_agent_config)
    info2 = AgentInfo.from_agent_config(mock_agent_config)
    assert info1.agent_id == info2.agent_id


def test_agent_info_has_no_tools_field(mock_agent_config) -> None:
    info = AgentInfo.from_agent_config(mock_agent_config)
    assert not hasattr(info, "tools")
    assert not hasattr(info, "tool_names")


def test_agent_info_llm_model_extracted() -> None:
    from cube_harness.agent import AgentConfig

    class LLMAgentConfig(AgentConfig):
        llm_config: dict = {"model_name": "gpt-4o"}

        def make(self, action_set=None, **kwargs):  # type: ignore[override]
            raise NotImplementedError

    cfg = LLMAgentConfig()
    info = AgentInfo.from_agent_config(cfg)
    assert info.llm_model == "gpt-4o"


# ---------------------------------------------------------------------------
# BenchmarkSubset
# ---------------------------------------------------------------------------


def test_benchmark_subset_from_benchmark(mock_cube_benchmark) -> None:
    subset = BenchmarkSubset.from_benchmark(mock_cube_benchmark)
    assert subset.name == "mock-cube"
    assert subset.n_tasks == 2
    assert subset.filter is None


def test_benchmark_subset_unknown_benchmark() -> None:
    subset = BenchmarkSubset.from_benchmark(object())
    assert subset.name == "unknown"
    assert subset.n_tasks == 0


# ---------------------------------------------------------------------------
# ExperimentRecord
# ---------------------------------------------------------------------------


def test_experiment_record_experiment_id_is_stable(mock_agent_config, mock_cube_benchmark, tmp_dir) -> None:
    rec1 = ExperimentRecord.from_experiment("my_exp", tmp_dir, mock_agent_config, mock_cube_benchmark)
    rec2 = ExperimentRecord.from_experiment("my_exp", tmp_dir, mock_agent_config, mock_cube_benchmark)
    assert rec1.experiment_id == rec2.experiment_id
    assert len(rec1.experiment_id) == 16


def test_experiment_record_fields(mock_agent_config, mock_cube_benchmark, tmp_dir) -> None:
    rec = ExperimentRecord.from_experiment("test_exp", tmp_dir, mock_agent_config, mock_cube_benchmark)
    assert rec.experiment_name == "test_exp"
    assert rec.benchmark_name == "mock-cube"
    assert rec.benchmark_version == "0.1.0"
    assert rec.benchmark_subset.n_tasks == 2
    assert rec.judge_config is None


def test_experiment_record_roundtrip(mock_agent_config, mock_cube_benchmark, tmp_dir) -> None:
    rec = ExperimentRecord.from_experiment("roundtrip_exp", tmp_dir, mock_agent_config, mock_cube_benchmark)
    serialized = rec.model_dump_json()
    restored = ExperimentRecord.model_validate_json(serialized)
    assert restored.experiment_id == rec.experiment_id
    assert restored.benchmark_name == rec.benchmark_name


# ---------------------------------------------------------------------------
# EpisodeRecord
# ---------------------------------------------------------------------------


def test_episode_record_success() -> None:
    traj = _trajectory(reward=1.0)
    record = EpisodeRecord.from_trajectory(traj, experiment_id="abc123")
    assert record.success is True
    assert record.reward == 1.0
    assert record.trajectory_id == "t1_ep0"


def test_episode_record_failure() -> None:
    traj = _trajectory(reward=0.0)
    record = EpisodeRecord.from_trajectory(traj, experiment_id="abc123")
    assert record.success is False
    assert record.reward == 0.0


def test_episode_record_wall_time() -> None:
    traj = _trajectory(reward=1.0)
    record = EpisodeRecord.from_trajectory(traj, experiment_id="abc123")
    assert record.wall_time_s == pytest.approx(10.0)


def test_episode_record_n_steps() -> None:
    traj = _trajectory(reward=1.0, n_agent_steps=3)
    record = EpisodeRecord.from_trajectory(traj, experiment_id="abc123")
    assert record.n_steps == len(traj.steps)
    assert record.n_agent_steps == 3


def test_episode_record_tool_names_from_metadata() -> None:
    traj = _trajectory(reward=1.0)
    traj.metadata["action_schemas"] = [{"type": "function", "function": {"name": "click"}}]
    record = EpisodeRecord.from_trajectory(traj, experiment_id="abc123")
    assert record.tool_names == ["click"]


def test_episode_record_tool_names_empty_without_metadata() -> None:
    traj = _trajectory(reward=1.0)
    record = EpisodeRecord.from_trajectory(traj, experiment_id="abc123")
    assert record.tool_names == []


def test_episode_record_with_task_metadata() -> None:
    from cube.task import TaskMetadata

    traj = _trajectory(task_id="click-dialog")
    tm = TaskMetadata(id="click-dialog", split="test", abstract_description="Click a dialog button")
    record = EpisodeRecord.from_trajectory(traj, experiment_id="abc123", task_metadata=tm)
    assert record.split == "test"
    assert record.task_description == "Click a dialog button"


def test_episode_record_with_task_config(mock_tool_config) -> None:
    from cube.task import TaskConfig

    class MockTaskConfig(TaskConfig):
        def make(self, runtime_context=None, container_backend=None):  # type: ignore[override]
            raise NotImplementedError

    traj = _trajectory(task_id="t1")
    tc = MockTaskConfig(task_id="t1", seed=42, tool_config=mock_tool_config)
    record = EpisodeRecord.from_trajectory(traj, experiment_id="abc123", task_config=tc)
    assert record.seed == 42
    assert record.task_version_hash is not None
    assert len(record.task_version_hash) == 64


def test_episode_record_judge_output_optional() -> None:
    traj = _trajectory(reward=1.0)
    record = EpisodeRecord.from_trajectory(traj, experiment_id="abc123")
    assert record.judge_output is None
    assert record.verifier is None


def test_episode_record_with_judge_output() -> None:
    traj = _trajectory(reward=0.0)
    record = EpisodeRecord.from_trajectory(traj, experiment_id="abc123")
    record = record.model_copy(
        update={
            "judge_output": JudgeOutput(
                difficulty="hard",
                feasible=True,
                failure_root_cause="Agent did not find the submit button",
            )
        }
    )
    assert record.judge_output.difficulty == "hard"
    assert record.judge_output.feasible is True


# ---------------------------------------------------------------------------
# EvalLog: two-level round-trip
# ---------------------------------------------------------------------------


def test_eval_log_save_and_load(mock_agent_config, mock_cube_benchmark, tmp_dir) -> None:
    traj = _trajectory(reward=1.0, task_id="task-a")
    exp_rec = ExperimentRecord.from_experiment("test_exp", tmp_dir, mock_agent_config, mock_cube_benchmark)
    ep_rec = EpisodeRecord.from_trajectory(traj, experiment_id=exp_rec.experiment_id)
    log = EvalLog(experiment=exp_rec, episodes=[ep_rec])

    with tempfile.TemporaryDirectory() as out:
        out_dir = Path(out)
        log.save(out_dir)
        assert (out_dir / "experiment_record.json").exists()
        assert (out_dir / "eval_log.jsonl").exists()
        loaded = EvalLog.load(out_dir)

    assert loaded.experiment.experiment_id == exp_rec.experiment_id
    assert len(loaded.episodes) == 1
    assert loaded.episodes[0].trajectory_id == "task-a_ep0"


def test_eval_log_episode_jsonl_is_valid_json_per_line(mock_agent_config, mock_cube_benchmark, tmp_dir) -> None:
    traj = _trajectory(reward=0.5)
    exp_rec = ExperimentRecord.from_experiment("test_exp", tmp_dir, mock_agent_config, mock_cube_benchmark)
    ep_rec = EpisodeRecord.from_trajectory(traj, experiment_id=exp_rec.experiment_id)
    log = EvalLog(experiment=exp_rec, episodes=[ep_rec])

    with tempfile.TemporaryDirectory() as out:
        log.save(Path(out))
        lines = (Path(out) / "eval_log.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])

    assert "experiment_id" in parsed
    assert "task_id" in parsed
    assert "reward" in parsed


def test_eval_log_experiment_record_is_valid_json(mock_agent_config, mock_cube_benchmark, tmp_dir) -> None:
    exp_rec = ExperimentRecord.from_experiment("test_exp", tmp_dir, mock_agent_config, mock_cube_benchmark)
    log = EvalLog(experiment=exp_rec, episodes=[])

    with tempfile.TemporaryDirectory() as out:
        log.save(Path(out))
        parsed = json.loads((Path(out) / "experiment_record.json").read_text())

    assert "experiment_id" in parsed
    assert "agent" in parsed
    assert "benchmark_subset" in parsed


def test_eval_log_append_episode(mock_agent_config, mock_cube_benchmark, tmp_dir) -> None:
    traj1 = _trajectory(reward=1.0, task_id="t1")
    traj2 = _trajectory(reward=0.0, task_id="t2")
    exp_rec = ExperimentRecord.from_experiment("test_exp", tmp_dir, mock_agent_config, mock_cube_benchmark)
    rec1 = EpisodeRecord.from_trajectory(traj1, experiment_id=exp_rec.experiment_id)
    rec2 = EpisodeRecord.from_trajectory(traj2, experiment_id=exp_rec.experiment_id)

    with tempfile.TemporaryDirectory() as out:
        out_path = Path(out) / "eval_log.jsonl"
        EvalLog.append_episode(rec1, out_path)
        EvalLog.append_episode(rec2, out_path)
        lines = out_path.read_text().strip().splitlines()

    assert len(lines) == 2
    task_ids = {json.loads(line)["task_id"] for line in lines}
    assert task_ids == {"t1", "t2"}


def test_eval_log_experiment_id_fk_consistent(mock_agent_config, mock_cube_benchmark, tmp_dir) -> None:
    """EpisodeRecords carry the same experiment_id as ExperimentRecord."""
    exp_rec = ExperimentRecord.from_experiment("fk_test", tmp_dir, mock_agent_config, mock_cube_benchmark)
    traj = _trajectory(reward=1.0)
    ep_rec = EpisodeRecord.from_trajectory(traj, experiment_id=exp_rec.experiment_id)
    assert ep_rec.experiment_id == exp_rec.experiment_id


# ---------------------------------------------------------------------------
# Optional models
# ---------------------------------------------------------------------------


def test_judge_config_roundtrip() -> None:
    cfg = JudgeConfig(model="claude-opus-4-7", prompt_version="v1.2", judged_at="2026-04-28T12:00:00Z")
    restored = JudgeConfig.model_validate_json(cfg.model_dump_json())
    assert restored.model == "claude-opus-4-7"
    assert restored.judged_at == "2026-04-28T12:00:00Z"


def test_verifier_roundtrip() -> None:
    v = Verifier(ref="https://github.com/org/repo/blob/abc123/eval.py", source="def evaluate(): return 1.0")
    restored = Verifier.model_validate_json(v.model_dump_json())
    assert "abc123" in restored.ref


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_export_eval_log_integration(tmp_dir, mock_agent_config, mock_cube_benchmark) -> None:
    """Full experiment run followed by export_eval_log produces correct two-level output."""
    from cube_harness.exp_runner import run_sequentially
    from cube_harness.experiment import Experiment

    exp = Experiment(
        name="integration_test",
        output_dir=tmp_dir,
        agent_config=mock_agent_config,
        benchmark=mock_cube_benchmark,
    )
    run_sequentially(exp)

    eval_log = exp.export_eval_log(tmp_dir)

    exp_record_path = tmp_dir / "experiment_record.json"
    eval_log_path = tmp_dir / "eval_log.jsonl"
    assert exp_record_path.exists(), "experiment_record.json was not created"
    assert eval_log_path.exists(), "eval_log.jsonl was not created"

    exp_data = json.loads(exp_record_path.read_text())
    assert exp_data["experiment_name"] == "integration_test"
    assert "agent" in exp_data
    assert exp_data["benchmark_subset"]["n_tasks"] == 2

    lines = eval_log_path.read_text().strip().splitlines()
    assert len(lines) == 2, f"Expected 2 episode records, got {len(lines)}"

    experiment_id = eval_log.experiment.experiment_id
    for line in lines:
        episode = json.loads(line)
        assert episode["experiment_id"] == experiment_id
        assert episode["reward"] == pytest.approx(1.0)
        assert episode["success"] is True

    loaded = EvalLog.load(tmp_dir)
    assert loaded.experiment.experiment_id == experiment_id
    assert len(loaded.episodes) == 2
