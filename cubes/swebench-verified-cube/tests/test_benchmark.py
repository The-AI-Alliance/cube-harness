"""Docker-free unit tests for swebench-verified-cube — covers the BenchmarkConfig
contract (registry wiring, subsetting, metadata stamping, debug factory,
serialization round-trip) and tool-layer helpers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from cube.benchmark import BenchmarkConfig
from cube.container import ExecResult
from cube.task import TaskExecutionInfo

from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmarkConfig
from swebench_verified_cube.debug import _TASK_ACTIONS, get_debug_benchmark
from swebench_verified_cube.task import (
    SWEBenchVerifiedExecutionInfo,
    SWEBenchVerifiedTaskConfig,
    SWEBenchVerifiedTaskMetadata,
)
from swebench_verified_cube.tool import BashOnlySWEBenchTool, BashOnlySWEBenchToolConfig


_DEBUG_TASK_IDS = list(_TASK_ACTIONS)


def test_config_roundtrip():
    """``model_dump_json`` → ``model_validate_json`` produces an equivalent config."""
    cfg = SWEBenchVerifiedBenchmarkConfig(include_hints=True, oracle_mode=True).subset_from_list(_DEBUG_TASK_IDS)
    js = cfg.model_dump_json()
    restored = SWEBenchVerifiedBenchmarkConfig.model_validate_json(js)
    assert restored.include_hints is True
    assert restored.oracle_mode is True
    assert restored.task_ids == _DEBUG_TASK_IDS
    assert restored.num_tasks == len(_DEBUG_TASK_IDS)
    assert restored.benchmark_metadata.name == "swebench-verified-cube"


def test_task_metadata_loaded():
    """``task_metadata`` ClassVar is auto-loaded from task_metadata.json with 500 entries."""
    cfg = SWEBenchVerifiedBenchmarkConfig()
    assert cfg.benchmark_metadata.num_tasks == 500
    assert len(cfg.task_metadata) == 500
    sample = next(iter(cfg.task_metadata.values()))
    assert isinstance(sample, SWEBenchVerifiedTaskMetadata)
    # SWE-bench Verified specific fields are present
    assert sample.repo
    assert sample.base_commit


def test_get_task_configs_stamps_metadata():
    """Every emitted ``TaskConfig`` carries the full ``TaskMetadata`` (no task_id-only stub)."""
    cfg = SWEBenchVerifiedBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    configs = list(cfg.get_task_configs())
    assert len(configs) == len(_DEBUG_TASK_IDS)
    for tc in configs:
        assert isinstance(tc, SWEBenchVerifiedTaskConfig)
        assert isinstance(tc.metadata, SWEBenchVerifiedTaskMetadata)
        assert tc.metadata.id == tc.task_id
        # Stamped metadata carries subclass-specific fields, not just base TaskMetadata.
        assert tc.metadata.repo


def test_subset_from_list():
    """``subset_from_list`` scopes the config to exactly the requested task IDs."""
    cfg = SWEBenchVerifiedBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    assert cfg.task_ids == _DEBUG_TASK_IDS
    assert set(cfg.tasks().keys()) == set(_DEBUG_TASK_IDS)
    assert cfg.num_tasks == len(_DEBUG_TASK_IDS)


def test_debug_benchmark_type():
    """``get_debug_benchmark()`` returns a ``BenchmarkConfig`` (not a live ``Benchmark``).

    The harness owns ``config.install()`` and ``config.make(infra)``; the debug factory
    must not call either.
    """
    cfg = get_debug_benchmark()
    assert isinstance(cfg, SWEBenchVerifiedBenchmarkConfig)
    assert isinstance(cfg, BenchmarkConfig)
    assert cfg.oracle_mode is True
    # Scoped to the debug task subset
    assert cfg.task_ids == _DEBUG_TASK_IDS


def test_execution_info_roundtrip():
    """Typed ``SWEBenchVerifiedExecutionInfo`` round-trips through the TaskExecutionInfo discriminator."""
    ei = SWEBenchVerifiedExecutionInfo(
        problem_statement="test issue",
        hints_text="useful hint",
        patch="diff --git a/x b/x",
        test_patch="diff --git a/y b/y",
        fail_to_pass=["test_a", "test_b"],
        pass_to_pass=["test_c"],
    )
    assert isinstance(ei, TaskExecutionInfo)
    restored = SWEBenchVerifiedExecutionInfo.model_validate_json(ei.model_dump_json())
    assert restored.problem_statement == "test issue"
    assert restored.fail_to_pass == ["test_a", "test_b"]
    assert restored.eval_timeout == 1800  # default preserved



def test_bash_only_tool_output_truncation() -> None:
    """bash() head+tail truncates large output without raising."""
    container = MagicMock()
    big_output = "x" * 200_000
    container.exec.return_value = ExecResult(stdout=big_output, stderr="", exit_code=0)

    config = BashOnlySWEBenchToolConfig(max_output_bytes=100_000)
    tool = BashOnlySWEBenchTool(config=config, container=container)

    result = tool.bash(command="echo hi")
    assert "[... " in result
    assert "bytes elided" in result
    assert len(result.encode()) < 200_000


def test_bash_only_tool_no_output() -> None:
    """bash() returns '(no output)' when stdout and stderr are both empty."""
    container = MagicMock()
    container.exec.return_value = ExecResult(stdout="", stderr="", exit_code=0)

    tool = BashOnlySWEBenchTool(config=BashOnlySWEBenchToolConfig(), container=container)
    assert tool.bash(command="true") == "(no output)"
