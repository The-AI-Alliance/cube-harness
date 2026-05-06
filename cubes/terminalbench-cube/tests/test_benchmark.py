"""Docker-free unit tests for terminalbench-cube BenchmarkConfig migration."""

from __future__ import annotations

from cube.benchmark import BenchmarkConfig
from cube.task import TaskExecutionInfo

from terminalbench_cube.benchmark import TerminalBenchBenchmarkConfig
from terminalbench_cube.debug import _TASK_ACTIONS, get_debug_benchmark
from terminalbench_cube.task import (
    TerminalBenchExecutionInfo,
    TerminalBenchTaskConfig,
    TerminalBenchTaskMetadata,
)

_DEBUG_TASK_IDS = list(_TASK_ACTIONS)


def test_config_roundtrip() -> None:
    cfg = TerminalBenchBenchmarkConfig(oracle_mode=True).subset_from_list(_DEBUG_TASK_IDS)
    js = cfg.model_dump_json()
    restored = TerminalBenchBenchmarkConfig.model_validate_json(js)
    assert restored.oracle_mode is True
    assert restored.task_ids == _DEBUG_TASK_IDS
    assert restored.num_tasks == len(_DEBUG_TASK_IDS)
    assert restored.benchmark_metadata.name == "terminalbench-cube"


def test_task_metadata_loaded() -> None:
    cfg = TerminalBenchBenchmarkConfig()
    assert cfg.benchmark_metadata.num_tasks == 89
    assert len(cfg.task_metadata) == 89
    sample = next(iter(cfg.task_metadata.values()))
    assert isinstance(sample, TerminalBenchTaskMetadata)
    assert sample.difficulty
    assert isinstance(sample.tags, list)


def test_get_task_configs_stamps_metadata() -> None:
    cfg = TerminalBenchBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    configs = list(cfg.get_task_configs())
    assert len(configs) == len(_DEBUG_TASK_IDS)
    for tc in configs:
        assert isinstance(tc, TerminalBenchTaskConfig)
        assert isinstance(tc.metadata, TerminalBenchTaskMetadata)
        assert tc.metadata.id == tc.task_id
        assert tc.metadata.category


def test_subset_from_list() -> None:
    cfg = TerminalBenchBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    assert cfg.task_ids == _DEBUG_TASK_IDS
    assert set(cfg.tasks().keys()) == set(_DEBUG_TASK_IDS)
    assert cfg.num_tasks == len(_DEBUG_TASK_IDS)


def test_debug_benchmark_type() -> None:
    cfg = get_debug_benchmark()
    assert isinstance(cfg, TerminalBenchBenchmarkConfig)
    assert isinstance(cfg, BenchmarkConfig)
    assert cfg.oracle_mode is True
    assert cfg.task_ids == _DEBUG_TASK_IDS


def test_execution_info_roundtrip() -> None:
    ei = TerminalBenchExecutionInfo(
        instruction="Follow the steps",
        archive="ZmFrZV90YXI=",
    )
    assert isinstance(ei, TaskExecutionInfo)
    restored = TerminalBenchExecutionInfo.model_validate_json(ei.model_dump_json())
    assert restored.instruction == "Follow the steps"
    assert restored.archive == "ZmFrZV90YXI="
    assert restored.max_test_timeout_sec == 900
