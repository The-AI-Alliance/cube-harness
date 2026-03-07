"""Smoke tests for terminalbench-cube."""

from terminalbench_cube.benchmark import TerminalBenchBenchmark


def test_benchmark_metadata() -> None:
    meta = TerminalBenchBenchmark.benchmark_metadata
    assert meta.name == "terminalbench-cube"
    assert meta.version
    assert "TODO" not in meta.description


def test_task_config_class() -> None:
    from terminalbench_cube.task import TerminalBenchTaskConfig
    assert TerminalBenchBenchmark.task_config_class is TerminalBenchTaskConfig
