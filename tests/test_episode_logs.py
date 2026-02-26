"""Tests for agentlab2.episode_logs."""

import logging
from pathlib import Path

import pytest

from agentlab2.episode_logs import get_log_path, redirect_output_to_log, trajectory_log_id


def test_trajectory_log_id() -> None:
    """Test trajectory log naming convention."""
    assert trajectory_log_id("my_task", 7) == "my_task_ep7"


def test_get_log_path(tmp_path: Path) -> None:
    """Test log path construction."""
    assert get_log_path(tmp_path, "traj_1") == tmp_path / "logs" / "traj_1.log"


def test_redirect_output_to_log_restores_logging_on_exception(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test logger stream restoration even when body raises."""
    log_file = tmp_path / "logs" / "traj_exception.log"
    test_logger = logging.getLogger("tests.episode_logs")

    with pytest.raises(RuntimeError, match="boom"):
        with redirect_output_to_log(log_file, append=False, tee=True):
            print("stdout-before-error")
            test_logger.info("inside-context")
            raise RuntimeError("boom")

    test_logger.info("outside-context")
    captured = capsys.readouterr()
    assert "outside-context" in captured.err
    assert "I/O operation on closed file" not in captured.err

    content = log_file.read_text()
    assert "stdout-before-error" in content
    assert "inside-context" in content
    assert "outside-context" not in content


def test_redirect_output_to_log_append_mode(tmp_path: Path) -> None:
    """Test append mode preserves previous log output."""
    log_file = tmp_path / "logs" / "traj_append.log"

    with redirect_output_to_log(log_file, append=False, tee=False):
        print("first-line")
    with redirect_output_to_log(log_file, append=True, tee=False):
        print("second-line")

    assert log_file.read_text() == "first-line\nsecond-line\n"
