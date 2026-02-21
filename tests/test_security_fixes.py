"""Security regression tests for benchmark and tool hardening changes."""

import base64
import pickle
import subprocess
import zlib
from pathlib import Path

import pytest

from agentlab2.benchmarks.livecodebench.task import LiveCodeBenchTask
from agentlab2.benchmarks.terminalbench.benchmark import TerminalBenchBenchmark
from agentlab2.benchmarks.terminalbench.task import TerminalBenchTask
from agentlab2.tools.daytona import _truncate_output as daytona_truncate_output
from agentlab2.tools.docker import DockerSWEToolConfig
from agentlab2.tools.docker import _truncate_output as docker_truncate_output


class RecordingTool:
    """Simple tool stub that records shell commands for assertions."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.writes: list[tuple[str, str]] = []

    def bash(self, command: str, timeout: int = 120) -> str:
        self.commands.append(command)
        return ""

    def write_file(self, path: str, content: str) -> None:
        self.writes.append((path, content))


def make_terminalbench_task() -> TerminalBenchTask:
    """Create a minimal TerminalBenchTask for helper-method unit tests."""
    return TerminalBenchTask(
        id="task_1",
        instruction="Do the task",
        archive=b"",
        difficulty="easy",
        category="general",
        tags=[],
        docker_image="python:3.13",
        cpus=1,
        memory="1G",
        storage="1G",
        max_agent_timeout_sec=60,
        max_test_timeout_sec=60,
    )


def make_livecodebench_task() -> LiveCodeBenchTask:
    """Create a minimal LiveCodeBench task for parser tests."""
    return LiveCodeBenchTask(
        id="lcb_1",
        question_title="Title",
        question_content="Question",
        platform="platform",
        difficulty="easy",
        starter_code="",
        public_test_cases="",
        private_test_cases="",
        metadata="{}",
    )


def test_terminalbench_upload_directory_quotes_remote_paths(tmp_path: Path) -> None:
    """Remote paths must be shell-quoted in mkdir commands."""
    task = make_terminalbench_task()
    tool = RecordingTool()
    task._tool = tool  # type: ignore[assignment]

    local_dir = tmp_path / "local"
    local_dir.mkdir()
    (local_dir / "a b.txt").write_text("content", encoding="utf-8")

    remote_dir = "/app/dir with spaces; touch /tmp/pwned"
    task._upload_directory(local_dir, remote_dir)

    assert tool.commands[0] == "mkdir -p '/app/dir with spaces; touch /tmp/pwned'"
    assert tool.commands[1] == "mkdir -p '/app/dir with spaces; touch /tmp/pwned'"


def test_terminalbench_upload_binary_file_quotes_remote_path(tmp_path: Path) -> None:
    """Binary upload command must quote destination path and avoid raw echo."""
    task = make_terminalbench_task()
    tool = RecordingTool()
    task._tool = tool  # type: ignore[assignment]

    local_path = tmp_path / "blob.bin"
    local_path.write_bytes(b"\x00\x01\x02")
    remote_path = "/tmp/out; rm -rf /"
    task._upload_binary_file(local_path, remote_path)

    command = tool.commands[0]
    assert command.startswith("printf %s ")
    assert command.endswith("> '/tmp/out; rm -rf /'")


def test_livecodebench_parser_rejects_pickled_encoded_payload() -> None:
    """Encoded pickle payloads should be rejected instead of deserialized."""
    task = make_livecodebench_task()
    payload = base64.b64encode(zlib.compress(pickle.dumps('[{"input":"1","output":"1"}]'))).decode("ascii")

    assert task._parse_test_cases(payload) == []


def test_livecodebench_parser_accepts_json_payload() -> None:
    """Normal JSON payloads should still parse."""
    task = make_livecodebench_task()
    payload = '[{"input":"1","output":"1"}]'

    assert task._parse_test_cases(payload) == [{"input": "1", "output": "1"}]


def test_docker_config_has_hardened_defaults() -> None:
    """Docker tool should default to isolated, non-root configuration."""
    config = DockerSWEToolConfig()
    assert config.network_mode == "none"
    assert config.user == "1000:1000"
    assert config.enforce_disk_quota is True
    assert config.writable_tmpfs_dirs == ["/tests", "/logs", "/workspace", "/solution"]
    assert config.max_output_bytes == 100_000


def test_docker_output_truncation() -> None:
    """Docker output truncation should append a truncation marker."""
    output = "a" * 20
    truncated = docker_truncate_output(output, max_output_bytes=10)
    assert truncated.startswith("a" * 10)
    assert "[truncated at 10 bytes]" in truncated


def test_daytona_output_truncation() -> None:
    """Daytona output truncation should append a truncation marker."""
    output = "b" * 20
    truncated = daytona_truncate_output(output, max_output_bytes=10)
    assert truncated.startswith("b" * 10)
    assert "[truncated at 10 bytes]" in truncated


def test_terminalbench_install_sets_clone_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Install should enforce a clone timeout to avoid hanging subprocesses."""
    captured_timeout: dict[str, int] = {}

    def fake_run(command: list[str], check: bool, timeout: int) -> subprocess.CompletedProcess[str]:
        captured_timeout["value"] = timeout
        repo_dir = Path(command[-1])
        repo_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(command, 0)

    class FakeDataset:
        """Dataset stub to avoid touching real Arrow serialization in unit test."""

        @staticmethod
        def from_list(tasks: list[dict]) -> "FakeDataset":
            _ = tasks
            return FakeDataset()

        def save_to_disk(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("agentlab2.benchmarks.terminalbench.benchmark.subprocess.run", fake_run)
    monkeypatch.setattr("agentlab2.benchmarks.terminalbench.benchmark.Dataset", FakeDataset)

    benchmark = TerminalBenchBenchmark(tool_config=DockerSWEToolConfig(), dataset_path=str(tmp_path / "tb2_dataset"))
    benchmark.install()

    assert captured_timeout["value"] == 300
