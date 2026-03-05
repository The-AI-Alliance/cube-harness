import json
import logging
import subprocess
import sys
import tempfile
import time
import urllib.request
from importlib.resources import files
from pathlib import Path
from typing import ClassVar, Generator

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata

from miniwob_cube.task import MiniWobTaskConfig

logger = logging.getLogger(__name__)


def _load_miniwob_task_metadata() -> dict[str, TaskMetadata]:
    tasks_file = Path(__file__).parent / "miniwob_tasks.json"
    with open(tasks_file) as f:
        task_infos = json.load(f)
    return {
        t["subdomain"]: TaskMetadata(
            id=t["subdomain"],
            abstract_description=t["desc"],
            extra_info={"nondeterministic": t.get("nondeterministic", False)},
        )
        for t in task_infos
    }


class MiniWobBenchmark(Benchmark):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="miniwob-cube",
        version="1.0.0",
        description="MiniWob++ browser automation benchmark tasks",
        num_tasks=125,
        tags=["browser", "web", "ui"],
    )
    task_metadata: ClassVar[dict[str, TaskMetadata]] = _load_miniwob_task_metadata()
    task_config_class: ClassVar[type[TaskConfig]] = MiniWobTaskConfig

    html_path: str = files("miniwob").joinpath("html").as_posix()  # type: ignore
    port: int = 8000
    remove_human_display: bool = True
    episode_max_time: int = 1000000
    server_start_timeout: float = 10.0
    server_start_poll_interval: float = 0.1

    # Runtime state (not serialized)
    _server_process: subprocess.Popen | None = None
    _stdout_file: object | None = None
    _stderr_file: object | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/miniwob"

    def _setup(self) -> None:
        tmp_dir = Path(tempfile.gettempdir())
        self._stdout_file = open(tmp_dir / "miniwob_server_stdout.log", "w")
        self._stderr_file = open(tmp_dir / "miniwob_server_stderr.log", "w")
        logger.info(f"Starting MiniWob server at port {self.port} serving from {self.html_path}...")
        self._server_process = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(self.port)],
            cwd=self.html_path,
            stdout=self._stdout_file,
            stderr=self._stderr_file,
        )
        startup_deadline = time.monotonic() + self.server_start_timeout
        last_response_error: Exception | None = None

        while time.monotonic() < startup_deadline:
            if self._server_process.poll() is not None:
                self._stderr_file.flush()
                stderr_path = Path(tempfile.gettempdir()) / "miniwob_server_stderr.log"
                stderr_content = stderr_path.read_text() if stderr_path.exists() else "No stderr available"
                returncode = self._server_process.returncode
                self.close()
                raise RuntimeError(f"MiniWob server failed to start (exit code {returncode}): {stderr_content}")

            try:
                urllib.request.urlopen(self.base_url, timeout=1)
                logger.info(f"MiniWob server responding at {self.base_url}")
                break
            except Exception as e:
                last_response_error = e
                time.sleep(self.server_start_poll_interval)
        else:
            self.close()
            raise RuntimeError(
                f"MiniWob server failed to respond at {self.base_url} within {self.server_start_timeout:.1f}s"
            ) from last_response_error

        self._runtime_context = {"base_url": self.base_url}

    def get_task_configs(self) -> Generator[MiniWobTaskConfig, None, None]:
        for tm in self.task_metadata.values():
            yield MiniWobTaskConfig(
                task_id=tm.id,
                task_metadata=tm,
                tool_config=self.default_tool_config,
                base_url=self.base_url,
                remove_human_display=self.remove_human_display,
                episode_max_time=self.episode_max_time,
            )

    def close(self) -> None:
        if self._server_process is not None:
            logger.info("Shutting down MiniWob server...")
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not terminate gracefully, killing...")
                self._server_process.kill()
            self._server_process = None

        if self._stdout_file is not None:
            self._stdout_file.close()
            self._stdout_file = None

        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None
