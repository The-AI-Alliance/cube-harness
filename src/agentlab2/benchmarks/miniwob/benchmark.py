import json
import logging
import random
import subprocess
import sys
import tempfile
import time
import urllib.request
from importlib.resources import files
from pathlib import Path
from random import shuffle
from typing import TextIO

from agentlab2.benchmarks.miniwob.task import MiniWobTask
from agentlab2.legacy import Benchmark

logger = logging.getLogger(__name__)


class MiniWobBenchmark(Benchmark):
    html_path: str = files("miniwob").joinpath("html").as_posix()  # type: ignore
    port: int = 8000
    remove_human_display: bool = True
    episode_max_time: int = 1000000
    shuffle: bool = True
    shuffle_seed: int = 42

    # Runtime state (not serialized)
    _server_process: subprocess.Popen | None = None
    _stdout_file: TextIO | None = None
    _stderr_file: TextIO | None = None

    model_config = {"arbitrary_types_allowed": True}

    def load_task_infos(self) -> list[dict]:
        _tasks_file = Path(__file__).parent / "miniwob_tasks.json"
        with open(_tasks_file) as f:
            task_infos = json.load(f)
        return task_infos

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/miniwob"

    def setup(self):
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
        time.sleep(1)

        # Check if the process started successfully
        if self._server_process.poll() is not None:
            self._stderr_file.flush()
            stderr_path = Path(tempfile.gettempdir()) / "miniwob_server_stderr.log"
            stderr_content = stderr_path.read_text() if stderr_path.exists() else "No stderr available"
            self.close()
            raise RuntimeError(
                f"MiniWob server failed to start (exit code {self._server_process.returncode}): {stderr_content}"
            )

        # Check if the server is running by attempting to connect
        try:
            urllib.request.urlopen(self.base_url, timeout=5)
            logger.info(f"MiniWob server responding at {self.base_url}")
        except Exception as e:
            self.close()
            raise RuntimeError(f"MiniWob server failed to respond {self.base_url}: {e}")

    def load_tasks(self) -> list[MiniWobTask]:
        tasks = [
            MiniWobTask(
                id=task["subdomain"],
                desc=task["desc"],
                subdomain=task["subdomain"],
                base_url=self.base_url,
                remove_human_display=self.remove_human_display,
                episode_max_time=self.episode_max_time,
            )
            for task in self.load_task_infos()
        ]
        if self.shuffle:
            random.seed(self.shuffle_seed)
            shuffle(tasks)
        return tasks

    def close(self):
        """Shutdown the MiniWob server and close file handles."""
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
