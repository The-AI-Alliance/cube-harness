import json
import logging
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from random import Random
from typing import TextIO

from datasets import load_dataset

from agentlab2.benchmark import Benchmark
from agentlab2.benchmarks.mind2web.task import Mind2WebTask

logger = logging.getLogger(__name__)


class Mind2WebBenchmark(Benchmark):
    data_dir: str | None = None
    port: int = 8001
    split: str = "train"
    episode_max_time: int = 1000000
    shuffle: bool = True
    shuffle_seed: int = 42
    max_tasks: int | None = None

    _server_process: subprocess.Popen | None = None
    _stdout_file: TextIO | None = None
    _stderr_file: TextIO | None = None
    _html_dir: Path | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/mind2web"

    def setup(self) -> None:
        self._download_data()
        self._prepare_html_files()
        self._start_server()

    def _download_data(self) -> None:
        if self.data_dir is None:
            self.data_dir = str(Path.home() / ".agentlab2" / "mind2web")

        data_path = Path(self.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        split_file = data_path / f"{self.split}.json"
        if split_file.exists() and split_file.stat().st_size > 0:
            logger.info(f"Mind2Web {self.split} data already exists at {split_file}")
            return

        logger.info(f"Downloading Mind2Web {self.split} split...")
        try:
            dataset = load_dataset("osunlp/Mind2Web", split=self.split)
            with open(split_file, "w") as f:
                json.dump([item for item in dataset], f)
            logger.info(f"Downloaded {len(dataset)} tasks to {split_file}")
        except Exception as e:
            if split_file.exists():
                split_file.unlink()
            raise RuntimeError(f"Failed to download Mind2Web data: {e}")

    def _prepare_html_files(self) -> None:
        tmp_dir = Path(tempfile.gettempdir()) / "mind2web_html"
        tmp_dir.mkdir(exist_ok=True)
        self._html_dir = tmp_dir

        # Create mind2web subdirectory to match base_url path
        mind2web_dir = tmp_dir / "mind2web"
        mind2web_dir.mkdir(exist_ok=True)

        tasks_data = self.load_task_infos()
        for task_info in tasks_data:
            for action_idx, action in enumerate(task_info["actions"]):
                html_file = mind2web_dir / f"{task_info['annotation_id']}_{action_idx}.html"
                if not html_file.exists():
                    html_content = action.get("cleaned_html", action.get("raw_html", ""))
                    html_file.write_text(html_content, encoding="utf-8")

        logger.info(f"Prepared HTML files in {mind2web_dir}")

    def _start_server(self) -> None:
        tmp_dir = Path(tempfile.gettempdir())
        self._stdout_file = open(tmp_dir / "mind2web_server_stdout.log", "w")
        self._stderr_file = open(tmp_dir / "mind2web_server_stderr.log", "w")
        self._server_process = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(self.port)],
            cwd=self._html_dir,
            stdout=self._stdout_file,
            stderr=self._stderr_file,
        )
        time.sleep(1)

        try:
            # Check root URL for server health (base_url points to subdirectory)
            root_url = f"http://localhost:{self.port}/"
            urllib.request.urlopen(root_url, timeout=5)
            logger.info(f"Mind2Web server responding at {self.base_url}")
        except Exception as e:
            self.close()
            raise RuntimeError(f"Mind2Web server failed to respond: {e}")

    def load_task_infos(self) -> list[dict]:
        data_path = Path(self.data_dir) / f"{self.split}.json"
        with open(data_path) as f:
            tasks = json.load(f)

        if self.max_tasks:
            tasks = tasks[: self.max_tasks]

        if self.shuffle:
            rng = Random(self.shuffle_seed)
            rng.shuffle(tasks)

        return tasks

    def load_tasks(self) -> list[Mind2WebTask]:
        tasks_data = self.load_task_infos()
        tasks = [
            Mind2WebTask(
                id=task_data["annotation_id"],
                task_desc=task_data["confirmed_task"],
                website=task_data["website"],
                domain=task_data["domain"],
                actions_data=task_data["actions"],
                base_url=self.base_url,
                episode_max_time=self.episode_max_time,
            )
            for task_data in tasks_data
        ]
        return tasks

    def close(self) -> None:
        if self._server_process is not None:
            logger.info("Shutting down Mind2Web server...")
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
