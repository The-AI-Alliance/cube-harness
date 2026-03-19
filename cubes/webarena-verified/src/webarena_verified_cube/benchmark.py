import logging
import urllib.error
import urllib.request
from typing import ClassVar, Generator

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata
from webarena_verified.api.webarena_verified import WebArenaVerified
from webarena_verified.types.agent_response import MainObjectiveType
from webarena_verified.types.config import WebArenaVerifiedConfig
from webarena_verified.types.task import WebArenaSite

from cube_harness.tools.toolbox import ToolboxConfig

from webarena_verified_cube.task import WebArenaVerifiedTaskConfig
from webarena_verified_cube.tool import HarPlaywrightConfig, SubmitResponseConfig

logger = logging.getLogger(__name__)


def _load_task_metadata() -> dict[str, TaskMetadata]:
    wav = WebArenaVerified()
    return {
        str(t.task_id): TaskMetadata(
            id=str(t.task_id),
            abstract_description=t.intent,
            recommended_max_steps=30,
            extra_info={
                "sites": [s.value for s in t.sites],
                "expected_action": t.expected_action,
                "intent_template_id": t.intent_template_id,
            },
        )
        for t in wav.get_tasks()
    }


class WebArenaVerifiedBenchmark(Benchmark):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="webarena-verified-cube",
        version="1.0.0",
        description="WebArena-Verified benchmark — 812 verified web automation tasks across 6 platforms",
        num_tasks=812,
        tags=["browser", "web", "ui", "webarena"],
    )
    # Populated lazily on first use to avoid side-effects at import time.
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {}
    task_config_class: ClassVar[type[TaskConfig]] = WebArenaVerifiedTaskConfig
    _task_metadata_loaded: ClassVar[bool] = False

    @classmethod
    def _ensure_task_metadata_loaded(cls) -> None:
        if not cls._task_metadata_loaded:
            cls.task_metadata = _load_task_metadata()
            cls._task_metadata_loaded = True

    wav_config: WebArenaVerifiedConfig
    sites_filter: list[WebArenaSite] | None = None
    action_filter: MainObjectiveType | None = None
    task_ids_filter: list[int] | None = None

    def _setup(self) -> None:
        if self.wav_config.environments is None:
            return
        for site, env_config in self.wav_config.environments.items():
            url = env_config.active_url
            if url is None:
                continue
            try:
                urllib.request.urlopen(url, timeout=5)
            except urllib.error.URLError as e:
                raise RuntimeError(
                    f"Cannot reach {site} at {url}. "
                    f"Start the Docker container with: webarena-verified env start {site.value}"
                ) from e

    def close(self) -> None:
        pass

    def get_task_configs(self) -> Generator[WebArenaVerifiedTaskConfig, None, None]:
        self._ensure_task_metadata_loaded()
        wav = WebArenaVerified(config=self.wav_config)
        tasks = wav.get_tasks(sites=self.sites_filter, action=self.action_filter)
        if self.task_ids_filter is not None:
            task_ids_set = set(self.task_ids_filter)
            tasks = [t for t in tasks if t.task_id in task_ids_set]
        for t in tasks:
            task_id_str = str(t.task_id)
            task_meta = self.task_metadata[task_id_str]
            yield WebArenaVerifiedTaskConfig(
                task_id=task_id_str,
                task_metadata=task_meta,
                tool_config=self.default_tool_config
                or ToolboxConfig(tool_configs=[HarPlaywrightConfig(), SubmitResponseConfig()]),
                wav_task=t,
                wav_config=self.wav_config,
            )
