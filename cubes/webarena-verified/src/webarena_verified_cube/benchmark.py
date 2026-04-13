import logging
import urllib.error
import urllib.request
from typing import Any, ClassVar, Generator

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata
from webarena_verified.api.webarena_verified import WebArenaVerified
from webarena_verified.types.agent_response import MainObjectiveType
from webarena_verified.types.config import WebArenaVerifiedConfig
from webarena_verified.types.task import WebArenaSite

from cube.tool import ToolboxConfig

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


_TASK_METADATA_UNLOADED: dict[str, TaskMetadata] = {}


class WebArenaVerifiedBenchmark(Benchmark):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="webarena-verified-cube",
        version="1.0.0",
        description="WebArena-Verified benchmark — 812 verified web automation tasks across 6 platforms",
        num_tasks=812,
        tags=["browser", "web", "ui", "webarena"],
    )
    task_metadata: ClassVar[dict[str, TaskMetadata]] = _TASK_METADATA_UNLOADED
    task_config_class: ClassVar[type[TaskConfig]] = WebArenaVerifiedTaskConfig

    def model_post_init(self, __context: Any) -> None:
        if type(self).task_metadata is _TASK_METADATA_UNLOADED:
            type(self).task_metadata = _load_task_metadata()

    default_tool_config: ToolboxConfig = ToolboxConfig(tool_configs=[HarPlaywrightConfig(), SubmitResponseConfig()])  # type: ignore

    wav_config: WebArenaVerifiedConfig
    sites_filter: list[WebArenaSite] | None = None
    action_filter: MainObjectiveType | None = None
    task_ids_filter: list[int] | None = None

    def _setup(self) -> None:
        """
        Verify that every configured environment URL is reachable before running tasks.

        Raises RuntimeError if any site cannot be reached, with a hint to start
        the corresponding Docker container.
        """
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
                    f"Start the Docker container with: `webarena-verified env start --site {site.value}`"
                ) from e

    def close(self) -> None:
        pass

    def get_task_configs(self) -> Generator[WebArenaVerifiedTaskConfig, None, None]:
        """Yield task configs, applying any active site, action, or task-ID filters.

        Overrides the base class because each config must carry ``wav_task`` and
        ``wav_config`` (required by the WebArena tools), and because filtering by
        ``sites_filter``, ``action_filter``, and ``task_ids_filter`` is done here
        rather than via the generic ``task_metadata`` dict.
        """
        wav = WebArenaVerified(config=self.wav_config)
        tasks = wav.get_tasks(sites=self.sites_filter, action=self.action_filter)
        if self.task_ids_filter is not None:
            task_ids_set = set(self.task_ids_filter)
            tasks = [t for t in tasks if t.task_id in task_ids_set]
        for t in tasks:
            task_id_str = str(t.task_id)
            yield WebArenaVerifiedTaskConfig(
                task_id=task_id_str,
                tool_config=self.default_tool_config,
                wav_task=t,
                wav_config=self.wav_config,
            )
