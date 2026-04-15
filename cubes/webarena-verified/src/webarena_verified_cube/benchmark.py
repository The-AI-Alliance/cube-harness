import logging
import urllib.error
import urllib.request
from collections.abc import Generator
from typing import ClassVar, override

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.task import TaskConfig
from webarena_verified.types.config import WebArenaVerifiedConfig

from cube.tool import ToolboxConfig

from webarena_verified_cube.task import WebArenaVerifiedTaskConfig, WebArenaVerifiedTaskMetadata
from webarena_verified_cube.tool import HarPlaywrightConfig, SubmitResponseConfig

logger = logging.getLogger(__name__)


class WebArenaVerifiedBenchmark(Benchmark):
    """WebArena Verified — 812 verified web automation tasks across 6 platforms.

    task_metadata.json is a shipped package resource containing lightweight public fields
    (sites, expected_action, intent_template_id). No heavy execution data exists — all
    task information is available from the webarena-verified library at runtime.

    Filtering is done in user-land via subset_from_glob() / subset_from_list():
        bench.subset_from_glob("sites", "*shopping_admin*")
        bench.subset_from_glob("expected_action", "RETRIEVE")
        bench.subset_from_list(["0", "1", "5"])

    To regenerate task_metadata.json (developer use only), run:
        scripts/generate_task_metadata.py
    """

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="webarena-verified-cube",
        version="1.0.0",
        description="WebArena-Verified benchmark — 812 verified web automation tasks across 6 platforms",
        num_tasks=812,
        tags=["browser", "web", "ui", "webarena"],
    )
    task_metadata: ClassVar[dict[str, WebArenaVerifiedTaskMetadata]]  # type: ignore - populated automatically at import time in Benchmark.__init_subclass__
    task_config_class: ClassVar[type[TaskConfig]] = WebArenaVerifiedTaskConfig

    default_tool_config: ToolboxConfig = ToolboxConfig(tool_configs=[HarPlaywrightConfig(), SubmitResponseConfig()])  # type: ignore

    wav_config: WebArenaVerifiedConfig

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
        """Yield TaskConfigs with wav_config forwarded from benchmark settings."""
        for tm in self.task_metadata.values():
            yield WebArenaVerifiedTaskConfig(
                task_id=tm.id,
                tool_config=self.default_tool_config,
                wav_config=self.wav_config,
            )
