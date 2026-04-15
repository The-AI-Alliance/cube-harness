import logging
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

from cube import LocalInfraConfig
from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.resource import InfraConfig
from cube.task import TaskConfig
from cube.tool import ToolboxConfig
from pydantic import Field, PrivateAttr
from webarena_verified.environments.container import ContainerManager
from webarena_verified.types.config import WebArenaVerifiedConfig

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

    Sites that require large data assets before _setup() can start their containers:
        - Wikipedia: ~80 GB ZIM file, bind-mounted at /data inside the container.
        - Map: three S3 tarballs extracted into 9 named Docker volumes.
    Download them via:
        webarena-verified env setup init --site wikipedia --data-dir <site_data_dir>
        webarena-verified env setup init --site map      --data-dir <site_data_dir>
    where <site_data_dir> is the site_data_dir field (default: cache_dir()/site_data).
    _setup() raises RuntimeError with this exact command if the data is missing.

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
    infra: InfraConfig = Field(default_factory=LocalInfraConfig)
    """InfraConfig used to start and stop site containers. Defaults to LocalInfraConfig."""

    site_data_dir: Path | None = None
    """Directory for large site data files (Wikipedia ZIM, Map tarballs).
    If None, defaults to cache_dir() / "site_data".
    Set this to reuse data already downloaded to a non-default location."""

    _managers: dict = PrivateAttr(default_factory=dict)
    """Maps WebArenaSite → ContainerManager for containers started by this benchmark."""

    def _resolved_site_data_root(self) -> Path:
        return self.site_data_dir or (self.cache_dir() / "site_data")

    def _setup(self) -> None:
        """Start Docker containers for each configured site.

        Reads the host port from each site's active URL in wav_config.environments.
        The env-ctrl port comes from the container config (either the user-provided
        ContainerConfig in EnvironmentConfig.container, or the default from
        DEFAULT_CONTAINER_CONFIGS). Equivalent to running:
            webarena-verified env start --site <site> --port <port>

        For sites that require bind-mounted data (e.g. Wikipedia), raises RuntimeError
        if the per-site data directory is missing or empty, with the exact command to fix it.

        If a container is already running, logs a warning and reuses it without
        touching it. Only containers started here are stopped in close().

        Does nothing if no environments are configured.
        """
        if self.wav_config.environments is None:
            return

        site_data_root = self._resolved_site_data_root()

        for site, env_config in self.wav_config.environments.items():
            active_url = env_config.active_url
            if active_url is None:
                continue

            parsed = urlparse(active_url)
            host_port = parsed.port or 80
            hostname = parsed.hostname or "localhost"

            manager = ContainerManager(site=site, config=env_config.container, hostname=hostname)
            host_env_ctrl_port = manager.config.host_env_ctrl_port or (host_port + 1)

            per_site_data_dir = site_data_root / site.value
            if manager.config.data_dir_mount:
                if not per_site_data_dir.exists() or not any(per_site_data_dir.iterdir()):
                    raise RuntimeError(
                        f"Site {site.value!r} requires data files in {per_site_data_dir}.\n"
                        f"Download them first:\n"
                        f"  webarena-verified env setup init --site {site.value} --data-dir {per_site_data_dir}"
                    )

            if manager.is_running():
                logger.warning(f"Container for {site.value} already running at {active_url} — reusing")
                continue

            logger.info(f"Starting container for {site.value} on port {host_port}")
            manager.start(
                port=host_port,
                env_ctrl_port=host_env_ctrl_port,
                data_dir=per_site_data_dir if manager.config.data_dir_mount else None,
            )
            self._managers[site] = manager

    def close(self) -> None:
        """Stop all Docker containers started by _setup()."""
        for site, manager in list(self._managers.items()):
            logger.info(f"Stopping container for {site.value}")
            manager.stop()
        self._managers.clear()

    def get_task_configs(self) -> Generator[WebArenaVerifiedTaskConfig, None, None]:
        """Yield TaskConfigs with wav_config forwarded from benchmark settings."""
        for tm in self.task_metadata.values():
            yield WebArenaVerifiedTaskConfig(
                task_id=tm.id,
                tool_config=self.default_tool_config,
                wav_config=self.wav_config,
            )
