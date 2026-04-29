import logging
import urllib.error
import urllib.request
from collections.abc import Generator
from typing import ClassVar

from pydantic import PrivateAttr
from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.resource import DockerServiceConfig, InfraConfig, ResourceHandle
from cube.task import TaskConfig
from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
from webarena_verified.types.task import WebArenaSite

from cube.tool import ToolboxConfig

from webarena_verified_cube.task import WebArenaVerifiedTaskConfig, WebArenaVerifiedTaskMetadata
from webarena_verified_cube.tool import HarPlaywrightConfig, SubmitResponseConfig

logger = logging.getLogger(__name__)


class WebArenaVerifiedBenchmark(Benchmark):
    """WebArena Verified — 812 verified web automation tasks across 6 platforms.

    Exactly one of two setup modes must be configured before calling ``setup()``:

    **Automatic** — infra provisions and starts the Docker stack, then populates
    ``wav_config.environments`` from the live endpoints::

        WebArenaVerifiedBenchmark(
            infra=LocalInfraConfig(),
            resources=[WEBARENA_SHOPPING_ADMIN],
        )

    **Manual** — caller is responsible for starting the server; ``wav_config``
    must have ``environments`` populated with reachable URLs::

        WebArenaVerifiedBenchmark(
            wav_config=WebArenaVerifiedConfig(
                environments={WebArenaSite.SHOPPING_ADMIN: EnvironmentConfig(urls=["http://..."])}
            )
        )

    Mixing both modes (or providing neither) raises a ``ValueError`` at ``setup()`` time.

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
    """
    task_metadata.json is a shipped package resource containing lightweight public fields
    (sites, expected_action, intent_template_id). No heavy execution data exists.
    """
    task_config_class: ClassVar[type[TaskConfig]] = WebArenaVerifiedTaskConfig
    """
    TaskConfig subclass used for this benchmark.  Must be set for Benchmark to construct TaskConfigs.
    """
    default_tool_config: ToolboxConfig = ToolboxConfig(tool_configs=[HarPlaywrightConfig(), SubmitResponseConfig()])  # type: ignore
    """
    Default ToolboxConfig with tools for HAR-based environment observation and agent response submission.
    """

    wav_config: WebArenaVerifiedConfig = WebArenaVerifiedConfig()
    """WAV client configuration.

    *Automatic mode*: leave as default — ``environments`` is populated after launch.
    *Manual mode*: must have ``environments`` set with reachable URLs.
    """
    infra: InfraConfig | None = None
    """*Automatic mode only.* InfraConfig that provisions and launches the Docker stack.
    Pass the site resource via ``resources=[...]`` at construction time.
    """

    _handle: ResourceHandle | None = PrivateAttr(default=None)

    def _setup(self) -> None:
        # Validation to ensure exactly one mode is configured correctly.
        has_environments = self.wav_config.environments is not None
        has_infra = self.infra is not None
        has_docker_resources = any(isinstance(r, DockerServiceConfig) for r in self.resources)

        if has_environments and has_infra:
            raise ValueError(
                "Ambiguous setup: provide either wav_config.environments (manual mode) "
                "or infra + resources (automatic mode) — not both."
            )
        if not has_environments and not has_infra:
            raise ValueError(
                "No setup configured. Provide either:\n"
                "  • wav_config=WebArenaVerifiedConfig(environments={...})  — manual mode\n"
                "  • infra=<InfraConfig> + resources=[<DockerServiceConfig>]  — automatic mode"
            )
        if has_infra and not has_docker_resources:
            raise ValueError(
                "infra= is set but no DockerServiceConfig found in resources=. "
                "Pass resources=[WEBARENA_SHOPPING_ADMIN] (or another site config) at construction time."
            )

        # If infra is provided, take control of provisioning/launching and populating wav_config.
        if has_infra:
            self._handle = self._launch_and_configure()
            return

        # Manual mode: verify all configured URLs are reachable.
        for site, env_config in self.wav_config.environments.items():  # type: ignore[union-attr]
            url = env_config.active_url
            if url is None:
                continue
            try:
                urllib.request.urlopen(url, timeout=5)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise RuntimeError(
                    f"Cannot reach {site} at {url}. "
                    f"Start the Docker container with: `webarena-verified env start --site {site.value}`"
                ) from e

    def _launch_and_configure(self) -> ResourceHandle:
        """Provision (if needed) + launch the resource and populate self.wav_config."""
        assert self.infra is not None
        docker_resources = [r for r in self.resources if isinstance(r, DockerServiceConfig)]
        if len(docker_resources) > 1:
            raise ValueError(
                f"infra= only supports a single DockerServiceConfig, but {len(docker_resources)} were found: "
                f"{[r.name for r in docker_resources]}. "
                "Pass exactly one DockerServiceConfig in resources=."
            )
        # Validator guarantees at least one DockerServiceConfig is present.
        (resource,) = docker_resources

        if self.infra.provision_status(resource) == "needs_provisioning":
            logger.info("Provisioning %r on %s …", resource.name, self.infra.fingerprint())
            self.infra.provision(resource)

        logger.info("Launching %r …", resource.name)
        handle = self.infra.launch(resource)
        try:
            self._configure_wav_from_handle(handle, resource)
        except Exception:
            handle.close()
            raise
        return handle

    def _configure_wav_from_handle(self, handle: ResourceHandle, resource: DockerServiceConfig) -> None:
        """Translate handle.endpoints → wav_config.environments via resource.endpoint_to_site."""
        if not resource.endpoint_to_site:
            raise ValueError(
                f"DockerServiceConfig {resource.name!r} has no endpoint_to_site mapping. "
                "Add endpoint_to_site={'service_key': 'webarena_site_value', ...} to the resource."
            )
        environments: dict[WebArenaSite, EnvironmentConfig] = {}
        for service_name, url in handle.endpoints.items():
            site_value = resource.endpoint_to_site.get(service_name)
            if site_value is None:
                continue
            try:
                site = WebArenaSite(site_value)
            except ValueError:
                raise ValueError(
                    f"endpoint_to_site maps {service_name!r} → {site_value!r} which is not a valid WebArenaSite. "
                    f"Valid values: {[s.value for s in WebArenaSite]}"
                )
            if site not in environments:
                environments[site] = EnvironmentConfig(urls=[url])

        if not environments:
            raise ValueError(
                f"handle.endpoints {list(handle.endpoints)!r} produced no WebArenaSite entries. "
                "Check resource.endpoint_to_site."
            )
        self.wav_config = self.wav_config.model_copy(update={"environments": environments})
        logger.info(
            "wav_config.environments: %s",
            {s.value: e.active_url for s, e in environments.items()},
        )

    def close(self) -> None:
        if self._handle is not None:
            logger.info("Closing infra handle for run_id=%s", self._handle.run_id[:8])
            self._handle.close()
            self._handle = None

    def get_task_configs(self) -> Generator[WebArenaVerifiedTaskConfig, None, None]:
        """Yield TaskConfigs with wav_config forwarded from benchmark settings."""
        for tm in self.task_metadata.values():
            yield WebArenaVerifiedTaskConfig(
                task_id=tm.id,
                tool_config=self.default_tool_config,
                wav_config=self.wav_config,
            )
