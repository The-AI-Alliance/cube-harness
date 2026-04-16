import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import ClassVar, Generator

from pydantic import PrivateAttr
from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.resource import DockerServiceConfig, InfraConfig, ResourceHandle
from cube.task import TaskConfig, TaskMetadata
from webarena_verified.api.webarena_verified import WebArenaVerified
from webarena_verified.types.agent_response import MainObjectiveType
from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
from webarena_verified.types.task import WebArenaSite

from cube.tool import ToolboxConfig

from webarena_verified_cube.task import WebArenaVerifiedTaskConfig
from webarena_verified_cube.tool import HarPlaywrightConfig, SubmitResponseConfig

logger = logging.getLogger(__name__)

_TASK_METADATA_JSON = Path(__file__).parent / "task_metadata.json"


class WebArenaVerifiedBenchmark(Benchmark):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="webarena-verified-cube",
        version="1.0.0",
        description="WebArena-Verified benchmark — 812 verified web automation tasks across 6 platforms",
        num_tasks=812,
        tags=["browser", "web", "ui", "webarena"],
    )
    # task_metadata: populated automatically at import time in Benchmark.__init_subclass__
    task_config_class: ClassVar[type[TaskConfig]] = WebArenaVerifiedTaskConfig

    default_tool_config: ToolboxConfig = ToolboxConfig(tool_configs=[HarPlaywrightConfig(), SubmitResponseConfig()])  # type: ignore

    wav_config: WebArenaVerifiedConfig = WebArenaVerifiedConfig()
    sites_filter: list[WebArenaSite] | None = None
    action_filter: MainObjectiveType | None = None
    task_ids_filter: list[int] | None = None

    infra: InfraConfig | None = None
    """When set, provision (if needed) + launch happens automatically in setup().

    Pass the ``DockerServiceConfig`` via ``resources=[...]`` at construction time.
    The handle's ``endpoints`` are translated to ``wav_config.environments`` using
    ``resource.endpoint_to_site``.  If ``None``, the benchmark behaves exactly as
    before: ``wav_config.environments`` must be populated manually.
    """

    _handle: ResourceHandle | None = PrivateAttr(default=None)

    @classmethod
    def install(cls) -> None:
        """Generate and cache task_metadata.json from the webarena-verified library.

        No download required — data comes from the installed package.
        Safe to call multiple times: skips generation if the file already exists.
        """
        if _TASK_METADATA_JSON.exists():
            logger.info("task_metadata.json already exists, skipping generation")
            return
        logger.info("Generating task_metadata.json from webarena-verified library...")
        wav = WebArenaVerified()
        metadata = {
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

        _TASK_METADATA_JSON.write_text(json.dumps([tm.model_dump() for tm in metadata.values()], indent=2))
        cls.task_metadata = metadata
        logger.info(f"Saved {len(metadata)} tasks to {_TASK_METADATA_JSON}")

    @classmethod
    def uninstall(cls) -> None:
        if _TASK_METADATA_JSON.exists():
            _TASK_METADATA_JSON.unlink()
            cls.task_metadata = {}
            logger.info(f"Removed {_TASK_METADATA_JSON}")

    def _setup(self) -> None:
        if self.infra is not None:
            self._handle = self._launch_and_configure()
            return

        # Legacy path: verify manually-configured URLs are reachable.
        if self.wav_config.environments is None:
            return
        for site, env_config in self.wav_config.environments.items():
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
        if not docker_resources:
            raise ValueError(
                "infra= is set but no DockerServiceConfig found in resources=. "
                "Pass the DockerServiceConfig as resources=[...] at construction time."
            )
        if len(docker_resources) > 1:
            raise ValueError(
                f"infra= only supports a single DockerServiceConfig, but {len(docker_resources)} were found: "
                f"{[r.name for r in docker_resources]}. "
                "Pass exactly one DockerServiceConfig in resources=."
            )
        # Unpack the single validated item — tuple unpacking enforces the invariant at runtime.
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
