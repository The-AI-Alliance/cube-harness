"""Docker-based SWE tool for running tasks in local Docker containers.

This tool implements the SWEActionSpace interface using local Docker containers,
providing an alternative to Daytona for running benchmarks locally.
"""

import io
import logging
import shlex
import tarfile
from pathlib import Path

import docker
from docker.errors import ImageNotFound, NotFound
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from agentlab2.action_spaces.swe_action_space import SWEActionSpace
from agentlab2.tool import Tool, ToolConfig

logger = logging.getLogger(__name__)

# Retry decorators for transient Docker errors
_retry_container = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
_retry_io = retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


class DockerSWEToolConfig(ToolConfig):
    """Config for Docker SWE tool.

    Mirrors DaytonaSWEToolConfig interface for easy swapping.
    """

    image: str = "python:3.13"
    cpus: int = 2
    memory_gb: int = 4
    disk_gb: int = 10  # Not enforced locally, kept for interface compatibility
    working_dir: str = "/app"
    network_mode: str = "bridge"  # or "none" for isolated containers
    remove_on_close: bool = True
    pull_policy: str = "missing"  # "always", "missing", or "never"

    def make(self) -> "DockerSWETool":
        return DockerSWETool(self)


class DockerSWETool(Tool, SWEActionSpace):
    """SWE tool backed by local Docker container."""

    action_space = SWEActionSpace

    def __init__(self, config: DockerSWEToolConfig) -> None:
        self.config = config
        self._client = docker.from_env()
        self._container = None

    def _ensure_container(self) -> None:
        """Create container if not exists."""
        if self._container is None:
            self._container = self._create_container()

    @_retry_container
    def _create_container(self):
        """Create a Docker container with retry on transient failures."""
        image = self.config.image
        logger.info(f"Creating Docker container with image: {image}")

        # Pull image if needed
        if self.config.pull_policy == "always":
            logger.info(f"Pulling image: {image}")
            self._client.images.pull(image)
        elif self.config.pull_policy == "missing":
            try:
                self._client.images.get(image)
            except ImageNotFound:
                logger.info(f"Image not found locally, pulling: {image}")
                self._client.images.pull(image)

        # Resource limits
        mem_limit = f"{self.config.memory_gb}g"
        nano_cpus = int(self.config.cpus * 1e9)  # Docker uses nano CPUs

        container = self._client.containers.run(
            image,
            command="sleep infinity",  # Keep container running
            detach=True,
            working_dir=self.config.working_dir,
            mem_limit=mem_limit,
            nano_cpus=nano_cpus,
            network_mode=self.config.network_mode,
            stdin_open=True,
            tty=True,
        )

        logger.info(f"Docker container created: {container.short_id}")
        return container

    def bash(self, command: str, timeout: int = 120) -> str:
        """Execute a bash command and return structured output."""
        self._ensure_container()

        try:
            # Use bash -lc to get login shell environment
            wrapped_cmd = f"bash -lc {shlex.quote(command)}"

            exit_code, output = self._exec_command(wrapped_cmd, timeout)

            # Decode output
            stdout = output.decode("utf-8", errors="replace").strip() if output else ""

            parts = []
            if stdout:
                parts.append(stdout)
            if exit_code != 0:
                parts.append(f"[exit_code: {exit_code}]")

            return "\n".join(parts) if parts else "(no output)"

        except Exception as e:
            return f"[error] {e}"

    @_retry_io
    def _exec_command(self, command: str, timeout: int) -> tuple[int, bytes]:
        """Execute command in container with retry."""
        exec_result = self._container.exec_run(
            command,
            workdir=self.config.working_dir,
            demux=False,  # Combined stdout/stderr
            timeout=timeout,
        )
        return exec_result.exit_code, exec_result.output

    def read_file(self, path: str) -> str:
        """Read file contents from container."""
        self._ensure_container()
        try:
            return self._download_file(path)
        except Exception as e:
            return f"Error reading {path}: {e}"

    @_retry_io
    def _download_file(self, path: str) -> str:
        """Download file from container with retry."""
        try:
            bits, _ = self._container.get_archive(path)
            # Extract from tar archive
            tar_stream = io.BytesIO()
            for chunk in bits:
                tar_stream.write(chunk)
            tar_stream.seek(0)

            with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                # Get the file from the archive (filename is the basename)
                member = tar.getmembers()[0]
                f = tar.extractfile(member)
                if f is None:
                    return ""
                return f.read().decode("utf-8")
        except NotFound:
            return f"Error: File not found: {path}"

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the container."""
        self._ensure_container()
        self._upload_file(path, content)

    @_retry_io
    def _upload_file(self, path: str, content: str) -> None:
        """Upload file to container with retry."""
        # Create tar archive in memory
        tar_stream = io.BytesIO()
        content_bytes = content.encode("utf-8")

        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            # Create file info
            file_info = tarfile.TarInfo(name=Path(path).name)
            file_info.size = len(content_bytes)
            tar.addfile(file_info, io.BytesIO(content_bytes))

        tar_stream.seek(0)

        # Ensure parent directory exists
        parent_dir = str(Path(path).parent)
        self._container.exec_run(f"mkdir -p {shlex.quote(parent_dir)}")

        # Put archive into container
        self._container.put_archive(parent_dir, tar_stream)

    def reset(self) -> None:
        """Delete and recreate container."""
        self._delete_container()

    def close(self) -> None:
        """Clean up container."""
        self._delete_container()

    def _delete_container(self) -> None:
        """Delete container with error handling."""
        if self._container:
            try:
                self._delete_container_with_retry()
            except Exception as e:
                logger.error(f"Failed to delete container after retries: {e}")
            finally:
                self._container = None

    @_retry_io
    def _delete_container_with_retry(self) -> None:
        """Stop and remove container with retry."""
        logger.info(f"Stopping container: {self._container.short_id}")
        self._container.stop(timeout=5)
        if self.config.remove_on_close:
            logger.info(f"Removing container: {self._container.short_id}")
            self._container.remove()
