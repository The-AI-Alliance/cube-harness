import io
import logging
import shlex
import tarfile
import tempfile
from pathlib import Path
from uuid import uuid4

from daytona import CreateSandboxFromImageParams, Daytona, DaytonaConfig, Image, Resources, SessionExecuteRequest

from agentlab2.action_spaces.swe_action_space import SWEActionSpace
from agentlab2.tool import Tool, ToolConfig

logger = logging.getLogger(__name__)


class DaytonaSWEToolConfig(ToolConfig):
    """Config for Daytona SWE tool."""

    api_key: str | None = None
    image: str = "python:3.13"
    dockerfile_archive: bytes | None = None  # If set, extracts archive and builds from Dockerfile
    cpus: int = 2
    memory_gb: int = 4
    disk_gb: int = 10
    ephemeral: bool = True

    model_config = {"arbitrary_types_allowed": True}

    def make(self) -> "DaytonaSWETool":
        return DaytonaSWETool(self)


class DaytonaSWETool(Tool, SWEActionSpace):
    """SWE tool backed by Daytona sandbox."""

    action_space = SWEActionSpace

    def __init__(self, config: DaytonaSWEToolConfig) -> None:
        self.config = config
        daytona_config = DaytonaConfig(api_key=config.api_key) if config.api_key else DaytonaConfig()
        self._client = Daytona(daytona_config)
        self._sandbox = None
        self._dockerfile_temp_dir: tempfile.TemporaryDirectory | None = None

    def _ensure_sandbox(self) -> None:
        """Create sandbox if not exists."""
        if self._sandbox is None:
            # Determine image: use Dockerfile from archive if provided, otherwise base image
            if self.config.dockerfile_archive:
                # Extract archive to temp directory (provides build context for COPY commands)
                self._dockerfile_temp_dir = tempfile.TemporaryDirectory()
                extract_path = Path(self._dockerfile_temp_dir.name)
                with tarfile.open(fileobj=io.BytesIO(self.config.dockerfile_archive), mode="r:gz") as tar:
                    tar.extractall(path=extract_path, filter="data")

                # Find the Dockerfile in the extracted archive
                dockerfile_path = self._find_dockerfile(extract_path)
                if dockerfile_path:
                    image = Image.from_dockerfile(str(dockerfile_path))
                    logger.info(f"Building image from Dockerfile: {dockerfile_path}")
                else:
                    image = Image.base(self.config.image)
                    logger.warning(f"No Dockerfile found in archive, using base image: {self.config.image}")
            else:
                image = Image.base(self.config.image)
                logger.info(f"Using base image: {self.config.image}")

            params = CreateSandboxFromImageParams(
                image=image,
                resources=Resources(
                    cpu=self.config.cpus,
                    memory=self.config.memory_gb,
                    disk=self.config.disk_gb,
                ),
                auto_stop_interval=0,
                auto_delete_interval=0,
                ephemeral=self.config.ephemeral,
            )
            self._sandbox = self._client.create(params)
            logger.info(f"Daytona sandbox SSH URL: `{self._sandbox.create_ssh_access().ssh_command}`")

    def _find_dockerfile(self, extract_path: Path) -> Path | None:
        """Find Dockerfile in extracted archive."""
        # Check common locations
        for candidate in [
            extract_path / "Dockerfile",
            extract_path / "." / "Dockerfile",
        ]:
            if candidate.exists():
                return candidate

        # Search recursively
        for dockerfile in extract_path.rglob("Dockerfile"):
            return dockerfile

        return None

    def bash(self, command: str, timeout: int = 120) -> str:
        """Execute a bash command and return structured output."""
        self._ensure_sandbox()
        session_id = str(uuid4())

        try:
            self._sandbox.process.create_session(session_id)
            # Use login shell (-l) without interactive mode (-i) to avoid terminal noise
            wrapped_cmd = f"bash -lc {shlex.quote(command)}"

            response = self._sandbox.process.execute_session_command(
                session_id,
                SessionExecuteRequest(command=wrapped_cmd, run_async=True),
                timeout=timeout,
            )

            if response.cmd_id is None:
                return "[error] No command ID returned"

            # Poll for completion
            cmd = self._sandbox.process.get_session_command(session_id, response.cmd_id)
            while cmd.exit_code is None:
                import time

                time.sleep(0.5)
                cmd = self._sandbox.process.get_session_command(session_id, response.cmd_id)

            logs = self._sandbox.process.get_session_command_logs(session_id, response.cmd_id)

            # Build structured output
            stdout = logs.stdout.strip() if logs.stdout else ""
            stderr = logs.stderr.strip() if logs.stderr else ""

            parts = []
            if stdout:
                parts.append(stdout)
            if stderr:
                parts.append(f"[stderr]\n{stderr}")
            if cmd.exit_code != 0:
                parts.append(f"[exit_code: {cmd.exit_code}]")

            return "\n".join(parts) if parts else "(no output)"

        except Exception as e:
            return f"[error] {e}"

    def read_file(self, path: str) -> str:
        """Read file contents."""
        self._ensure_sandbox()
        try:
            content = self._sandbox.fs.download_file(path)
            return content.decode("utf-8") if content else ""
        except Exception as e:
            return f"Error reading {path}: {e}"

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file."""
        self._ensure_sandbox()
        self._sandbox.fs.upload_file(content.encode("utf-8"), path)

    def reset(self) -> None:
        """Delete and recreate sandbox."""
        if self._sandbox:
            self._client.delete(self._sandbox)
            self._sandbox = None

    def close(self) -> None:
        """Clean up sandbox."""
        if self._sandbox:
            self._client.delete(self._sandbox)
            self._sandbox = None
        if self._dockerfile_temp_dir:
            self._dockerfile_temp_dir.cleanup()
            self._dockerfile_temp_dir = None
