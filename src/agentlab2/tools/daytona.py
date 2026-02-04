import logging
import shlex
import time
from uuid import uuid4

from daytona import CreateSandboxFromImageParams, Daytona, DaytonaConfig, Image, Resources, SessionExecuteRequest
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from agentlab2.action_spaces.swe_action_space import SWEActionSpace
from agentlab2.tool import Tool, ToolConfig

logger = logging.getLogger(__name__)

# Retry decorators following Harbor's pattern (tenacity + exponential backoff)
_retry_sandbox = retry(
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
_retry_poll = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


class DaytonaSWEToolConfig(ToolConfig):
    """Config for Daytona SWE tool."""

    api_key: str | None = None
    image: str = "python:3.13"
    cpus: int = 2
    memory_gb: int = 4
    disk_gb: int = 10
    ephemeral: bool = True
    auto_stop_minutes: int = 10  # Auto-stop after N minutes of inactivity (safety net)
    auto_delete_minutes: int = 5  # Auto-delete N minutes after stopping

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

    def _ensure_sandbox(self) -> None:
        """Create sandbox if not exists (with retry)."""
        if self._sandbox is None:
            self._sandbox = self._create_sandbox()

    @_retry_sandbox
    def _create_sandbox(self):
        """Create a Daytona sandbox with retry on transient failures."""
        logger.info(f"Creating sandbox with image: {self.config.image}")
        params = CreateSandboxFromImageParams(
            image=Image.base(self.config.image),
            resources=Resources(
                cpu=self.config.cpus,
                memory=self.config.memory_gb,
                disk=self.config.disk_gb,
            ),
            auto_stop_interval=self.config.auto_stop_minutes,
            auto_delete_interval=self.config.auto_delete_minutes,
            ephemeral=self.config.ephemeral,
        )
        sandbox = self._client.create(params)
        logger.info(f"Daytona sandbox SSH URL: `{sandbox.create_ssh_access().ssh_command}`")
        return sandbox

    @_retry_poll
    def _get_session_command(self, session_id: str, command_id: str):
        """Get session command status with retry on transient failures."""
        return self._sandbox.process.get_session_command(session_id, command_id)

    @_retry_poll
    def _get_session_command_logs(self, session_id: str, command_id: str):
        """Get session command logs with retry on transient failures."""
        return self._sandbox.process.get_session_command_logs(session_id, command_id)

    def _poll_command(self, session_id: str, command_id: str, timeout: int) -> tuple:
        """Poll for command completion with a hard timeout."""
        deadline = time.monotonic() + timeout
        cmd = self._get_session_command(session_id, command_id)

        while cmd.exit_code is None:
            if time.monotonic() > deadline:
                return cmd, True  # timed out
            time.sleep(0.5)
            cmd = self._get_session_command(session_id, command_id)

        return cmd, False

    def bash(self, command: str, timeout: int = 120) -> str:
        """Execute a bash command and return structured output."""
        self._ensure_sandbox()
        session_id = str(uuid4())

        try:
            self._sandbox.process.create_session(session_id)
            wrapped_cmd = f"bash -lc {shlex.quote(command)}"

            response = self._sandbox.process.execute_session_command(
                session_id,
                SessionExecuteRequest(command=wrapped_cmd, run_async=True),
                timeout=timeout,
            )

            if response.cmd_id is None:
                return "[error] No command ID returned"

            cmd, timed_out = self._poll_command(session_id, response.cmd_id, timeout)

            if timed_out:
                return f"[error] Command timed out after {timeout}s"

            logs = self._get_session_command_logs(session_id, response.cmd_id)

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
            return self._download_file(path)
        except Exception as e:
            return f"Error reading {path}: {e}"

    @_retry_io
    def _download_file(self, path: str) -> str:
        """Download file with retry on transient failures."""
        content = self._sandbox.fs.download_file(path)
        return content.decode("utf-8") if content else ""

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file."""
        self._ensure_sandbox()
        self._upload_file(path, content)

    @_retry_io
    def _upload_file(self, path: str, content: str) -> None:
        """Upload file with retry on transient failures."""
        self._sandbox.fs.upload_file(content.encode("utf-8"), path)

    def reset(self) -> None:
        """Delete and recreate sandbox."""
        self._delete_sandbox()

    def close(self) -> None:
        """Clean up sandbox."""
        self._delete_sandbox()

    def _delete_sandbox(self) -> None:
        """Delete sandbox with retry on transient failures."""
        if self._sandbox:
            try:
                self._delete_sandbox_with_retry()
            except Exception as e:
                logger.error(f"Failed to delete sandbox after retries: {e}")
            finally:
                self._sandbox = None

    @_retry_io
    def _delete_sandbox_with_retry(self) -> None:
        self._client.delete(self._sandbox)
