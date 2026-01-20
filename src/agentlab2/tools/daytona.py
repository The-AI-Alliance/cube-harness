"""Daytona-backed SWE tool for sandboxed code execution."""

import shlex
from uuid import uuid4

from daytona import CreateSandboxFromImageParams, Daytona, DaytonaConfig, Image, Resources, SessionExecuteRequest

from agentlab2.action_spaces.swe_action_space import SWEActionSpace
from agentlab2.tool import Tool, ToolConfig


class DaytonaSWEToolConfig(ToolConfig):
    """Config for Daytona SWE tool."""

    api_key: str | None = None
    image: str = "python:3.13"
    cpus: int = 2
    memory_gb: int = 4
    disk_gb: int = 10

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
        """Create sandbox if not exists."""
        if self._sandbox is None:
            params = CreateSandboxFromImageParams(
                image=Image.base(self.config.image),
                resources=Resources(
                    cpu=self.config.cpus,
                    memory=self.config.memory_gb,
                    disk=self.config.disk_gb,
                ),
                auto_stop_interval=0,
                auto_delete_interval=0,
            )
            self._sandbox = self._client.create(params)

    def bash(self, command: str, timeout: int = 120) -> str:
        """Execute a bash command."""
        self._ensure_sandbox()
        session_id = str(uuid4())

        try:
            self._sandbox.process.create_session(session_id)
            wrapped_cmd = f"bash -ic {shlex.quote(command)}"

            response = self._sandbox.process.execute_session_command(
                session_id,
                SessionExecuteRequest(command=wrapped_cmd, run_async=True),
                timeout=timeout,
            )

            if response.cmd_id is None:
                return "Error: No command ID returned"

            # Poll for completion
            cmd = self._sandbox.process.get_session_command(session_id, response.cmd_id)
            while cmd.exit_code is None:
                import time

                time.sleep(0.5)
                cmd = self._sandbox.process.get_session_command(session_id, response.cmd_id)

            logs = self._sandbox.process.get_session_command_logs(session_id, response.cmd_id)
            output = logs.stdout + logs.stderr

            if cmd.exit_code != 0:
                return f"Exit {cmd.exit_code}: {output}"
            return output

        except Exception as e:
            return f"Error: {e}"

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
