from typing import Protocol, runtime_checkable

from agentlab2.core import ActionSpace


@runtime_checkable
class SWEActionSpace(ActionSpace, Protocol):
    """Actions for SWE tasks in a sandboxed environment."""

    def bash(self, command: str, timeout: int = 120) -> str:
        """Execute a bash command."""
        ...

    def read_file(self, path: str) -> str:
        """Read file contents."""
        ...

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file."""
        ...
