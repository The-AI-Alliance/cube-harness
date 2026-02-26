from typing import Protocol


class ChatActionSpace(Protocol):
    """Action space for sending user-facing chat messages."""

    def send_message(self, message: str) -> str:
        """Send a message to the user chat."""
        ...
