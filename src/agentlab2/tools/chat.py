"""Chat tool for sending messages to benchmark users."""

from agentlab2.action_spaces.chat_action_space import ChatActionSpace
from agentlab2.tool import Tool, ToolConfig
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str

class ChatConfig(ToolConfig):
    """Configuration for ChatTool."""

    def make(self) -> "ChatTool":
        return ChatTool()


class ChatTool(Tool, ChatActionSpace):
    """In-memory chat tool used by tasks that validate user messages."""

    action_space = ChatActionSpace

    def __init__(self) -> None:
        self._messages: list[ChatMessage] = []

    @property
    def messages(self) -> list[ChatMessage]:
        """Return all sent chat messages in order."""
        return list(self._messages)
    
    def add_message(self, message: str, role: str = "assistant") -> None:
        self._messages.append({"role": role, "content": message})

    def send_message(self, message: str) -> str:
        """Send a message to the user chat."""
        self.add_message(message, "assistant")
        return "Success"

    def reset(self) -> None:
        """Clear chat history for a new task."""
        self._messages.clear()
