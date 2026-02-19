"""Computer action space for desktop/VM interactions.

This module defines the ComputerActionSpace protocol for OSWorld-style desktop automation.
Based on the Computer_13 action set with 15 actions for mouse, keyboard, and control operations.
"""

from typing import List, Literal, Optional, Protocol

ButtonType = Literal["left", "right", "middle"]


class ComputerActionSpace(Protocol):
    """Computer use action space for desktop/VM interactions.

    Based on OSWorld's Computer_13 action set with 15 actions:
    - Mouse: move_to, click, double_click, right_click, mouse_down, mouse_up, drag_to, scroll
    - Keyboard: typing, press, key_down, key_up, hotkey
    - Control: wait, done, fail
    """

    def move_to(self, x: int, y: int) -> str:
        """Move the cursor to the specified position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Success message
        """
        ...

    def click(
        self,
        button: ButtonType = "left",
        x: Optional[int] = None,
        y: Optional[int] = None,
        num_clicks: Literal[1, 2, 3] = 1,
    ) -> str:
        """Click the mouse button.

        Click the left button if button not specified, otherwise click the specified button.
        Click at the current position if x and y are not specified, otherwise click at the specified position.

        Args:
            button: Mouse button to click ("left", "right", or "middle")
            x: X coordinate (optional)
            y: Y coordinate (optional)
            num_clicks: Number of clicks (1, 2, or 3)

        Returns:
            Success message
        """
        ...

    def mouse_down(self, button: ButtonType = "left") -> str:
        """Press the mouse button down.

        Press the left button if button not specified, otherwise press the specified button.

        Args:
            button: Mouse button to press ("left", "right", or "middle")

        Returns:
            Success message
        """
        ...

    def mouse_up(self, button: ButtonType = "left") -> str:
        """Release the mouse button.

        Release the left button if button not specified, otherwise release the specified button.

        Args:
            button: Mouse button to release ("left", "right", or "middle")

        Returns:
            Success message
        """
        ...

    def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> str:
        """Right click the mouse.

        Right click at the current position if x and y are not specified,
        otherwise right click at the specified position.

        Args:
            x: X coordinate (optional)
            y: Y coordinate (optional)

        Returns:
            Success message
        """
        ...

    def double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> str:
        """Double click the mouse.

        Double click at the current position if x and y are not specified,
        otherwise double click at the specified position.

        Args:
            x: X coordinate (optional)
            y: Y coordinate (optional)

        Returns:
            Success message
        """
        ...

    def drag_to(self, x: int, y: int) -> str:
        """Drag the cursor to the specified position with the left button pressed.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Success message
        """
        ...

    def scroll(self, dx: int, dy: int) -> str:
        """Scroll the mouse wheel.

        Args:
            dx: Horizontal scroll amount (negative = left, positive = right)
            dy: Vertical scroll amount (negative = down, positive = up)

        Returns:
            Success message
        """
        ...

    def typing(self, text: str) -> str:
        """Type the specified text.

        Args:
            text: Text to type

        Returns:
            Success message
        """
        ...

    def press(self, key: str) -> str:
        """Press the specified key and release it.

        Args:
            key: Key to press

        Returns:
            Success message
        """
        ...

    def key_down(self, key: str) -> str:
        """Press the specified key down (without releasing).

        Args:
            key: Key to press down

        Returns:
            Success message
        """
        ...

    def key_up(self, key: str) -> str:
        """Release the specified key.

        Args:
            key: Key to release

        Returns:
            Success message
        """
        ...

    def hotkey(self, keys: List[str]) -> str:
        """Press the specified key combination.

        Args:
            keys: Array of keys to press simultaneously (e.g., ["ctrl", "c"])

        Returns:
            Success message
        """
        ...

    def wait(self) -> str:
        """Wait until the next action (no-op).

        Returns:
            Success message
        """
        ...

    def fail(self) -> str:
        """Signal that the task cannot be performed.

        Returns:
            Success message
        """
        ...

    def done(self) -> str:
        """Signal that the task is complete.

        Returns:
            Success message
        """
        ...
