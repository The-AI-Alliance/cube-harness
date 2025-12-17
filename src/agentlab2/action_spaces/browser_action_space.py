from typing import Protocol

from agentlab2.core import ActionSpace


class BrowserActionSpace(ActionSpace, Protocol):
    def browser_press_key(self, key: str):
        """Press a key on the keyboard."""
        ...

    def browser_type(self, selector: str, text: str):
        """Type text into the focused element."""
        ...

    def browser_click(self, selector: str):
        """Click on a selector."""
        ...

    def browser_drag(self, from_selector: str, to_selector: str):
        """Drag and drop from one selector to another."""
        ...

    def browser_hover(self, selector: str):
        """Hover over a given element."""
        ...

    def browser_select_option(self, selector: str, value: str):
        """Select an option from a given element."""
        ...

    def browser_mouse_click_xy(self, x: int, y: int):
        """Click at a given x, y coordinate using the mouse."""
        ...

    def browser_wait(self, seconds: int):
        """Wait for a given number of seconds, up to max_wait"""
        ...

    def browser_back(self):
        """Navigate back in browser history."""
        ...

    def browser_forward(self):
        """Navigate forward in browser history."""
        ...

    def noop(self):
        """No operation action."""
        ...
