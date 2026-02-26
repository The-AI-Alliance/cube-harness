from typing import Protocol


class BrowserActionSpace(Protocol):
    """Browser action space using CSS selectors.

    Used by tools that interact with elements via CSS selectors (e.g., Playwright).
    """

    def browser_press_key(self, key: str) -> str:
        """Press a key on the keyboard."""
        ...

    def browser_type(self, selector: str, text: str) -> str:
        """Type text into the element specified by CSS selector."""
        ...

    def browser_click(self, selector: str) -> str:
        """Click on an element specified by CSS selector."""
        ...

    def browser_drag(self, from_selector: str, to_selector: str) -> str:
        """Drag and drop from one element to another using CSS selectors."""
        ...

    def browser_hover(self, selector: str) -> str:
        """Hover over an element specified by CSS selector."""
        ...

    def browser_select_option(self, selector: str, value: str) -> str:
        """Select an option from an element specified by CSS selector."""
        ...

    def browser_mouse_click_xy(self, x: int, y: int) -> str:
        """Click at a given x, y coordinate using the mouse."""
        ...

    def browser_wait(self, seconds: int) -> str:
        """Wait for a given number of seconds, up to max_wait."""
        ...

    def browser_back(self) -> str:
        """Navigate back in browser history."""
        ...

    def browser_forward(self) -> str:
        """Navigate forward in browser history."""
        ...

    def noop(self) -> str:
        """No operation action."""
        ...


class BidBrowserActionSpace(Protocol):
    """Browser action space using Browser IDs (BIDs).

    Used by tools that interact with elements via BrowserGym's BID system,
    where each interactive element is assigned a unique identifier (e.g., "a51", "b12").
    """

    def browser_press_key(self, key: str) -> str:
        """Press a key on the keyboard."""
        ...

    def browser_type(self, bid: str, text: str) -> str:
        """Type text into the element specified by BID."""
        ...

    def browser_click(self, bid: str) -> str:
        """Click on an element specified by BID."""
        ...

    def browser_drag(self, from_bid: str, to_bid: str) -> str:
        """Drag and drop from one element to another using BIDs."""
        ...

    def browser_hover(self, bid: str) -> str:
        """Hover over an element specified by BID."""
        ...

    def browser_select_option(self, bid: str, value: str) -> str:
        """Select an option from an element specified by BID."""
        ...

    def browser_mouse_click_xy(self, x: int, y: int) -> str:
        """Click at a given x, y coordinate using the mouse."""
        ...

    def browser_wait(self, seconds: int) -> str:
        """Wait for a given number of seconds, up to max_wait."""
        ...

    def browser_back(self) -> str:
        """Navigate back in browser history."""
        ...

    def browser_forward(self) -> str:
        """Navigate forward in browser history."""
        ...

    def browser_scroll(self, delta_x: float, delta_y: float) -> str:
        """Scroll the page by horizontal and vertical deltas."""
        ...

    def browser_dbclick(self, bid: str) -> str:
        """Double-click on an element specified by BID."""
        ...

    def browser_press(self, bid: str, comb: str) -> str:
        """Press key combination on an element specified by BID."""
        ...

    def browser_clear(self, bid: str) -> str:
        """Clear text/content from an element specified by BID."""
        ...

    def browser_goto(self, url: str) -> str:
        """Navigate the current page to a URL."""
        ...

    def browser_focus(self, bid: str) -> str:
        """Focus an element specified by BID."""
        ...

    def noop(self) -> str:
        """No operation action."""
        ...
