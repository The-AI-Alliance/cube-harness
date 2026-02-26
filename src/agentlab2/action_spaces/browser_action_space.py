from abc import ABC, abstractmethod

from cube.tool import tool_action


class BrowserActionSpace(ABC):
    """Abstract base class for browser tools using CSS selectors.

    Subclasses must implement all methods. The @tool_action decorator on each
    method registers it as a discoverable action — subclass overrides inherit
    that registration automatically without needing to repeat the decorator.

    Used by tools that interact with elements via CSS selectors (e.g., Playwright).
    """

    @tool_action
    @abstractmethod
    def browser_press_key(self, key: str) -> str:
        """Press a key on the keyboard."""
        ...

    @tool_action
    @abstractmethod
    def browser_type(self, selector: str, text: str) -> str:
        """Type text into the element specified by CSS selector."""
        ...

    @tool_action
    @abstractmethod
    def browser_click(self, selector: str) -> str:
        """Click on an element specified by CSS selector."""
        ...

    @tool_action
    @abstractmethod
    def browser_drag(self, from_selector: str, to_selector: str) -> str:
        """Drag and drop from one element to another using CSS selectors."""
        ...

    @tool_action
    @abstractmethod
    def browser_hover(self, selector: str) -> str:
        """Hover over an element specified by CSS selector."""
        ...

    @tool_action
    @abstractmethod
    def browser_select_option(self, selector: str, value: str) -> str:
        """Select an option from an element specified by CSS selector."""
        ...

    @tool_action
    @abstractmethod
    def browser_mouse_click_xy(self, x: int, y: int) -> str:
        """Click at a given x, y coordinate using the mouse."""
        ...

    @tool_action
    @abstractmethod
    def browser_wait(self, seconds: int) -> str:
        """Wait for a given number of seconds, up to max_wait."""
        ...

    @tool_action
    @abstractmethod
    def browser_back(self) -> str:
        """Navigate back in browser history."""
        ...

    @tool_action
    @abstractmethod
    def browser_forward(self) -> str:
        """Navigate forward in browser history."""
        ...

    @tool_action
    @abstractmethod
    def noop(self) -> str:
        """No operation action."""
        ...


class BidBrowserActionSpace(ABC):
    """Abstract base class for browser tools using Browser IDs (BIDs).

    Subclasses must implement all methods. The @tool_action decorator on each
    method registers it as a discoverable action — subclass overrides inherit
    that registration automatically without needing to repeat the decorator.

    Used by tools that interact with elements via BrowserGym's BID system,
    where each interactive element is assigned a unique identifier (e.g., "a51", "b12").
    """

    @tool_action
    @abstractmethod
    def browser_press_key(self, key: str) -> str:
        """Press a key on the keyboard."""
        ...

    @tool_action
    @abstractmethod
    def browser_type(self, bid: str, text: str) -> str:
        """Type text into the element specified by BID."""
        ...

    @tool_action
    @abstractmethod
    def browser_click(self, bid: str) -> str:
        """Click on an element specified by BID."""
        ...

    @tool_action
    @abstractmethod
    def browser_drag(self, from_bid: str, to_bid: str) -> str:
        """Drag and drop from one element to another using BIDs."""
        ...

    @tool_action
    @abstractmethod
    def browser_hover(self, bid: str) -> str:
        """Hover over an element specified by BID."""
        ...

    @tool_action
    @abstractmethod
    def browser_select_option(self, bid: str, value: str) -> str:
        """Select an option from an element specified by BID."""
        ...

    @tool_action
    @abstractmethod
    def browser_mouse_click_xy(self, x: int, y: int) -> str:
        """Click at a given x, y coordinate using the mouse."""
        ...

    @tool_action
    @abstractmethod
    def browser_wait(self, seconds: int) -> str:
        """Wait for a given number of seconds, up to max_wait."""
        ...

    @tool_action
    @abstractmethod
    def browser_back(self) -> str:
        """Navigate back in browser history."""
        ...

    @tool_action
    @abstractmethod
    def browser_forward(self) -> str:
        """Navigate forward in browser history."""
        ...

    @tool_action
    @abstractmethod
    def noop(self) -> str:
        """No operation action."""
        ...
