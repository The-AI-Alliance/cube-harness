"""Protocol for browser tools that support web-based task benchmarks."""

from typing import Any, Protocol, runtime_checkable

from agentlab2.core import Observation


@runtime_checkable
class BrowserTaskTool(Protocol):
    """
    Protocol defining the interface that browser tools must implement to support
    web-based benchmarks like MiniWob, WebArena, VisualWebArena, etc.

    This protocol defines task utility methods (navigation, JS evaluation, observation)
    that are separate from agent actions. Tasks use these methods for setup, validation,
    and observation collection.

    Implementations:
    - SyncPlaywrightTool: Direct Playwright browser control
    - BrowsergymTool: BrowserGym environment wrapper

    Note: This protocol may need revisiting if we add fundamentally different tool types
    (e.g., mobile apps, desktop automation) that don't fit the browser paradigm.
    """

    def goto(self, url: str) -> None:
        """
        Navigate to a URL.

        Args:
            url: The URL to navigate to.
        """
        ...

    def evaluate_js(self, js: str) -> Any:
        """
        Evaluate JavaScript code in the browser context and return the result.

        Args:
            js: JavaScript code to evaluate.

        Returns:
            The result of the JavaScript evaluation.
        """
        ...

    def page_obs(self) -> Observation:
        """
        Get the current page observation (HTML, screenshot, accessibility tree).

        Returns:
            Observation containing the current page state.
        """
        ...

    @property
    def page(self) -> Any:
        """
        Access the underlying page object (e.g., Playwright Page) for advanced use cases.

        This provides an "escape hatch" for tasks that need direct access to browser
        APIs not covered by this protocol.

        Returns:
            The raw page object (Playwright Page or equivalent).
        """
        ...
