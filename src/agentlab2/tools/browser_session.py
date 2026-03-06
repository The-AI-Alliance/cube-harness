"""Browser session abstraction for AgentLab2 tools.

A BrowserSession is a handle to a running browser instance, designed to support
three use cases:

  1. Cross-process (same computer): Pass the session to a Ray worker or subprocess.
     TODO: Implement __getstate__/__setstate__ to drop live objects and reconnect
     via cdp_url using pw.chromium.connect_over_cdp(). The cdp_url is already available.

  2. Cross-backend: The task sets up the environment via Playwright (e.g. WorkArena's
     setup(page)), while the tool acts via a different protocol (Puppeteer, raw CDP).
     PlaywrightSession.cdp_url is the shared reference — any backend can attach to it:
         pw.chromium.connect_over_cdp(session.cdp_url)       # Playwright
         connect(browserURL=session.cdp_url)                  # Puppeteer/Pyppeteer

  3. CUA (Computer Use Agent): The tool bypasses the browser protocol and acts at the
     OS level (screenshot + keyboard/mouse). No CDP needed; the session identifies the
     browser window at the OS level instead.
     TODO: CUASession — store PID and/or DISPLAY env var (Linux) / window handle (macOS).

Concrete implementations:
- PlaywrightConfig / PlaywrightSession: Chromium via Playwright, CDP port always enabled.
"""

import logging
import socket
from pathlib import Path

from browsergym.core import _get_global_playwright
from cube.core import TypedBaseModel
from playwright.sync_api import Browser, BrowserContext, Page
from pydantic import Field

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class BrowserConfig(TypedBaseModel):
    """Abstract serializable config for a browser session.

    Call make() to launch a browser and get a live BrowserSession. The config holds
    all parameters needed to reproduce the launch and must be fully serializable.

    Subclasses:
    - PlaywrightSessionConfig: Chromium via Playwright (current)
    # Future: CUAConfig — no browser protocol; just OS-level window metadata
    """

    def make(self) -> "BrowserSession":
        """Launch a browser and return a live BrowserSession."""
        raise NotImplementedError


class BrowserSession:
    """Abstract live browser session handle.

    Represents a running browser instance that can be shared across processes and
    backends. See the module docstring for the three design goals this abstraction serves.

    Implementations own the live browser resources and must implement stop().

    All sessions must implement get_playwright_session() — Playwright is the standard
    interface for browser interaction in this codebase. Non-Playwright backends (e.g.
    CUASession) connect lazily via CDP: pw.chromium.connect_over_cdp(self.cdp_url).

    cdp_url is set by subclasses that expose a CDP port. It is None for session types
    that identify the browser at the OS level (PID/Display) rather than via a network port.

    Subclasses:
    - PlaywrightSession: owns Playwright objects directly; cdp_url always set
    # Future: CUASession — identified via OS process PID and/or Display env var;
    #   get_playwright_session() connects via pw.chromium.connect_over_cdp(cdp_url)
    """

    cdp_url: str | None = None

    def get_playwright_session(self) -> tuple[Page, BrowserContext]:
        """Return a live Playwright (page, context) for this browser.

        For Playwright-native sessions this returns the live objects directly.
        For other backends (e.g. CUASession) this connects via CDP lazily:
            pw.chromium.connect_over_cdp(self.cdp_url)
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Close the browser and release all resources."""
        raise NotImplementedError


class PlaywrightSessionConfig(BrowserConfig):
    """Serializable Playwright launch parameters.

    Call make() to start a Chromium browser and get a live PlaywrightSession.
    The browser is always launched with --remote-debugging-port so the returned
    session exposes a cdp_url for cross-backend access.
    """

    headless: bool = True
    viewport: dict[str, int] = Field(default_factory=lambda: {"width": 1280, "height": 720})
    slow_mo: int | None = None
    timeout: int | None = None
    locale: str | None = None
    timezone_id: str | None = None

    # Advanced Playwright options (rarely needed)
    resizeable_window: bool = False
    pw_chromium_kwargs: dict = Field(default_factory=dict)
    pw_context_kwargs: dict = Field(default_factory=dict)
    record_video_dir: str | None = None

    def make(self) -> "PlaywrightSession":
        """Launch a Chromium browser and return a live PlaywrightSession."""
        pw = _get_global_playwright()

        cdp_port = _find_free_port()
        args = [
            f"--window-size={self.viewport['width']},{self.viewport['height']}" if self.resizeable_window else None,
            "--disable-features=OverlayScrollbars,ExtendedOverlayScrollbars",
            f"--remote-debugging-port={cdp_port}",
        ]
        browser = pw.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=[arg for arg in args if arg is not None],
            ignore_default_args=["--hide-scrollbars"],
            **self.pw_chromium_kwargs,
        )
        context = browser.new_context(
            no_viewport=True if self.resizeable_window else None,
            viewport=self.viewport if not self.resizeable_window else None,
            record_video_dir=Path(self.record_video_dir) / "task_video" if self.record_video_dir else None,
            record_video_size=self.viewport,
            locale=self.locale,
            timezone_id=self.timezone_id,
            **self.pw_context_kwargs,
        )
        if self.timeout is not None:
            context.set_default_timeout(self.timeout)
        page = context.new_page()
        return PlaywrightSession(page=page, context=context, browser=browser, cdp_url=f"http://localhost:{cdp_port}")


class PlaywrightSession(BrowserSession):
    """Live Playwright browser session.

    Owns the Playwright page, context, and browser launched by PlaywrightSessionConfig.
    Always exposes a cdp_url (--remote-debugging-port) for cross-backend access.
    """

    def __init__(self, page: Page, context: BrowserContext, browser: Browser, cdp_url: str) -> None:
        self._page: Page | None = page
        self._context: BrowserContext | None = context
        self._browser: Browser | None = browser
        self.cdp_url: str = cdp_url

    def get_playwright_session(self) -> tuple[Page, BrowserContext]:
        """Return the live (page, context)."""
        if self._page is None or self._context is None:
            raise RuntimeError("BrowserSession has been stopped or was never started.")
        return self._page, self._context

    def stop(self) -> None:
        """Close the browser and release all Playwright resources."""
        if self._context is not None:
            try:
                self._context.close()
            except Exception as e:
                logger.warning(f"Error closing browser context: {e}")
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
        self._page = None
        self._context = None
        self._browser = None
