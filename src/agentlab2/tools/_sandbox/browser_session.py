"""
BrowserSession — prototype of the new browser lifecycle abstraction.

The session is created by the task during setup() and passed to the tool via
tool.connect(session). This decouples "who starts the browser" from "who acts on it".
"""

from dataclasses import dataclass, field

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, ViewportSize, sync_playwright


@dataclass
class BrowserSessionConfig:
    """Declarative config for starting a browser. Lives on the task, not the tool."""

    headless: bool = True
    viewport: ViewportSize = field(default_factory=lambda: {"width": 1280, "height": 720})
    cdp_port: int | None = None  # if set, enables chrome remote debugging on that port

    def start(self) -> "BrowserSession":
        """Start a Chromium browser and return a BrowserSession handle."""
        pw = sync_playwright().start()
        launch_args = [f"--remote-debugging-port={self.cdp_port}"] if self.cdp_port else []
        browser = pw.chromium.launch(headless=self.headless, args=launch_args)
        context = browser.new_context(viewport=self.viewport)
        page = context.new_page()
        # connect_over_cdp expects an HTTP URL; Playwright fetches /json/version to get the real WS URL.
        cdp_url = f"http://localhost:{self.cdp_port}" if self.cdp_port else None
        return BrowserSession(page=page, browser=browser, context=context, playwright=pw, cdp_url=cdp_url)


@dataclass
class BrowserSession:
    """
    Runtime handle to a running browser instance.

    Owned by the task (task creates it, task closes it).
    Shared with the tool via tool.connect(session).

    same-process use:  tool uses session.page directly.
    cross-process use: tool calls playwright.chromium.connect_over_cdp(session.cdp_url).
    """

    page: Page
    browser: Browser
    context: BrowserContext
    playwright: Playwright
    cdp_url: str | None = None  # http://localhost:PORT — pass to connect_over_cdp()

    def stop(self) -> None:
        """Close all browser resources. Called by the task in teardown()."""
        self.context.close()
        self.browser.close()
        self.playwright.stop()
