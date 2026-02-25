"""
Sandbox script: BrowserSession pattern playground.

Run with:
    uv run python src/agentlab2/tools/_sandbox/test_browser_session.py

Three demos:
  1. Same-process  — tool receives session.page directly.
  2. Cross-process — tool worker runs in a subprocess, connects via CDP URL.
  3. Shared state  — subprocess sees DOM mutations made by the task's page.
"""

import multiprocessing as mp
import socket

from playwright.sync_api import Page

from agentlab2.core import Action, ActionSchema, Content, Observation, Task
from agentlab2.tool import AbstractTool
from agentlab2.tools._sandbox.browser_session import (
    BrowserSession,
    BrowserSessionConfig,
)

_HTML = "<h1 id='title'>Hello Browser</h1><button id='btn'>Click me</button><p id='out'></p>"
_DATA_URL = f"data:text/html,{_HTML}"

DIVIDER = "-" * 60


def find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class SimplePageTask(Task):
    """Task that owns the browser: starts it in setup(), closes it in teardown()."""

    id = "simple_page_task"
    validate_per_step = False

    def __init__(self, url: str, browser_config: BrowserSessionConfig | None = None) -> None:
        self.url = url
        self.browser_config = browser_config or BrowserSessionConfig()
        self._session: BrowserSession | None = None

    def setup(self, tool: "SimplePageTool") -> tuple[Observation, dict]:  # type: ignore[override]
        print(f"  [task] starting browser (cdp_port={self.browser_config.cdp_port})")
        self._session = self.browser_config.start()
        self._session.page.goto(self.url)
        print("  [task] page loaded — calling tool.connect(session)")
        tool.connect(self._session)
        heading = self._session.page.inner_text("h1")
        return Observation.from_text(f"Page loaded. Heading: {heading}"), {"url": self.url}

    def teardown(self) -> None:
        if self._session:
            print("  [task] tearing down browser")
            self._session.stop()
            self._session = None

    def validate_task(self, obs: Observation) -> tuple[float, dict]:
        return 1.0, {"done": True}

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        return actions


# ---------------------------------------------------------------------------
# Same-process tool — connects via session.page
# ---------------------------------------------------------------------------


class SimplePageTool(AbstractTool):
    """Browser tool that is inert until task.setup() calls tool.connect(session)."""

    def __init__(self) -> None:
        self._page: Page | None = None

    def connect(self, session: BrowserSession) -> None:
        """Receive the page from the task. Same-process: store the reference directly."""
        self._page = session.page
        print(f"  [tool] connected via page object — url: {self._page.url}")

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Tool not connected. Call task.setup(tool) first.")
        return self._page

    # actions ---------------------------------------------------------------

    def get_heading(self) -> str:
        """Return the text of the first <h1> on the page."""
        return self.page.inner_text("h1")

    def click(self, selector: str) -> str:
        """Click an element by CSS selector."""
        self.page.click(selector)
        return f"clicked {selector!r}"

    def get_url(self) -> str:
        """Return the current page URL."""
        return self.page.url

    # AbstractTool ----------------------------------------------------------

    @property
    def action_set(self) -> list[ActionSchema]:
        return [
            ActionSchema.from_function(self.get_heading),
            ActionSchema.from_function(self.click),
            ActionSchema.from_function(self.get_url),
        ]

    def execute_action(self, action: Action) -> Observation:
        fn = getattr(self, action.name, None)
        if fn is None:
            raise ValueError(f"Unknown action: {action.name!r}")
        result = fn(**action.arguments) or "OK"
        return Observation(contents=[Content(data=str(result), tool_call_id=action.id)])


# ---------------------------------------------------------------------------
# Cross-process CDP worker
#
# In production: the task sends cdp_url (a plain string) over IPC to a tool
# process. The tool process creates its OWN playwright instance and connects.
# Here we use multiprocessing to simulate that faithfully.
# ---------------------------------------------------------------------------


def _cdp_worker(cdp_url: str, js_to_eval: str | None, queue: "mp.Queue[dict]") -> None:
    """
    Runs in a SEPARATE Python process — has its own Playwright event loop.
    Connects to the browser via CDP, reads page state, puts results in queue.
    """
    from playwright.sync_api import sync_playwright  # local import: runs in subprocess

    pw = sync_playwright().start()
    try:
        browser = pw.chromium.connect_over_cdp(cdp_url)
        page = browser.contexts[0].pages[0]
        result: dict = {
            "heading": page.inner_text("h1"),
            "url": page.url,
        }
        if js_to_eval:
            result["js_result"] = str(page.evaluate(js_to_eval))
        browser.close()
    finally:
        pw.stop()
    queue.put(result)


def run_cdp_worker(cdp_url: str, js_to_eval: str | None = None) -> dict:
    """Spawn a subprocess, run the CDP worker, return its results."""
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_cdp_worker, args=(cdp_url, js_to_eval, queue))
    proc.start()
    proc.join(timeout=15)
    if proc.exitcode != 0:
        raise RuntimeError(f"CDP worker process exited with code {proc.exitcode}")
    return queue.get_nowait()


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------


def demo_same_process() -> None:
    print(DIVIDER)
    print("DEMO 1 — Same-process connection (session.page)")
    print(DIVIDER)

    tool = SimplePageTool()
    task = SimplePageTask(url=_DATA_URL)

    print(f"  tool._page before setup: {tool._page}")

    obs, info = task.setup(tool)
    print(f"  initial obs: {obs.contents[0].data}")

    print("\n  executing actions via tool.execute_action():")
    for action in [
        Action(name="get_heading", arguments={}),
        Action(name="click", arguments={"selector": "#btn"}),
        Action(name="get_url", arguments={}),
    ]:
        obs = tool.execute_action(action)
        print(f"    {action.name:15} -> {obs.contents[0].data!r}")

    print(f"\n  action_set names (no Protocol): {[s.name for s in tool.action_set]}")

    task.teardown()
    print(f"  task._session after teardown: {task._session}")
    print()


def demo_cdp_cross_process() -> None:
    print(DIVIDER)
    print("DEMO 2 — Cross-process connection via Chrome CDP (subprocess)")
    print(DIVIDER)

    port = find_free_port()
    print(f"  using CDP port: {port}")

    # Task starts browser with remote debugging enabled.
    # The tool (in a real scenario) only receives the cdp_url string — not the page object.
    tool = SimplePageTool()  # placeholder to satisfy setup() signature
    task = SimplePageTask(url=_DATA_URL, browser_config=BrowserSessionConfig(cdp_port=port))
    task.setup(tool)

    cdp_url = task._session.cdp_url
    print(f"  [main] task exposes cdp_url: {cdp_url}")
    print("  [main] spawning subprocess with its own playwright…")

    result = run_cdp_worker(cdp_url)

    print(f"  [subprocess] heading -> {result['heading']!r}")
    print(f"  [subprocess] url     -> {result['url']!r}")

    task.teardown()
    print()


def demo_cdp_shared_state() -> None:
    print(DIVIDER)
    print("DEMO 3 — Subprocess sees DOM mutations made by the task's page")
    print(DIVIDER)

    port = find_free_port()
    print(f"  using CDP port: {port}")

    tool = SimplePageTool()
    task = SimplePageTask(url=_DATA_URL, browser_config=BrowserSessionConfig(cdp_port=port))
    task.setup(tool)

    # Task mutates DOM via its own page reference (same process)
    mutation = "written by task process"
    print(f"  [task] setting #out text to {mutation!r} via session.page")
    task._session.page.evaluate(f"document.getElementById('out').textContent = '{mutation}'")

    # Subprocess reads the mutation over CDP
    result = run_cdp_worker(
        task._session.cdp_url,
        js_to_eval="document.getElementById('out').textContent",
    )
    print(f"  [subprocess] reads #out -> {result['js_result']!r}")
    assert result["js_result"] == mutation, "subprocess should see the same live DOM"
    print("  ✓ cross-process CDP sees the same live DOM state")

    task.teardown()
    print()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    demo_same_process()
    demo_cdp_cross_process()
    demo_cdp_shared_state()

    print("All demos completed.")
