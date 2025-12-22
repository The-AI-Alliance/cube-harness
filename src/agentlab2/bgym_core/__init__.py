"""
BrowserGym Core - Minimal core for BrowserEnv.

This is a vendored subset of the browsergym-core package, containing only the
essential components needed to run BrowserEnv in AgentLab2.

Original source: https://github.com/ServiceNow/BrowserGym
License: Apache-2.0
"""

__version__ = "0.14.3.dev1"

import playwright.sync_api

# we use a global playwright instance
_PLAYWRIGHT = None


def _set_global_playwright(pw: playwright.sync_api.Playwright) -> None:
    global _PLAYWRIGHT
    _PLAYWRIGHT = pw


def _get_global_playwright() -> playwright.sync_api.Playwright:
    global _PLAYWRIGHT
    if not _PLAYWRIGHT:
        pw = playwright.sync_api.sync_playwright().start()
        _set_global_playwright(pw)

    return _PLAYWRIGHT 

# register the open-ended task
from .registration import register_task #noqa
from .task import OpenEndedTask #noqa #

register_task(OpenEndedTask.get_task_id(), OpenEndedTask)
