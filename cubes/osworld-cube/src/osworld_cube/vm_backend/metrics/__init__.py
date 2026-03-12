"""Metrics package — lazy loading to avoid heavy evaluation deps at import time.

Metric functions are resolved on first access via module-level ``__getattr__``,
which imports the relevant submodule only when an attribute is actually
requested.  This allows the vm_backend to be imported without requiring the
full evaluation dependency stack.

The ``infeasible`` sentinel is kept here as a no-op; infeasible task detection
is handled by the evaluator as a special case before metrics are dispatched.
"""

import importlib
from typing import Any

_SUBMODULES = (
    "general",
    "basic_os",
    "chrome",
    "docs",
    "gimp",
    "libreoffice",
    "others",
    "pdf",
    "slides",
    "table",
    "thunderbird",
    "vlc",
    "vscode",
)


def __getattr__(name: str) -> Any:
    for sub in _SUBMODULES:
        try:
            mod = importlib.import_module(f".{sub}", __package__)
        except ImportError:
            continue
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def infeasible() -> None:
    """Sentinel — infeasible task handling is a special case in the evaluator."""
