"""Getters package — lazy loading to avoid heavy evaluation deps at import time.

Functions are resolved on first access via module-level ``__getattr__``, which
imports the relevant submodule only when an attribute is actually requested.
This allows the vm_backend to be imported (e.g. for the QEMU manager or guest
agent) without requiring the full evaluation dependency stack.
"""

import importlib
from typing import Any

_SUBMODULES = (
    "chrome",
    "file",
    "general",
    "gimp",
    "impress",
    "info",
    "misc",
    "replay",
    "vlc",
    "vscode",
    "calc",
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
