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
    "fileexplorer",
    "general",
    "gimp",
    "impress",
    "info",
    "misc",
    "replay",
    "vlc",
    "vscode",
    "calc",
    "microsoftpaint",
    "windows_clock",
    "edge",
    "settings",
    "msedge",
)


def __getattr__(name: str) -> Any:
    import_errors: dict[str, str] = {}
    for sub in _SUBMODULES:
        try:
            mod = importlib.import_module(f".{sub}", __package__)
        except ImportError as exc:
            import_errors[sub] = str(exc)
            continue
        if hasattr(mod, name):
            return getattr(mod, name)
    if import_errors:
        detail = "; ".join(f"{s}: {e}" for s, e in import_errors.items())
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}. "
            f"Some submodules failed to import (may be the cause): {detail}"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
