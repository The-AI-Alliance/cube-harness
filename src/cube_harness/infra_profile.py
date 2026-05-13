"""Named-infra-profile resolver — `~/.cube/infra.json` → `InfraConfig`.

Lets recipes accept a single ``--infra <name>`` flag instead of a per-recipe
mix of ``--toolkit / --daytona / --eai-profile / --eai-path / --preemptable``
boolean knobs. Per-profile fields live in ``~/.cube/infra.json`` so the choice
of infra (and its parameters) is local to each developer's machine.

File format
-----------

``~/.cube/infra.json`` is a flat dict mapping profile name → spec. Each spec
has a ``"kind"`` field (``"local" | "toolkit" | "daytona"``) plus any
fields accepted by the corresponding ``InfraConfig`` constructor::

    {
      "local":      {"kind": "local"},
      "yul101":     {"kind": "toolkit", "profile": "yul101", "eai_path": "eai"},
      "yul101-pre": {"kind": "toolkit", "profile": "yul101", "preemptable": true},
      "daytona":    {"kind": "daytona"}
    }

Resolution order
----------------

1. Explicit ``name`` arg (e.g. recipe's ``--infra <name>``).
2. ``$CUBE_INFRA`` env var.
3. The literal ``"local"`` — works even with no config file.

If the resolved name is missing from ``~/.cube/infra.json`` (or the file doesn't
exist at all), ``load_infra`` raises ``KeyError`` unless the name is ``"local"``,
in which case a default ``LocalInfraConfig()`` is returned.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_PATH: Path = Path("~/.cube/infra.json").expanduser()


def load_infra(name: str | None = None) -> Any:
    """Resolve a named infra profile to an ``InfraConfig``.

    The return type is ``Any`` because the concrete ``InfraConfig`` subclass
    (``LocalInfraConfig`` / ``ToolkitInfraConfig`` / ``DaytonaInfraConfig``)
    depends on which extras are installed. Recipes get back an object that
    duck-types as ``cube.resource.InfraConfig``.
    """
    name = name or os.environ.get("CUBE_INFRA") or "local"

    profiles: dict[str, dict[str, Any]] = {}
    if CONFIG_PATH.exists():
        profiles = json.loads(CONFIG_PATH.read_text())

    if name not in profiles:
        if name == "local":
            from cube.infra_local import LocalInfraConfig

            return LocalInfraConfig()
        raise KeyError(
            f"infra profile {name!r} not found in {CONFIG_PATH}. "
            f"Available: {sorted(profiles) or '(empty file)'}. "
            f"Add a profile or pass --infra local."
        )

    spec = dict(profiles[name])
    kind = spec.pop("kind", None)
    if kind == "local":
        from cube.infra_local import LocalInfraConfig

        return LocalInfraConfig(**spec)
    if kind == "toolkit":
        from cube_infra_toolkit import ToolkitInfraConfig

        return ToolkitInfraConfig(**spec)
    if kind == "daytona":
        from cube_infra_daytona import DaytonaInfraConfig

        return DaytonaInfraConfig(**spec)
    raise ValueError(f"infra profile {name!r}: unknown kind {kind!r}. Expected local | toolkit | daytona.")
