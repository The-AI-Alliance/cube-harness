"""Named-infra-profile resolver — `~/.cube/infra.json` → `InfraConfig`.

Lets recipes accept a single ``--infra <name>`` flag instead of a per-recipe
mix of ``--toolkit / --daytona / --eai-profile / --eai-path / --preemptable``
boolean knobs. Per-profile fields live in ``~/.cube/infra.json`` so the choice
of infra (and its parameters) is local to each developer's machine.

File format
-----------

``~/.cube/infra.json`` maps profile name → ``InfraConfig`` payload. Each value
is whatever ``InfraConfig.model_validate(...)`` accepts: a dict with the
``_type`` field (fully-qualified class path) plus any fields the concrete
subclass takes. Since every InfraConfig is a ``TypedBaseModel``, the ``_type``
tag auto-discriminates — no per-kind branch lives here::

    {
      "yul101": {
        "_type": "cube_infra_toolkit.toolkit.ToolkitInfraConfig",
        "profile": "yul101",
        "eai_path": "eai"
      },
      "yul101-preempt": {
        "_type": "cube_infra_toolkit.toolkit.ToolkitInfraConfig",
        "profile": "yul101",
        "preemptable": true
      },
      "daytona": {
        "_type": "cube_infra_daytona.daytona.DaytonaInfraConfig"
      }
    }

Resolution order
----------------

1. Explicit ``name`` arg (e.g. recipe's ``--infra <name>``).
2. ``$CUBE_INFRA`` env var.
3. Literal ``"local"`` — falls back to ``LocalInfraConfig()`` even with no
   config file, so a fresh checkout runs without setup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cube.resource import InfraConfig

CONFIG_PATH: Path = Path("~/.cube/infra.json").expanduser()


def load_infra(name: str | None = None) -> InfraConfig:
    """Resolve a named infra profile to a concrete ``InfraConfig``."""
    name = name or os.environ.get("CUBE_INFRA") or "local"

    profiles: dict[str, dict] = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    if name not in profiles:
        if name == "local":
            from cube.infra_local import LocalInfraConfig

            return LocalInfraConfig()
        raise KeyError(
            f"infra profile {name!r} not found in {CONFIG_PATH}. "
            f"Available: {sorted(profiles) or '(empty file)'}. "
            f"Add a profile or pass --infra local."
        )
    return InfraConfig.model_validate(profiles[name])
