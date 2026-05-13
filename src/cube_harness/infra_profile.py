"""Named-infra-profile resolver — `~/.cube/infra.json` → `InfraConfig`.

Each entry is an ``InfraConfig.model_validate(...)`` payload (a ``_type``-tagged
``TypedBaseModel`` dict). Resolution order: explicit ``name`` arg → ``$CUBE_INFRA``
env var → literal ``"local"`` (which falls back to ``LocalInfraConfig()`` even
with no config file).
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
