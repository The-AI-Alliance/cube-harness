from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from cube.resource import InfraConfig

OSWORLD_CUBE_TEST_INFRA_CONFIG_FILE = "OSWORLD_CUBE_TEST_INFRA_CONFIG_FILE"


def load_runtime_infra_from_config_file(
    env_var: str = OSWORLD_CUBE_TEST_INFRA_CONFIG_FILE,
) -> InfraConfig | None:
    """Build an InfraConfig from a JSON config file referenced by ``env_var``.

    Supported shape:
    {
      "class": "package.module:InfraConfigClass",
      "kwargs": {"key": "value"}
    }
    """
    config_path_raw = os.environ.get(env_var)
    if not config_path_raw:
        return None

    config_path = Path(config_path_raw).expanduser()
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise TypeError(f"{env_var} must point to a JSON object")

    class_path = config.get("class")
    if not isinstance(class_path, str) or ":" not in class_path:
        raise ValueError("Infra config JSON must contain 'class' in the form 'package.module:ClassName'")

    kwargs = config.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise TypeError("Infra config JSON field 'kwargs' must be a JSON object")

    module_name, class_name = class_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    infra_cls = getattr(module, class_name)
    infra = infra_cls(**kwargs)
    if not isinstance(infra, InfraConfig):
        raise TypeError(f"{class_path!r} did not construct an InfraConfig instance")
    return infra
