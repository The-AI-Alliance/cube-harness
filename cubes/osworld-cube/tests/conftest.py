from __future__ import annotations

import logging

import pytest

from cube import LocalInfraConfig
from cube.resource import InfraConfig
from osworld_cube.infra_loader import load_runtime_infra_from_config_file

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def infra() -> InfraConfig:
    """Resolve integration-test infra, defaulting to LocalInfraConfig."""
    resolved = load_runtime_infra_from_config_file() or LocalInfraConfig()
    message = f"[osworld tests] Using infra: {type(resolved).__name__} (fingerprint={resolved.fingerprint()})"
    logger.info(message)
    print(message)
    return resolved
