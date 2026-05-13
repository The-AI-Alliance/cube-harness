"""Unit tests for cube_harness.infra_profile.

Verifies the TypedBaseModel-based resolver: no per-kind branches, just
`InfraConfig.model_validate(profile_dict)` with `_type` for discrimination.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cube_harness import infra_profile


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect CONFIG_PATH to a tmp file for the duration of the test."""
    cfg = tmp_path / "infra.json"
    monkeypatch.setattr(infra_profile, "CONFIG_PATH", cfg)
    monkeypatch.delenv("CUBE_INFRA", raising=False)
    return cfg


def test_load_local_default_no_config_file(tmp_config: Path) -> None:
    """`load_infra()` with no config file falls back to LocalInfraConfig."""
    from cube.infra_local import LocalInfraConfig

    infra = infra_profile.load_infra()
    assert isinstance(infra, LocalInfraConfig)


def test_load_local_explicit_name(tmp_config: Path) -> None:
    from cube.infra_local import LocalInfraConfig

    infra = infra_profile.load_infra("local")
    assert isinstance(infra, LocalInfraConfig)


def test_load_via_typed_basemodel_discrimination(tmp_config: Path) -> None:
    """The `_type` field on TypedBaseModel auto-instantiates the right class."""
    from cube.infra_local import LocalInfraConfig

    tmp_config.write_text(json.dumps({"my-local": {"_type": "cube.infra_local.LocalInfraConfig"}}))
    infra = infra_profile.load_infra("my-local")
    assert isinstance(infra, LocalInfraConfig)


def test_missing_profile_raises(tmp_config: Path) -> None:
    tmp_config.write_text(json.dumps({"yul101": {"_type": "cube.infra_local.LocalInfraConfig"}}))
    with pytest.raises(KeyError, match="absent_profile"):
        infra_profile.load_infra("absent_profile")


def test_unknown_type_raises(tmp_config: Path) -> None:
    """Pydantic surfaces a clear error for an unknown `_type`."""
    tmp_config.write_text(json.dumps({"weird": {"_type": "nonexistent.module.WhateverConfig"}}))
    with pytest.raises((ImportError, ValueError, ModuleNotFoundError)):
        infra_profile.load_infra("weird")


def test_env_var_picks_default_profile(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CUBE_INFRA env var picks the profile when no explicit name is passed."""
    from cube.infra_local import LocalInfraConfig

    tmp_config.write_text(json.dumps({"alt": {"_type": "cube.infra_local.LocalInfraConfig"}}))
    monkeypatch.setenv("CUBE_INFRA", "alt")
    infra = infra_profile.load_infra()
    assert isinstance(infra, LocalInfraConfig)


def test_explicit_name_overrides_env_var(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cube.infra_local import LocalInfraConfig

    tmp_config.write_text(
        json.dumps(
            {
                "a": {"_type": "cube.infra_local.LocalInfraConfig"},
                "b": {"_type": "cube.infra_local.LocalInfraConfig"},
            }
        )
    )
    monkeypatch.setenv("CUBE_INFRA", "a")
    infra = infra_profile.load_infra("b")
    assert isinstance(infra, LocalInfraConfig)
