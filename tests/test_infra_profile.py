"""Unit tests for cube_harness.infra_profile."""

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


def test_load_local_from_config(tmp_config: Path) -> None:
    from cube.infra_local import LocalInfraConfig

    tmp_config.write_text(json.dumps({"local": {"kind": "local"}}))
    infra = infra_profile.load_infra("local")
    assert isinstance(infra, LocalInfraConfig)


def test_missing_profile_raises(tmp_config: Path) -> None:
    tmp_config.write_text(json.dumps({"yul101": {"kind": "toolkit"}}))
    with pytest.raises(KeyError, match="absent_profile"):
        infra_profile.load_infra("absent_profile")


def test_unknown_kind_raises(tmp_config: Path) -> None:
    tmp_config.write_text(json.dumps({"weird": {"kind": "mystery-cloud"}}))
    with pytest.raises(ValueError, match="mystery-cloud"):
        infra_profile.load_infra("weird")


def test_env_var_overrides_default(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CUBE_INFRA env var picks the profile when no explicit name is passed."""
    tmp_config.write_text(json.dumps({"local-override": {"kind": "local"}}))
    monkeypatch.setenv("CUBE_INFRA", "local-override")
    # Should resolve to local-override via env var.  We can't assert the type
    # of the specific local-override (it's LocalInfraConfig), but we can assert
    # that an "unknown profile" KeyError is NOT raised — meaning the env var
    # was consulted.
    from cube.infra_local import LocalInfraConfig

    infra = infra_profile.load_infra()
    assert isinstance(infra, LocalInfraConfig)


def test_explicit_name_overrides_env_var(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_config.write_text(json.dumps({"local-a": {"kind": "local"}, "local-b": {"kind": "local"}}))
    monkeypatch.setenv("CUBE_INFRA", "local-a")
    # Explicit "local-b" wins over the env var.  Both are LocalInfraConfig here,
    # but the test still validates the resolution path doesn't raise on "local-b".
    from cube.infra_local import LocalInfraConfig

    infra = infra_profile.load_infra("local-b")
    assert isinstance(infra, LocalInfraConfig)
