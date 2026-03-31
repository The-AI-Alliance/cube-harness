"""
Integration test for OSWorldTask against a live VM via InfraConfig.

Requires:
  - CUBE_TEST_INFRA_PROVIDER=azure|aws (and matching credentials)
  - A provisioned osworld-ubuntu-vm image for that infra

Run with:
  CUBE_TEST_INFRA_PROVIDER=azure pytest tests/test_run_osworld.py -s -v
  (the -s flag shows the 60s stabilisation wait progress in logs)
"""

from __future__ import annotations

import os

import pytest

from cube.core import ImageContent, Observation, TextContent
from cube.task import TaskMetadata
from osworld_cube.computer import ComputerConfig
from osworld_cube.task import OSWorldTask


def _get_infra():
    provider = os.environ.get("CUBE_TEST_INFRA_PROVIDER")
    if provider == "azure":
        from cube_infra_azure import AzureInfraConfig

        return AzureInfraConfig()
    if provider == "aws":
        from cube_infra_aws import AWSInfraConfig

        return AWSInfraConfig()
    return None


_infra = _get_infra()
pytestmark = pytest.mark.skipif(_infra is None, reason="Set CUBE_TEST_INFRA_PROVIDER=azure|aws to run")


def test_instantiate_and_get_first_obs():
    metadata = TaskMetadata(
        id="demo-open-calculator",
        abstract_description="Open the Calculator application",
        extra_info={
            "domain": "os",
            "config": [],
            "evaluator": {"func": "infeasible"},
            "related_apps": ["gnome-calculator"],
        },
    )
    tool_config = ComputerConfig(
        headless=True,
        require_a11y_tree=True,
        observe_after_action=False,
    )

    task = OSWorldTask(metadata=metadata, tool_config=tool_config, infra=_infra)

    try:
        assert task.id == "demo-open-calculator"
        action_names = {a.name for a in task.action_set}
        for expected in ("click", "typing", "hotkey", "done", "fail"):
            assert expected in action_names

        obs, info = task.reset()

        assert isinstance(obs, Observation)
        texts = [c.data for c in obs.contents if isinstance(c, TextContent)]
        assert any("Open the Calculator" in t for t in texts)
        images = [c for c in obs.contents if isinstance(c, ImageContent)]
        assert len(images) >= 1

        assert info["task_id"] == "demo-open-calculator"
        assert info["task_domain"] == "os"
    finally:
        task.close()
