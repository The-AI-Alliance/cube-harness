"""
Integration test for OSWorldTask against a real desktop_env Docker VM.

Requires:
  - Docker running and the OSWorld VM image available
  - desktop_env installed
  - CUBE_CACHE_DIR set (or default ~/.cube used for VM storage)

Run with:
  pytest tests/test_run_osworld.py -s -v
  (the -s flag shows the 60s stabilisation wait progress in logs)
"""

from __future__ import annotations

from cube.core import ImageContent, Observation, TextContent
from cube.task import TaskMetadata
from osworld_cube.vm_backend import OSWorldQEMUVMBackend
from osworld_cube.computer import ComputerConfig
from osworld_cube.task import OSWorldTask


def test_instantiate_and_get_first_obs():
    metadata = TaskMetadata(
        id="demo-open-calculator",
        abstract_description="Open the Calculator application",
        extra_info={
            "domain": "os",
            "snapshot": "init_state",
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
    vm_backend = OSWorldQEMUVMBackend()

    task = OSWorldTask(metadata=metadata, tool_config=tool_config, vm_backend=vm_backend)

    try:
        # task.id and action_set populated immediately after construction
        assert task.id == "demo-open-calculator"
        action_names = {a.name for a in task.action_set}
        for expected in ("click", "typing", "hotkey", "done", "fail"):
            assert expected in action_names

        # reset() returns (Observation, info) — blocks ~60s for VM stabilisation
        obs, info = task.reset()

        assert isinstance(obs, Observation)
        # instruction text is prepended as the first content item
        texts = [c.data for c in obs.contents if isinstance(c, TextContent)]
        assert any("Open the Calculator" in t for t in texts)
        # screenshot is included as an ImageContent
        images = [c for c in obs.contents if isinstance(c, ImageContent)]
        assert len(images) >= 1

        assert info["task_id"] == "demo-open-calculator"
        assert info["task_domain"] == "os"
    finally:
        task.close()
