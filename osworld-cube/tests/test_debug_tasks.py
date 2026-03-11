"""Integration test: run OSWorld debug tasks with the scripted agent.

Requires Docker + /dev/kvm + OSWorld qcow2 (~23 GB download on first run).
Run with:

    uv run pytest tests/test_debug_tasks.py -m integration -v -s
"""

import pytest


@pytest.mark.integration
def test_debug_tasks():
    """All OSWorld debug tasks must complete with reward == 1.0."""
    import osworld_cube.debug as mod
    from cube.testing import assert_debug_tasks_reward_one

    assert_debug_tasks_reward_one(mod, max_steps=20)
