"""
Integration tests for OSWorldTask using the debug action sequences.

Accepts a runtime InfraConfig via:
    OSWORLD_CUBE_TEST_INFRA_CONFIG_FILE=/path/to/infra.json

Falls back to LocalInfraConfig() when none is provided.

Config file shape:
    {
      "class": "package.module:InfraConfigClass",
      "kwargs": {"key": "value"}
    }

Requires an InfraConfig pointing to a provisioned OSWorld VM image.
Run the integration test manually via:
    cube-resources/cube-infra-azure/test_run_debug_agent.py
    cube-resources/cube-infra-aws/test_run_debug_agent.py
"""

import pytest

from cube.resource import InfraConfig
from cube.testing import run_debug_episode
from osworld_cube.debug import get_debug_benchmark, make_debug_agent


@pytest.fixture(scope="session")
def debug_task_configs(infra: InfraConfig):
    benchmark = get_debug_benchmark(infra=infra)
    benchmark.install()
    benchmark.setup()
    configs = {tc.task_id: tc for tc in benchmark.get_task_configs()}
    yield configs
    benchmark.close()


@pytest.mark.integration
def test_debug_episodes(debug_task_configs) -> None:
    for task_id, tc in debug_task_configs.items():
        task = tc.make()
        agent = make_debug_agent(task_id)
        report = run_debug_episode(task, agent, max_steps=20)
        assert report["done"], f"Task {task_id}: episode did not complete: {report}"
        assert report["reward"] > 0, f"Task {task_id}: zero reward: {report}"
        assert not report["error"], f"Task {task_id}: episode error: {report['error']}"
