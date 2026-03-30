"""
Integration tests for OSWorldTask using the debug action sequences.

Requires an InfraConfig pointing to a provisioned OSWorld VM image.
Run the integration test manually via:
    cube-resources/cube-infra-azure/test_run_debug_agent.py
    cube-resources/cube-infra-aws/test_run_debug_agent.py

These tests are skipped by default — pass --run-integration to enable them.
"""

import pytest

from osworld_cube.debug import get_debug_benchmark, make_debug_agent


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-integration", action="store_true", default=False)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: mark test as requiring a live InfraConfig")


def _get_infra():
    """Return an InfraConfig from env if available, else None."""
    import os

    provider = os.environ.get("CUBE_TEST_INFRA_PROVIDER")
    if provider == "azure":
        from cube_infra_azure import AzureInfraConfig

        return AzureInfraConfig()
    if provider == "aws":
        from cube_infra_aws import AWSInfraConfig

        return AWSInfraConfig()
    return None


_infra = _get_infra()
_benchmark = get_debug_benchmark(infra=_infra)
_benchmark.install()
_benchmark.setup()
_DEBUG_TASK_CONFIGS = {tc.task_id: tc for tc in _benchmark.get_task_configs()}


@pytest.mark.integration
@pytest.mark.skipif(_infra is None, reason="Set CUBE_TEST_INFRA_PROVIDER=azure|aws to run")
@pytest.mark.parametrize("task_id", list(_DEBUG_TASK_CONFIGS))
def test_debug_episode(task_id: str) -> None:
    from cube.testing import run_debug_episode as _run

    task = _DEBUG_TASK_CONFIGS[task_id].make()
    agent = make_debug_agent(task_id)
    report = _run(task, agent, max_steps=20)
    assert report["done"], f"Episode did not complete: {report}"
    assert report["reward"] > 0, f"Zero/negative reward: {report}"
    assert not report["error"], f"Episode error: {report['error']}"
