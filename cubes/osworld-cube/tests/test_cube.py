import pytest

from osworld_cube.vm_backend import OSWorldQEMUVMBackend
from osworld_cube.debug import get_debug_benchmark, make_debug_agent

_benchmark = get_debug_benchmark(vm_backend=OSWorldQEMUVMBackend())
_benchmark.install()
_benchmark.setup()
_DEBUG_TASK_CONFIGS = {tc.task_id: tc for tc in _benchmark.get_task_configs()}


def run_debug_episode(task_id: str, max_steps: int = 20) -> dict:
    """
    Run a debug episode for an OSWorld task and return a minimal report dict.

    Delegates to the generic ``cube.testing.run_debug_episode`` harness.
    The report schema matches the stress-test MVP output (stress_test_specs.md §3.1).

    Args:
        task_id:    ID of the debug task (must be in debug_tasks.json).
        max_steps:  Safety cap on the step loop (default 20).

    Returns:
        dict with keys: task_id, done, reward, steps, episode_time_s,
        step_times_s, error.
    """
    from cube.testing import run_debug_episode as _run

    task = _DEBUG_TASK_CONFIGS[task_id].make()
    agent = make_debug_agent(task_id)
    return _run(task, agent, max_steps=max_steps)


@pytest.mark.parametrize("task_id", list(_DEBUG_TASK_CONFIGS))
def test_debug_episode(task_id: str) -> None:
    report = run_debug_episode(task_id)
    assert report["done"], f"Episode did not complete: {report}"
    assert report["reward"] > 0, f"Zero/negative reward: {report}"
    assert not report["error"], f"Episode error: {report['error']}"
