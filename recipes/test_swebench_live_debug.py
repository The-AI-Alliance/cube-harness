"""Test SWE-bench Live cube: run debug agent on known tasks via cube.testing pipeline.

Usage
-----
    DAYTONA_API_KEY=... uv run recipes/test_swebench_live_debug.py
"""

import logging
import sys

from cube.backends.daytona import DaytonaContainerBackend
from cube.testing import run_debug_episode
from swebench_live_cube.benchmark import SWEBenchLiveBenchmark
from swebench_live_cube.debug import _TASK_ACTIONS, get_debug_task_configs, make_debug_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

backend = DaytonaContainerBackend()

# Load only the debug task instances
bench = SWEBenchLiveBenchmark(
    container_backend=backend,
    instance_ids=list(_TASK_ACTIONS.keys()),
    split="lite",
)
bench.install()
bench.setup()

configs = get_debug_task_configs()
if not configs:
    logger.error("No debug task configs found — are the tasks in the dataset?")
    sys.exit(1)

logger.info(f"Running {len(configs)} debug episodes")

results = []
for tc in configs:
    logger.info(f"--- Starting task: {tc.task_id} ---")
    task = tc.make(container_backend=backend)
    agent = make_debug_agent(tc.task_id)
    report = run_debug_episode(task, agent)
    results.append(report)
    logger.info(
        f"Task {tc.task_id}: done={report['done']}, reward={report['reward']}, "
        f"steps={report['steps']}, error={report['error']}"
    )

failed = [r for r in results if r["error"]]
if failed:
    logger.error(f"{len(failed)} episode(s) had errors:")
    for r in failed:
        logger.error(f"  {r['task_id']}: {r['error']}")
else:
    logger.info(f"All {len(results)} episodes completed without errors")

sys.exit(1 if failed else 0)
