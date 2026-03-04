# osworld-cube: OSWorld benchmark ported to the CUBE protocol
#
# Minimal public surface for the simple agent loop:
#
#   from osworld_cube import OSWorldTask, Computer13Config, OSWorldBenchmark
#
#   # Simple loop (no harness):
#   task = OSWorldTask(
#       metadata=TaskMetadata(id="task-uuid", abstract_description="Open calculator",
#           extra_info={"domain": "os", "snapshot": "init_state",
#                       "config": [], "evaluator": {}, "related_apps": []}),
#       tool_config=Computer13Config(provider="docker"),
#   )
#   obs, info = task.reset()
#   while not done:
#       action = agent(obs, task.action_set)
#       env_out = task.step(action)      # inherited from cube.task.Task
#       obs, done = env_out.obs, env_out.done
#   task.close()
#
#   # Via benchmark (full run):
#   bench = OSWorldBenchmark(default_tool_config=Computer13Config(provider="docker"))
#   bench.setup()
#   for task_config in bench.get_task_configs():
#       task = task_config.make()
#       ...

from osworld_cube.computer import (
    Computer13,
    ComputerBase,
    ComputerConfig,
    PyAutoGUIComputer,
)
from osworld_cube.task import OSWorldTask
from osworld_cube.benchmark import OSWorldBenchmark, OSWorldTaskConfig
from osworld_cube.debug_agent import get_debug_task_configs, make_debug_agent

__all__ = [
    # Tool classes
    "ComputerBase",
    "Computer13",
    "PyAutoGUIComputer",
    # Config classes
    "ComputerConfig",
    # Task / benchmark
    "OSWorldTask",
    "OSWorldBenchmark",
    "OSWorldTaskConfig",
    # Debug helpers
    "get_debug_task_configs",
    "make_debug_agent",
]
