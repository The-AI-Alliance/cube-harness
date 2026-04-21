"""OSWorld task hints and precision for the Genny agent.

Exports two dicts consumed by GennyConfig:

- OSWORLD_TASK_PRECISION: task_id -> text that clarifies an under-defined goal.
  Injected as part of the goal context ("Additional task details").

- OSWORLD_TASK_HINTS: task_id -> domain guidance that helps the LLM work
  faster/better. Injected as a separate "Task Hint" section.

Usage:
    from osworld_hints import OSWORLD_TASK_HINTS, OSWORLD_TASK_PRECISION

    agent_config = GennyConfig(
        ...
        task_hints=OSWORLD_TASK_HINTS,
        task_precision=OSWORLD_TASK_PRECISION,
    )
"""

# ---------------------------------------------------------------------------
# Task precision — the goal description is under-defined
# ---------------------------------------------------------------------------

OSWORLD_TASK_PRECISION: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Task-specific hints — domain guidance the LLM needs but should eventually
# learn through trial and error
# ---------------------------------------------------------------------------

OSWORLD_TASK_HINTS: dict[str, str] = {}
