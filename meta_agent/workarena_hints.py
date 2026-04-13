"""WorkArena task hints and precision for the Genny agent.

Exports two dicts consumed by GennyConfig:

- WORKARENA_TASK_PRECISION: task_id -> text that clarifies an under-defined goal.
  These belong in the task description but are maintained here until WorkArena upstream
  is updated. Injected as part of the goal context ("Additional task details").

- WORKARENA_TASK_HINTS: task_id -> general or task-specific guidance that helps the LLM
  work faster/better. A strong LLM should eventually solve the task without these, but
  they reduce wasted turns. Injected as a separate "Task Hint" section.

Usage:
    from workarena_hints import WORKARENA_TASK_HINTS, WORKARENA_TASK_PRECISION

    agent_config = GennyConfig(
        ...
        task_hints=WORKARENA_TASK_HINTS,
        task_precision=WORKARENA_TASK_PRECISION,
    )
"""

# ---------------------------------------------------------------------------
# Task precision — the goal description is under-defined
# ---------------------------------------------------------------------------
# These fix cases where a competent LLM cannot know what is expected without
# extra information that the task goal should have provided.

# Sort: goal says "sort by X" but doesn't say to use the filter UI.
# Column header clicks look correct but don't update the sysparm_query URL
# parameter that WorkArena's verifier checks.
_SORT_PRECISION: str = (
    "Use the filter panel (funnel icon) to set sort order — "
    "do NOT click column headers. Column header sorting is not reflected "
    "in the validated configuration."
)

# Filter: goal says "filter by X" but doesn't describe the filter UI workflow.
_FILTER_PRECISION: str = (
    "Use the filter panel (funnel icon) to add filter conditions."
)

# Chart (single/multi value): goal asks a question but doesn't specify the answer format.
_CHART_VALUE_PRECISION: str = (
    "Answer with ONLY the numeric value — no units, labels, or explanation. "
    "Example: send_message('42.5')"
)

# Chart min/max (multi): answer format is 'label, count', not just a number.
_CHART_MINMAX_PRECISION: str = (
    "Answer with both the label and the count, comma-separated. "
    "Example: send_message('2026-01-19, 18')"
)

# Create: goal lists fields but doesn't emphasize ALL must be filled, and doesn't
# mention that submit_form() is the correct submission method.
_CREATE_PRECISION: str = (
    "Fill in EVERY field specified in the goal — missing any field causes failure. "
    "When all fields are set, call submit_form() to submit. "
    "Do NOT click the visible Submit button — it navigates away without saving."
)

WORKARENA_TASK_PRECISION: dict[str, str] = {
    # Create (5)
    "workarena.servicenow.create-incident": _CREATE_PRECISION,
    "workarena.servicenow.create-hardware-asset": _CREATE_PRECISION,
    "workarena.servicenow.create-change-request": _CREATE_PRECISION,
    "workarena.servicenow.create-user": _CREATE_PRECISION,
    "workarena.servicenow.create-problem": _CREATE_PRECISION,
    # Chart (4)
    "workarena.servicenow.single-chart-value-retrieval": _CHART_VALUE_PRECISION,
    "workarena.servicenow.single-chart-min-max-retrieval": _CHART_VALUE_PRECISION,
    "workarena.servicenow.multi-chart-value-retrieval": _CHART_VALUE_PRECISION,
    "workarena.servicenow.multi-chart-min-max-retrieval": _CHART_MINMAX_PRECISION,
    # Sort (6)
    "workarena.servicenow.sort-asset-list": _SORT_PRECISION,
    "workarena.servicenow.sort-change-request-list": _SORT_PRECISION,
    "workarena.servicenow.sort-hardware-list": _SORT_PRECISION,
    "workarena.servicenow.sort-incident-list": _SORT_PRECISION,
    "workarena.servicenow.sort-service-catalog-item-list": _SORT_PRECISION,
    "workarena.servicenow.sort-user-list": _SORT_PRECISION,
    # Filter (6)
    "workarena.servicenow.filter-asset-list": _FILTER_PRECISION,
    "workarena.servicenow.filter-change-request-list": _FILTER_PRECISION,
    "workarena.servicenow.filter-hardware-list": _FILTER_PRECISION,
    "workarena.servicenow.filter-incident-list": _FILTER_PRECISION,
    "workarena.servicenow.filter-service-catalog-item-list": _FILTER_PRECISION,
    "workarena.servicenow.filter-user-list": _FILTER_PRECISION,
}


# ---------------------------------------------------------------------------
# Task-specific hints — LLM guidance that reduces wasted turns
# ---------------------------------------------------------------------------
# A strong LLM should eventually figure these out through trial and error,
# but they prevent common failure modes (e.g. retrying a non-interactive element
# 15 times, using fill() on an autocomplete field).

# Sort: step-by-step filter UI interaction pattern.
_SORT_HINT: str = (
    "Open the filter panel, then noop() to wait for it to load. "
    "The panel has 'Order results by' rows with a FIELD selector and a DIRECTION selector.\n\n"
    "FIELD selector is a custom combobox (input + adjacent button). "
    "The input element times out — click the adjacent BUTTON instead. "
    "If it also times out, try bid+1. Never retry the same BID twice.\n\n"
    "Steps: browser_click(button_bid) -> noop() -> browser_click(option_bid). "
    "DIRECTION is a native <select>: browser_select_option(bid, 'a to z') or 'z to a'.\n\n"
    "To add another sort field: click 'Add Sort', then repeat. "
    "After all sort fields are set: click 'Run' to apply."
)

# Filter: step-by-step filter condition interaction pattern.
_FILTER_HINT: str = (
    "Open the filter panel, then noop() to wait for it to load. "
    "Each row has: FIELD selector, OPERATOR selector, VALUE field.\n\n"
    "FIELD and OPERATOR selectors are custom comboboxes (input + adjacent button). "
    "The input element times out — click the adjacent BUTTON instead. "
    "If it also times out, try bid+1. Never retry the same BID twice.\n\n"
    "Steps: browser_click(button_bid) -> noop() -> browser_click(option_bid). "
    "Common operators: 'is', 'is not', 'contains', 'starts with', 'is empty'. "
    "For 'is empty': select operator only, no value needed.\n\n"
    "VALUE field type varies:\n"
    "  - Choice/boolean: <select> -> browser_select_option()\n"
    "  - Reference (names, groups): keyboard_type_into() for autocomplete, then noop() + click suggestion\n"
    "  - Plain text: browser_type()\n\n"
    "To add conditions: click 'AND', then repeat. After all set: click 'Run'."
)

# Create: autocomplete reference field workflow.
_CREATE_HINT: str = (
    "For reference fields with autocomplete (e.g. Caller, Assignment group): "
    "use keyboard_type_into(bid, text) to type character-by-character. "
    "Then noop() to wait, find the suggestion in the AXTree, and click it. "
    "For plain text use browser_type(). For dropdowns use browser_select_option()."
)

WORKARENA_TASK_HINTS: dict[str, str] = {
    # Create (5)
    "workarena.servicenow.create-incident": _CREATE_HINT,
    "workarena.servicenow.create-hardware-asset": _CREATE_HINT,
    "workarena.servicenow.create-change-request": _CREATE_HINT,
    "workarena.servicenow.create-user": _CREATE_HINT,
    "workarena.servicenow.create-problem": _CREATE_HINT,
    # Sort (6)
    "workarena.servicenow.sort-asset-list": _SORT_HINT,
    "workarena.servicenow.sort-change-request-list": _SORT_HINT,
    "workarena.servicenow.sort-hardware-list": _SORT_HINT,
    "workarena.servicenow.sort-incident-list": _SORT_HINT,
    "workarena.servicenow.sort-service-catalog-item-list": _SORT_HINT,
    "workarena.servicenow.sort-user-list": _SORT_HINT,
    # Filter (6)
    "workarena.servicenow.filter-asset-list": _FILTER_HINT,
    "workarena.servicenow.filter-change-request-list": _FILTER_HINT,
    "workarena.servicenow.filter-hardware-list": _FILTER_HINT,
    "workarena.servicenow.filter-incident-list": _FILTER_HINT,
    "workarena.servicenow.filter-service-catalog-item-list": _FILTER_HINT,
    "workarena.servicenow.filter-user-list": _FILTER_HINT,
}

# Backward compat
WORKARENA_DEFAULT_HINT: str = ""
