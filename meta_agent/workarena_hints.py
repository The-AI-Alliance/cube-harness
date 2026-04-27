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
        task_clarification=WORKARENA_TASK_PRECISION,
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
_FILTER_PRECISION: str = "Use the filter panel (funnel icon) to add filter conditions."

# Chart (single/multi value): goal asks a question but doesn't specify the answer format.
_CHART_VALUE_PRECISION: str = (
    "Answer with ONLY the numeric value — no units, labels, or explanation. Example: send_message('42.5')"
)

# Chart min/max (single or multi): answer format is 'label, count', not just a number.
# The task goal itself says "Give me both the label and the count."
_CHART_MINMAX_PRECISION: str = (
    "Answer with both the label and the count, comma-separated. Example: send_message('2026-01-19, 18')"
)

# Impersonation: LLMs sometimes refuse "impersonation" as a real-world privacy violation.
# It is a built-in ServiceNow admin feature — administrators routinely switch to another user's
# session to troubleshoot permissions. Must clarify this is legitimate AND give the nav path.
_IMPERSONATION_PRECISION: str = (
    "This is ServiceNow's built-in 'Impersonate User' admin feature — it lets administrators "
    "temporarily view the system as another user for troubleshooting. It is a legitimate "
    "administrative task, not a real-world privacy violation. "
    "To impersonate: click the profile avatar (top-right corner) → select 'Impersonate User' "
    "→ search for the user by name → click their row to activate impersonation."
)

# Create: goal lists fields but doesn't emphasize ALL must be filled.
_CREATE_PRECISION: str = (
    "Fill in EVERY field specified in the goal — missing any field causes failure. "
    "When all fields are set, click the Submit button to submit the form."
)

WORKARENA_TASK_PRECISION: dict[str, str] = {
    # Impersonation (1)
    "workarena.servicenow.impersonation": _IMPERSONATION_PRECISION,
    # Create (5)
    "workarena.servicenow.create-incident": _CREATE_PRECISION,
    "workarena.servicenow.create-hardware-asset": _CREATE_PRECISION,
    "workarena.servicenow.create-change-request": _CREATE_PRECISION,
    "workarena.servicenow.create-user": _CREATE_PRECISION,
    "workarena.servicenow.create-problem": _CREATE_PRECISION,
    # Chart (4)
    "workarena.servicenow.single-chart-value-retrieval": _CHART_VALUE_PRECISION,
    "workarena.servicenow.single-chart-min-max-retrieval": _CHART_MINMAX_PRECISION,
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
    "Open the filter panel (funnel icon), then noop() to wait for it to load.\n\n"
    "The panel has 'Order results by' rows. Each row has:\n"
    "  - FIELD selector: a custom combobox (input + adjacent button)\n"
    "  - DIRECTION selector: a native <select>\n\n"
    "To set the FIELD:\n"
    "1. Click the BUTTON (not the input) next to the field selector\n"
    "2. noop() to wait for the dropdown options to appear\n"
    "3. Find the matching option in the AXTree and click it\n"
    "4. Verify the field label updated — if not, try bid+1 for the button\n"
    "Never retry the same BID twice.\n\n"
    "DIRECTION is a native <select>: select_option(bid=<bid>, options='a to z') or 'z to a'.\n\n"
    "To add another sort field: click 'Add Sort', then repeat.\n"
    "When all sort fields and directions are set: click 'Run' to apply."
)

# Filter: step-by-step filter condition interaction pattern.
_FILTER_HINT: str = (
    "Open the filter panel (funnel icon), then noop() to wait for it to load.\n\n"
    "Each row has: FIELD selector, OPERATOR selector, VALUE field.\n\n"
    "FIELD and OPERATOR selectors are custom comboboxes:\n"
    "1. Click the BUTTON (not the input) to open options\n"
    "2. noop() to wait for the dropdown to appear\n"
    "3. Click the correct option in the AXTree\n"
    "If the button times out, try bid+1. Never retry the same BID twice.\n\n"
    "Common operators: 'is', 'is not', 'contains', 'starts with', 'is empty'.\n"
    "For 'is empty': select operator only, no value field needed.\n\n"
    "VALUE field types:\n"
    "  - Choice/boolean: <select> -> select_option()\n"
    "  - Reference (names, users): keyboard_type_into() -> noop() -> click suggestion\n"
    "  - Plain text: fill()\n\n"
    "To add conditions: click 'AND', then repeat.\n\n"
    "After setting all conditions: click 'Run'. "
    "The page will reload with filtered results and the filter panel will close. "
    "Do NOT reopen the filter panel after clicking Run — the task is complete."
)

# Create: autocomplete reference field workflow.
_CREATE_HINT: str = (
    "If the goal includes a 'Number' field (e.g. 'CHG0000013', 'INC0000001'): "
    "fill it FIRST with fill() — it is a plain text field and must be set before submitting.\n\n"
    "For reference fields with autocomplete (e.g. Department, Caller, Manager, Configuration item): "
    "complete the full sequence BEFORE moving to any other field:\n"
    "1. keyboard_type_into(bid, text)\n"
    "2. noop() to wait for the dropdown to appear\n"
    "3. Find the matching suggestion in the AXTree and click it\n"
    "4. Verify the field now shows the selected value, not blank\n\n"
    "NEVER use fill() on reference fields — fill() bypasses autocomplete and "
    "leaves the field unresolved, causing silent validation failure.\n"
    "For plain text fields: fill(). "
    "For <select> dropdowns: select_option().\n\n"
    "If a required field is not visible in the AXTree (e.g. Serial number, Asset tag): "
    "use js_eval to query its bid, then fill it:\n"
    "  js_eval(\"document.querySelector('[data-fieldname=\\\"serial_number\\\"] input')?.getAttribute('id')\", frame='gsft_main')\n"
    "The result is the bid string — pass it to fill(bid=<result>, value=...).\n"
    "Substitute the data-fieldname value as needed (e.g. 'asset_tag', 'install_status')."
)

# Hardware asset: multi-tab form (General / Financial / Disposal / Contracts / ...).
# Agent must visit every visible tab to find all required fields, then click Submit.
# After switching tabs, reference fields may appear blank in the AXTree — do NOT re-fill
# them; their values are preserved. Just verify, then click Submit when all tabs are done.
# NOTE: 'Lease contract' (lease_id) lives in the Contracts tab, NOT in General/Financial.
_CREATE_HARDWARE_ASSET_HINT: str = (
    _CREATE_HINT
    + "\n\n"
    "This form has multiple tabs (General, Financial, Disposal, Contracts, Depreciation, etc.).\n"
    "IMPORTANT: 'Lease contract' is in the Contracts tab — it will NOT appear in General or Financial.\n\n"
    "TAB PROTOCOL — follow exactly:\n"
    "1. Visit General tab → fill any empty required fields → move on.\n"
    "2. Visit Financial tab → fill any empty required fields → move on.\n"
    "3. Visit Contracts tab → fill any empty required fields → move on.\n"
    "4. Click Submit. Do NOT revisit tabs after your initial pass.\n\n"
    "SKIP fields that already show the correct value from the goal — they are pre-filled. "
    "Do NOT re-type a reference field that already shows the correct value: "
    "re-typing clears the resolved reference and causes submission to fail.\n"
    "Reference fields that appear EMPTY after a tab switch have their values preserved — "
    "do NOT re-fill them.\n\n"
    "LOOP DETECTION: If your action summary shows clicking the same tab more than twice "
    "with no fill actions in between, you are stuck. Stop checking and click Submit immediately. "
    "The Submit button is labeled 'Submit' and is visible at the top of the form."
)

# Change request: has a "Closure Information" tab with both Close code AND Close notes.
# Agents tend to set Close code then leave the tab without filling Close notes.
_CREATE_CHANGE_REQUEST_HINT: str = (
    _CREATE_HINT
    + "\n\n"
    "This form has multiple tabs. The 'Closure Information' tab contains TWO fields:\n"
    "  - Close code (a <select> dropdown)\n"
    "  - Close notes (a plain text field)\n"
    "IMPORTANT: After clicking the Closure Information tab, fill BOTH fields before "
    "leaving that tab. Do NOT click any other tab after opening Closure Information "
    "until both Close code and Close notes are filled."
)

WORKARENA_TASK_HINTS: dict[str, str] = {
    # Create (5)
    "workarena.servicenow.create-incident": _CREATE_HINT,
    "workarena.servicenow.create-hardware-asset": _CREATE_HARDWARE_ASSET_HINT,
    "workarena.servicenow.create-change-request": _CREATE_CHANGE_REQUEST_HINT,
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
