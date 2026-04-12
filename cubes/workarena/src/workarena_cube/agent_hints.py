"""WorkArena task-specific agent hints for the meta-agent iterative improvement loop.

Hints here are accumulated through meta-agent debugging iterations.
Each hint explains *why* it was added so future iterations can refine or remove it.

Usage in recipes:
    from workarena_cube.agent_hints import WORKARENA_TASK_HINTS, WORKARENA_DEFAULT_HINT

    agent_config = GennyConfig(
        ...
        hint=WORKARENA_DEFAULT_HINT,
        task_hints=WORKARENA_TASK_HINTS,
    )
"""

# General fallback hint applied to all WorkArena tasks.
# Set to "" unless all tasks in a run share a common failure mode.
WORKARENA_DEFAULT_HINT: str = ""

# Per-task hints. Keys use the format "workarena.servicenow.<task-name>" with dashes.
# These take precedence over WORKARENA_DEFAULT_HINT.
#
# Iteration history:
#   v1 (2026-04-10): chart tasks all failed because agent didn't know to call send_message.
#                    Sort tasks failed because agent clicked column headers instead of filter UI.
#   v2 (2026-04-12): create tasks: added submit_form() hint. Verified that submit_form() correctly
#                    writes localStorage sys_id — record is created in DB.
#                    Reference fields require keyboard_type_into to trigger autocomplete.
#                    Episode reaches done=True — remaining failures are semantic (missing fields).
#   v3 (2026-04-12): ported filter/sort hints from best gpt-5 run to new action names.
#                    Added all 6 filter tasks and all 6 sort tasks.
#                    Key change: old action names (click/fill/press/select_option) →
#                    new names (browser_click/browser_type/browser_press_key/browser_select_option).
#                    For reference-field values: use keyboard_type_into instead of fill+press.

# --- Create tasks ---
# submit_form() is required — the visible Submit buttons use sysverb_insert_and_stay,
# which navigates to a new empty form without saving (WorkArena's gsftSubmit hook is bypassed).
# submit_form() calls gsftSubmit() directly in the gsft_main iframe, which triggers WorkArena's
# patch that writes localStorage[session_sys_id_field] before submitting. Verified working.
_CREATE_HINT: str = (
    "Read the task goal carefully and fill in EVERY field it specifies before submitting. "
    "Missing even one field will result in failure — do not skip any. "
    "For reference fields that show an autocomplete dropdown (e.g. Caller, Assignment group, "
    "Configuration item): use `keyboard_type_into(bid=<field_bid>, text=<value>)` to type "
    "character-by-character, which triggers the autocomplete. After typing, wait one step "
    "(noop), then find the suggestion in the AXTree and click it using browser_click(). "
    "For plain text fields, use browser_type(). For dropdowns/selects, use browser_select_option(). "
    "To submit the form: call `submit_form()` — do NOT click the visible Submit button. "
    "The Submit button navigates away without saving. submit_form() calls the correct action. "
    "IMPORTANT: submit_form() is final — once called the episode ends, so set ALL fields first."
)

# --- Sort tasks ---
# Validation checks sysparm_query URL param for ORDERBY fields.
# Column header clicks do NOT update sysparm_query — only the filter UI does.
# The 'Order results by' field selector is a custom combobox (input + adjacent button):
# always click the BUTTON (not the input), then click the desired field option.
# The direction selector IS a native <select> — use browser_select_option().
_SORT_HINT: str = (
    "Use the FILTER UI to sort — do NOT click column headers. "
    "Open the filter panel (funnel/filter icon), then call noop() to wait for it to load. "
    "The panel has 'Order results by' rows. Each row has a FIELD selector and a DIRECTION selector. "
    "FIELD selector is a CUSTOM COMBOBOX (input + adjacent button). "
    "CRITICAL: the combobox INPUT element will ALWAYS time out when clicked. "
    "The BUTTON is the next element in the AXTree (BID = input_BID + 1 or nearby). "
    "Steps: "
    "  1. browser_click(bid=<button_bid>) — if it times out, try bid+1 immediately (do NOT retry same BID). "
    "  2. noop() to wait for the field list to appear. "
    "  3. browser_click(bid=<option_bid>) to select the desired field. "
    "DIRECTION selector is a native <select>: browser_select_option(bid=<select_bid>, value='a to z') "
    "or value='z to a'. "
    "To add another sort field: browser_click() the 'Add Sort' button, then repeat. "
    "After ALL sort fields are set: browser_click() the 'Run' button to apply. "
    "Do NOT use js_eval during this task — it wastes turns. "
    "Do NOT call send_message — sort tasks are validated automatically."
)

# --- Filter tasks ---
# The filter condition builder uses custom comboboxes for field and operator selection.
# Always click the BUTTON adjacent to the combobox (not the combobox itself).
# For reference field values (person names, groups etc): use keyboard_type_into for autocomplete.
# For empty-value conditions (field 'is empty'): select the 'is empty' operator, no value needed.
_FILTER_HINT: str = (
    "Use the FILTER UI to add filter conditions. "
    "Open the filter panel (funnel/filter icon), then call noop() to wait for it to load. "
    "Each filter row has: FIELD selector, OPERATOR selector, VALUE field. "
    "FIELD and OPERATOR selectors are CUSTOM COMBOBOXes (input + adjacent button). "
    "CRITICAL: the combobox INPUT element will ALWAYS time out when clicked. "
    "The BUTTON is the next element in the AXTree (BID = input_BID + 1 or nearby). "
    "Steps for FIELD selector: "
    "  1. browser_click(bid=<button_bid>) — if it times out, try bid+1 immediately (do NOT retry same BID). "
    "  2. noop(), then browser_click(bid=<option_bid>) to select the field. "
    "Steps for OPERATOR selector: same pattern — click button adjacent to operator input, then click option. "
    "  Common operators: 'is', 'is not', 'contains', 'starts with', 'is empty'. "
    "  For 'is empty' conditions: just select the operator, no value is needed. "
    "VALUE field: "
    "  - For CHOICE/BOOLEAN fields: a <select> — use browser_select_option(). "
    "  - For reference fields (person names, groups): use keyboard_type_into(bid=<input_bid>, "
    "    text=<value>) to trigger autocomplete. Then noop(), find the suggestion and browser_click() it. "
    "  - For plain text: use browser_type(bid=<input_bid>, text=<value>). "
    "To add another condition: browser_click() the 'AND' button, then repeat for the new row. "
    "After ALL conditions are fully set: browser_click() the 'Run' button to apply. "
    "Do NOT use js_eval during this task — it wastes turns. "
    "Do NOT call send_message — filter tasks are validated automatically."
)

WORKARENA_TASK_HINTS: dict[str, str] = {
    # --- Create tasks (5 total) ---
    "workarena.servicenow.create-incident": _CREATE_HINT,
    "workarena.servicenow.create-hardware-asset": _CREATE_HINT,
    "workarena.servicenow.create-change-request": _CREATE_HINT,
    "workarena.servicenow.create-user": _CREATE_HINT,
    "workarena.servicenow.create-problem": _CREATE_HINT,
    # --- Chart tasks (4 total) ---
    # Single-chart: agent reads numeric value from AXTree chart, responds with send_message.
    # Root cause v1: agent navigated away instead of calling send_message.
    "workarena.servicenow.single-chart-value-retrieval": (
        "When you have found the answer to the question, you MUST call `send_message` "
        "with ONLY the numeric value (no units, no explanation). "
        "Example: send_message('42.5'). Do NOT navigate away or take any other action first."
    ),
    "workarena.servicenow.single-chart-min-max-retrieval": (
        "When you have found the answer to the question, you MUST call `send_message` "
        "with ONLY the numeric value (no units, no explanation). "
        "Example: send_message('42.5'). Do NOT navigate away or take any other action first."
    ),
    # Multi-chart value retrieval: same as single-chart, answer is a single number.
    "workarena.servicenow.multi-chart-value-retrieval": (
        "When you have found the answer to the question, you MUST call `send_message` "
        "with ONLY the numeric value (no units, no explanation). "
        "Example: send_message('42.5'). Do NOT navigate away or take any other action first."
    ),
    # Multi-chart min/max: task asks for BOTH label and count in 'label, count' format.
    # The AXTree chart data: image '<n>. <label>, <value>. <series>.'
    "workarena.servicenow.multi-chart-min-max-retrieval": (
        "Read the chart data from the accessibility tree — each data point appears as "
        "an image element with format '<label>, <value>. <series>'. "
        "When you have found the min or max value, call `send_message` with "
        "the label and count in the format: 'label, count'. "
        "Example: send_message('2026-01-19, 18'). Do NOT send just the number — include the label too."
    ),
    # --- Sort tasks (6 total) ---
    # Validation checks sysparm_query URL for ORDERBY — column headers don't update it.
    "workarena.servicenow.sort-asset-list": _SORT_HINT,
    "workarena.servicenow.sort-change-request-list": _SORT_HINT,
    "workarena.servicenow.sort-hardware-list": _SORT_HINT,
    "workarena.servicenow.sort-incident-list": _SORT_HINT,
    "workarena.servicenow.sort-service-catalog-item-list": _SORT_HINT,
    "workarena.servicenow.sort-user-list": _SORT_HINT,
    # --- Filter tasks (6 total) ---
    # Custom combobox pattern — must click adjacent BUTTON, not the input itself.
    "workarena.servicenow.filter-asset-list": _FILTER_HINT,
    "workarena.servicenow.filter-change-request-list": _FILTER_HINT,
    "workarena.servicenow.filter-hardware-list": _FILTER_HINT,
    "workarena.servicenow.filter-incident-list": _FILTER_HINT,
    "workarena.servicenow.filter-service-catalog-item-list": _FILTER_HINT,
    "workarena.servicenow.filter-user-list": _FILTER_HINT,
}
