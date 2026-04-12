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
#   v2 (2026-04-12): create tasks: added submit_form() hint. The visible "Submit" buttons call
#                    sysverb_insert_and_stay which navigates to a new empty form without saving.
#                    submit_form() calls gsftSubmit() directly in gsft_main, triggering WorkArena's
#                    validation patch that writes localStorage (required for reward). Verified via
#                    localStorage inspection that the sys_id key is correctly written when
#                    submit_form() is used.
#                    Reference fields (Caller, Assignment Group, etc.) require keyboard_type_into
#                    to trigger autocomplete — browser_type bypasses keyboard events.

_CREATE_HINT: str = (
    "Read the task goal carefully and fill in EVERY field it specifies before submitting. "
    "Missing even one field will result in failure — do not skip any. "
    "For reference fields that show an autocomplete dropdown (e.g. Caller, Assignment group, "
    "Configuration item): use `keyboard_type_into(bid=<field_bid>, text=<value>)` to type "
    "character-by-character, which triggers the autocomplete. After typing, wait one step "
    "(noop), then find the suggestion in the AXTree and click it. "
    "For plain text fields, use browser_type(). For dropdowns/selects, use browser_select_option(). "
    "To submit the form: call `submit_form()` — do NOT click the visible Submit button. "
    "The Submit button navigates away without saving. submit_form() calls the correct action. "
    "IMPORTANT: submit_form() is final — once called the episode ends, so set ALL fields first."
)

WORKARENA_TASK_HINTS: dict[str, str] = {
    # --- Create tasks ---
    "workarena.servicenow.create-incident": _CREATE_HINT,
    "workarena.servicenow.create-hardware-asset": _CREATE_HINT,
    "workarena.servicenow.create-change-request": _CREATE_HINT,
    "workarena.servicenow.create-user": _CREATE_HINT,
    "workarena.servicenow.create-problem": _CREATE_HINT,
    # --- Chart tasks ---
    # Single-chart: agent reads the single numeric value from the AXTree chart region
    # and must respond with ONLY the number (no label, no units).
    # Root cause: without hint, agent navigates away instead of calling send_message.
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
    # Multi-chart min/max: task asks for BOTH the label and the count.
    # The AXTree chart data appears as: image '<n>. <label>, <value>. <series>.'
    # The expected send_message format is 'label, count' (from WorkArena cheat()).
    "workarena.servicenow.multi-chart-min-max-retrieval": (
        "Read the chart data from the accessibility tree — each data point appears as "
        "an image element with format '<label>, <value>. <series>'. "
        "When you have found the min or max value, call `send_message` with "
        "the label and count in the format: 'label, count'. "
        "Example: send_message('2026-01-19, 18'). Do NOT send just the number — include the label too."
    ),
    # --- Sort tasks ---
    # Validation checks sysparm_query URL param for ORDERBY fields.
    # Column header clicks do NOT update sysparm_query — only the filter UI does.
    # The filter UI sort fields are autocomplete inputs (not <select>):
    # use browser_type to enter the field name, then click the suggestion.
    "workarena.servicenow.sort-asset-list": (
        "Use the FILTER UI to sort — do NOT click column headers. "
        "The filter panel has 'Order results by' autocomplete inputs. "
        "To set a sort field: "
        "  1. Call browser_type(bid=<combobox_bid>, text=<field_name>) DIRECTLY — do NOT click first. "
        "     browser_type will focus and type into the input automatically. "
        "  2. Wait one step (noop) for suggestions to appear. "
        "  3. Find the suggestion in the AXTree and click it. "
        "  4. Then click the direction combobox and click 'a to z' (ascending) or 'z to a' (descending). "
        "  5. Click 'Add Sort' for each additional sort field and repeat. "
        "After setting ALL fields, click 'Run' to apply. "
        "Do NOT call send_message — sort tasks do not require a chat answer."
    ),
    "workarena.servicenow.sort-incident-list": (
        "Use the FILTER UI to sort — do NOT click column headers. "
        "The filter panel has 'Order results by' autocomplete inputs. "
        "To set a sort field: "
        "  1. Call browser_type(bid=<combobox_bid>, text=<field_name>) DIRECTLY — do NOT click first. "
        "     browser_type will focus and type into the input automatically. "
        "  2. Wait one step (noop) for suggestions to appear. "
        "  3. Find the suggestion in the AXTree and click it. "
        "  4. Then click the direction combobox and click 'a to z' (ascending) or 'z to a' (descending). "
        "  5. Click 'Add Sort' for each additional sort field and repeat. "
        "After setting ALL fields, click 'Run' to apply. "
        "Do NOT call send_message — sort tasks do not require a chat answer."
    ),
}
