"""PyAutoGUI utilities: ported from desktop_env.desktop_env."""

import re


def fix_pyautogui_less_than_bug(command: str) -> str:
    """Fix PyAutoGUI '<' character bug by converting it to hotkey("shift", ',') calls.

    This fixes the known PyAutoGUI issue where typing '<' produces '>' instead.
    References:
    - https://github.com/asweigart/pyautogui/issues/198
    - https://github.com/xlang-ai/OSWorld/issues/257

    Parameters
    ----------
    command : str
        The original pyautogui command string.

    Returns
    -------
    str
        The fixed command with '<' characters handled properly.
    """
    # Pattern to match press('<') or press('\u003c') calls
    press_pattern = r'pyautogui\.press\(["\'](?:<|\\u003c)["\']\)'

    def replace_press_less_than(match: re.Match) -> str:
        return 'pyautogui.hotkey("shift", ",")'

    command = re.sub(press_pattern, replace_press_less_than, command)

    # Pattern to match typewrite calls with quoted strings
    typewrite_pattern = r'pyautogui\.typewrite\((["\'])(.*?)\1\)'

    def process_typewrite_match(match: re.Match) -> str:
        quote_char = match.group(1)
        content = match.group(2)

        try:
            decoded_content = content.encode("utf-8").decode("unicode_escape")
            content = decoded_content
        except UnicodeDecodeError:
            pass

        if "<" not in content:
            return match.group(0)

        parts = content.split("<")
        result_parts = []

        for i, part in enumerate(parts):
            if i == 0:
                if part:
                    result_parts.append(f"pyautogui.typewrite({quote_char}{part}{quote_char})")
            else:
                result_parts.append('pyautogui.hotkey("shift", ",")')
                if part:
                    result_parts.append(f"pyautogui.typewrite({quote_char}{part}{quote_char})")

        return "; ".join(result_parts)

    command = re.sub(typewrite_pattern, process_typewrite_match, command)

    return command
