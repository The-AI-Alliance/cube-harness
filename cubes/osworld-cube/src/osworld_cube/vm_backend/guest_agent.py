"""HTTP client for the Flask guest agent running inside the OSWorld VM.

Ported from desktop_env.controllers.python.PythonController.
The guest agent server runs at port 5000 inside the VM and exposes endpoints
for screenshots, accessibility trees, command execution, and file I/O.
"""

import json
import logging
import random
import time
import traceback
from typing import Any

import requests

logger = logging.getLogger(__name__)

_KEYBOARD_KEYS = [
    "\t",
    "\n",
    "\r",
    " ",
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "{",
    "|",
    "}",
    "~",
    "accept",
    "add",
    "alt",
    "altleft",
    "altright",
    "apps",
    "backspace",
    "browserback",
    "browserfavorites",
    "browserforward",
    "browserhome",
    "browserrefresh",
    "browsersearch",
    "browserstop",
    "capslock",
    "clear",
    "convert",
    "ctrl",
    "ctrlleft",
    "ctrlright",
    "decimal",
    "del",
    "delete",
    "divide",
    "down",
    "end",
    "enter",
    "esc",
    "escape",
    "execute",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "f12",
    "f13",
    "f14",
    "f15",
    "f16",
    "f17",
    "f18",
    "f19",
    "f20",
    "final",
    "fn",
    "hanguel",
    "hangul",
    "hanja",
    "help",
    "home",
    "insert",
    "junja",
    "kana",
    "kanji",
    "launchapp1",
    "launchapp2",
    "launchmail",
    "launchmediaselect",
    "left",
    "modechange",
    "multiply",
    "nexttrack",
    "nonconvert",
    "num0",
    "num1",
    "num2",
    "num3",
    "num4",
    "num5",
    "num6",
    "num7",
    "num8",
    "num9",
    "numlock",
    "pagedown",
    "pageup",
    "pause",
    "pgdn",
    "pgup",
    "playpause",
    "prevtrack",
    "print",
    "printscreen",
    "prntscrn",
    "prtsc",
    "prtscr",
    "return",
    "right",
    "scrolllock",
    "select",
    "separator",
    "shift",
    "shiftleft",
    "shiftright",
    "sleep",
    "space",
    "stop",
    "subtract",
    "tab",
    "up",
    "volumedown",
    "volumemute",
    "volumeup",
    "win",
    "winleft",
    "winright",
    "yen",
    "command",
    "option",
    "optionleft",
    "optionright",
]

_PYAUTOGUI_PREFIX = "import pyautogui; import time; pyautogui.FAILSAFE = False; {command}"

_RETRY_TIMES = 3
_RETRY_INTERVAL = 5


class GuestAgent:
    """HTTP client for the Flask agent server running inside the OSWorld VM.

    Parameters
    ----------
    host : str
        Hostname or IP of the server (typically "localhost" with port-forwarded QEMU).
    port : int
        Host port mapped to the guest's Flask server (default 5000).
    """

    def __init__(self, host: str = "localhost", port: int = 5000) -> None:
        self.host = host
        self.port = port
        self._base_url = f"http://{host}:{port}"

    # ------------------------------------------------------------------
    # Observation retrieval
    # ------------------------------------------------------------------

    def get_screenshot(self) -> bytes | None:
        """Return raw PNG/JPEG bytes of the current screen, or None on failure."""
        for attempt in range(_RETRY_TIMES):
            try:
                resp = requests.get(self._base_url + "/screenshot", timeout=10)
                if resp.status_code == 200 and self._is_valid_image(resp.headers.get("Content-Type", ""), resp.content):
                    return resp.content
                logger.error("Invalid screenshot response (attempt %d/%d)", attempt + 1, _RETRY_TIMES)
            except Exception as exc:
                logger.error("Screenshot error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        logger.error("Failed to get screenshot after %d attempts", _RETRY_TIMES)
        return None

    def get_accessibility_tree(self) -> str | None:
        """Return the XML accessibility tree string, or None on failure."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.get(self._base_url + "/accessibility")
                if resp.status_code == 200:
                    return resp.json()["AT"]
                logger.error("Accessibility tree error: %d", resp.status_code)
            except Exception as exc:
                logger.error("Accessibility tree error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        logger.error("Failed to get accessibility tree")
        return None

    def get_terminal_output(self) -> str | None:
        """Return the terminal output string, or None on failure."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.get(self._base_url + "/terminal")
                if resp.status_code == 200:
                    return resp.json()["output"]
                logger.error("Terminal output error: %d", resp.status_code)
            except Exception as exc:
                logger.error("Terminal output error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        logger.error("Failed to get terminal output")
        return None

    def get_file(self, file_path: str) -> bytes | None:
        """Download a file from the VM by path, or None on failure."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(self._base_url + "/file", data={"file_path": file_path})
                if resp.status_code == 200:
                    return resp.content
                logger.error("Get file error: %d", resp.status_code)
            except Exception as exc:
                logger.error("Get file error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        logger.error("Failed to get file: %s", file_path)
        return None

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def execute_python_command(self, command: str) -> dict[str, Any] | None:
        """Execute a Python command via pyautogui prefix inside the VM."""
        command_list = ["python", "-c", _PYAUTOGUI_PREFIX.format(command=command)]
        payload = json.dumps({"command": command_list, "shell": False})

        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/execute",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=90,
                )
                if resp.status_code == 200:
                    return resp.json()
                logger.error("Execute python error: %d", resp.status_code)
            except requests.exceptions.ReadTimeout:
                break
            except Exception as exc:
                logger.error("Execute python error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        logger.error("Failed to execute python command")
        return None

    def run_python_script(self, script: str) -> dict[str, Any] | None:
        """Execute a Python script file inside the VM via /run_python."""
        payload = json.dumps({"code": script})

        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/run_python",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=90,
                )
                if resp.status_code == 200:
                    return resp.json()
                return {
                    "status": "error",
                    "message": "Request failed",
                    "output": None,
                    "error": resp.json().get("error"),
                }
            except requests.exceptions.ReadTimeout:
                break
            except Exception:
                logger.error("Run python script error: %s", traceback.format_exc())
            time.sleep(_RETRY_INTERVAL)
        return {"status": "error", "message": "Retry limit reached", "output": "", "error": "Retry limit reached."}

    def run_bash_script(self, script: str, timeout: int = 30, working_dir: str | None = None) -> dict[str, Any] | None:
        """Execute a bash script inside the VM via /run_bash_script."""
        payload = json.dumps({"script": script, "timeout": timeout, "working_dir": working_dir})

        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/run_bash_script",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=timeout + 100,
                )
                if resp.status_code == 200:
                    return resp.json()
                logger.error("Run bash script error: %d %s", resp.status_code, resp.text)
            except requests.exceptions.ReadTimeout:
                return {"status": "error", "output": "", "error": f"Timed out after {timeout}s", "returncode": -1}
            except Exception as exc:
                logger.error("Run bash script error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return {"status": "error", "output": "", "error": f"Failed after {_RETRY_TIMES} retries", "returncode": -1}

    def execute_action(self, action: dict[str, Any]) -> None:
        """Dispatch a computer_13 action dict to the appropriate pyautogui command.

        Mirrors the dispatch table in desktop_env.controllers.python.PythonController.execute_action.
        """
        if action in ("WAIT", "FAIL", "DONE"):
            return

        action_type: str = action["action_type"]
        parameters: dict = action.get("parameters") or {k: v for k, v in action.items() if k != "action_type"}

        move_mode = random.choice(
            [
                "pyautogui.easeInQuad",
                "pyautogui.easeOutQuad",
                "pyautogui.easeInOutQuad",
                "pyautogui.easeInBounce",
                "pyautogui.easeInElastic",
            ]
        )
        duration = random.uniform(0.5, 1)

        if action_type == "MOVE_TO":
            if not parameters:
                self.execute_python_command("pyautogui.moveTo()")
            elif "x" in parameters and "y" in parameters:
                self.execute_python_command(
                    f"pyautogui.moveTo({parameters['x']}, {parameters['y']}, {duration}, {move_mode})"
                )
            else:
                raise ValueError(f"Unknown MOVE_TO parameters: {parameters}")

        elif action_type == "CLICK":
            if not parameters:
                self.execute_python_command("pyautogui.click()")
            elif "button" in parameters and "x" in parameters and "y" in parameters:
                btn, x, y = parameters["button"], parameters["x"], parameters["y"]
                nc = parameters.get("num_clicks")
                if nc:
                    self.execute_python_command(f"pyautogui.click(button='{btn}', x={x}, y={y}, clicks={nc})")
                else:
                    self.execute_python_command(f"pyautogui.click(button='{btn}', x={x}, y={y})")
            elif "button" in parameters:
                btn = parameters["button"]
                nc = parameters.get("num_clicks")
                if nc:
                    self.execute_python_command(f"pyautogui.click(button='{btn}', clicks={nc})")
                else:
                    self.execute_python_command(f"pyautogui.click(button='{btn}')")
            elif "x" in parameters and "y" in parameters:
                x, y = parameters["x"], parameters["y"]
                nc = parameters.get("num_clicks")
                if nc:
                    self.execute_python_command(f"pyautogui.click(x={x}, y={y}, clicks={nc})")
                else:
                    self.execute_python_command(f"pyautogui.click(x={x}, y={y})")
            else:
                raise ValueError(f"Unknown CLICK parameters: {parameters}")

        elif action_type == "MOUSE_DOWN":
            btn = parameters.get("button", "left")
            self.execute_python_command(f"pyautogui.mouseDown(button='{btn}')")

        elif action_type == "MOUSE_UP":
            btn = parameters.get("button", "left")
            self.execute_python_command(f"pyautogui.mouseUp(button='{btn}')")

        elif action_type == "RIGHT_CLICK":
            if "x" in parameters and "y" in parameters:
                self.execute_python_command(f"pyautogui.rightClick(x={parameters['x']}, y={parameters['y']})")
            else:
                self.execute_python_command("pyautogui.rightClick()")

        elif action_type == "DOUBLE_CLICK":
            if "x" in parameters and "y" in parameters:
                self.execute_python_command(f"pyautogui.doubleClick(x={parameters['x']}, y={parameters['y']})")
            else:
                self.execute_python_command("pyautogui.doubleClick()")

        elif action_type == "DRAG_TO":
            self.execute_python_command(
                f"pyautogui.dragTo({parameters['x']}, {parameters['y']}, duration=1.0, button='left', mouseDownUp=True)"
            )

        elif action_type == "SCROLL":
            dx = parameters.get("dx")
            dy = parameters.get("dy")
            if dx is not None:
                self.execute_python_command(f"pyautogui.hscroll({dx})")
            if dy is not None:
                self.execute_python_command(f"pyautogui.vscroll({dy})")

        elif action_type == "TYPING":
            text = parameters["text"]
            self.execute_python_command("pyautogui.typewrite({:})".format(repr(text)))

        elif action_type == "PRESS":
            key = parameters["key"]
            if key.lower() not in _KEYBOARD_KEYS:
                raise ValueError(f"Key must be one of the known keyboard keys, got: {key!r}")
            self.execute_python_command(f"pyautogui.press('{key}')")

        elif action_type == "KEY_DOWN":
            key = parameters["key"]
            if key.lower() not in _KEYBOARD_KEYS:
                raise ValueError(f"Key must be one of the known keyboard keys, got: {key!r}")
            self.execute_python_command(f"pyautogui.keyDown('{key}')")

        elif action_type == "KEY_UP":
            key = parameters["key"]
            if key.lower() not in _KEYBOARD_KEYS:
                raise ValueError(f"Key must be one of the known keyboard keys, got: {key!r}")
            self.execute_python_command(f"pyautogui.keyUp('{key}')")

        elif action_type == "HOTKEY":
            keys: list[str] = parameters["keys"]
            if not isinstance(keys, list):
                raise ValueError("HOTKEY keys must be a list")
            for k in keys:
                if k.lower() not in _KEYBOARD_KEYS:
                    raise ValueError(f"Key must be one of the known keyboard keys, got: {k!r}")
            self.execute_python_command("pyautogui.hotkey('" + "', '".join(keys) + "')")

        elif action_type in ("WAIT", "FAIL", "DONE"):
            pass

        else:
            raise ValueError(f"Unknown action type: {action_type!r}")

    # ------------------------------------------------------------------
    # VM info
    # ------------------------------------------------------------------

    def get_vm_platform(self) -> str:
        """Return the platform string (e.g. 'Linux', 'Windows')."""
        result = self.execute_python_command("import platform; print(platform.system())")
        if result and result.get("output"):
            return result["output"].strip()
        return ""

    def get_vm_screen_size(self) -> dict[str, Any] | None:
        """Return the VM screen size dict."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(self._base_url + "/screen_size")
                if resp.status_code == 200:
                    return resp.json()
            except Exception as exc:
                logger.error("Screen size error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return None

    def get_vm_window_size(self, app_class_name: str) -> dict[str, Any] | None:
        """Return the window size for an application by class name."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(self._base_url + "/window_size", data={"app_class_name": app_class_name})
                if resp.status_code == 200:
                    return resp.json()
            except Exception as exc:
                logger.error("Window size error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return None

    def get_vm_desktop_path(self) -> str | None:
        """Return the desktop directory path inside the VM."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(self._base_url + "/desktop_path")
                if resp.status_code == 200:
                    return resp.json()["desktop_path"]
            except Exception as exc:
                logger.error("Desktop path error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return None

    def get_vm_directory_tree(self, path: str) -> dict[str, Any] | None:
        """Return the directory tree for the given path inside the VM."""
        payload = json.dumps({"path": path})
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/list_directory",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                )
                if resp.status_code == 200:
                    return resp.json()["directory_tree"]
            except Exception as exc:
                logger.error("Directory tree error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return None

    def get_vm_wallpaper(self) -> bytes | None:
        """Return the current desktop wallpaper image bytes."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(self._base_url + "/wallpaper")
                if resp.status_code == 200:
                    return resp.content
            except Exception as exc:
                logger.error("Wallpaper error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_image(content_type: str, data: bytes | None) -> bool:
        if not isinstance(data, (bytes, bytearray)) or not data:
            return False
        if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
            return True
        if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
            return True
        if content_type and any(t in content_type for t in ("image/png", "image/jpeg", "image/jpg")):
            return True
        return False
