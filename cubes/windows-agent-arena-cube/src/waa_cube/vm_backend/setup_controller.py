"""WAA task setup controller — ported from WAA's controllers/setup.py.

Uses GuestAgent for all VM communication. Windows-specific methods are preserved;
Linux-only methods (proxy pool, tinyproxy) are omitted.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Union

import requests
from cube_computer_tool.guest_agent import GuestAgent
from requests_toolbelt.multipart.encoder import MultipartEncoder

logger = logging.getLogger(__name__)

MAX_RETRIES = 60

# HuggingFace-hosted empty history DB (same as osworld-cube)
_HISTORY_EMPTY_DB_URL = (
    "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/"
    "chrome/44ee5668-ecd5-4366-a6ce-c1c9b8d4e938/history_empty.sqlite?download=true"
)


class SetupController:
    """Orchestrates task-setup steps in the WAA Windows VM.

    Ported from WAA's desktop_env/controllers/setup.py.
    Uses :class:`GuestAgent` instead of raw ``vm_ip`` + ``server_port``.

    Parameters
    ----------
    guest : GuestAgent
        HTTP client connected to the running Windows VM.
    chromium_port : int
        Host port forwarded to the VM's Chrome DevTools port (9222).
    vlc_port : int
        Host port forwarded to the VM's VLC HTTP port (8080).
    cache_dir : str
        Directory for caching downloaded setup files.
    screen_width : int
        VM screen width in pixels.
    screen_height : int
        VM screen height in pixels.
    """

    def __init__(
        self,
        guest: GuestAgent,
        chromium_port: int = 9222,
        vlc_port: int = 8080,
        cache_dir: str = "cache",
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> None:
        self.guest = guest
        self.chromium_port = chromium_port
        self.vlc_port = vlc_port
        self.http_server = guest._base_url
        self.http_server_setup_root = guest._base_url + "/setup"
        self.cache_dir = cache_dir
        self.screen_width = screen_width
        self.screen_height = screen_height

    def reset_cache_dir(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir

    def setup(self, config: list[dict[str, Any]]) -> bool:
        """Run all setup steps from the task config.

        Returns True if all steps completed, False if VM was unreachable.
        """
        # Wait for VM connectivity
        for retry in range(MAX_RETRIES):
            try:
                requests.get(self.http_server + "/probe", timeout=5)
                break
            except Exception:
                time.sleep(10)
                logger.info("Waiting for WAA VM connectivity: %d/%d", retry + 1, MAX_RETRIES)
        else:
            logger.error("WAA VM unreachable after %d retries", MAX_RETRIES)
            return False

        for i, cfg in enumerate(config):
            config_type: str = cfg["type"]
            parameters: dict[str, Any] = cfg.get("parameters", {})
            setup_fn_name = "_{:}_setup".format(config_type)

            if not hasattr(self, setup_fn_name):
                raise AttributeError(f"SetupController has no method {setup_fn_name!r}")

            try:
                logger.info("Setup step %d/%d: %s", i + 1, len(config), setup_fn_name)
                getattr(self, setup_fn_name)(**parameters)
                logger.info("Setup completed: %s", setup_fn_name)
            except Exception as exc:
                logger.error("Setup failed at step %d/%d: %s — %s", i + 1, len(config), setup_fn_name, exc)
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Setup step {i + 1} failed: {setup_fn_name} — {exc}") from exc

        return True

    # ------------------------------------------------------------------
    # Setup step handlers
    # ------------------------------------------------------------------

    def _activate_window_setup(self, window_name: str, strict: bool = False, by_class: bool = False) -> None:
        payload = json.dumps({"window_name": window_name, "strict": strict, "by_class": by_class})
        try:
            resp = requests.post(
                self.http_server_setup_root + "/activate_window",
                headers={"Content-Type": "application/json"},
                data=payload,
            )
            if resp.status_code == 200:
                logger.info("Window activated: %s", window_name)
            else:
                logger.error("Failed to activate window %s: %s", window_name, resp.text)
        except requests.RequestException as exc:
            logger.error("activate_window error: %s", exc)

    def _change_wallpaper_setup(self, path: str) -> None:
        if not path:
            raise ValueError(f"Invalid wallpaper path: {path!r}")
        payload = json.dumps({"path": path})
        try:
            resp = requests.post(
                self.http_server_setup_root + "/change_wallpaper",
                headers={"Content-Type": "application/json"},
                data=payload,
            )
            if resp.status_code != 200:
                logger.error("Failed to change wallpaper: %s", resp.text)
        except requests.RequestException as exc:
            logger.error("change_wallpaper error: %s", exc)

    def _chrome_close_tabs_setup(self, urls_to_close: list[str]) -> None:
        from playwright.sync_api import sync_playwright

        from waa_cube.vm_backend.metrics.utils import compare_urls

        time.sleep(5)
        remote_debugging_url = f"http://{self.guest.host}:{self.chromium_port}"

        with sync_playwright() as p:
            browser = None
            for attempt in range(30):
                try:
                    browser = p.chromium.connect_over_cdp(remote_debugging_url)
                    break
                except Exception as exc:
                    if attempt < 29:
                        logger.error("Chrome CDP connect attempt %d failed: %s", attempt + 1, exc)
                        time.sleep(5)
                    else:
                        raise

            if not browser:
                return

            context = browser.contexts[0]
            for i, url in enumerate(urls_to_close):
                for page in context.pages:
                    if compare_urls(page.url, url):
                        page.close()
                        logger.info("Closed tab %d: %s", i + 1, url)
                        break

    def _chrome_open_tabs_setup(self, urls_to_open: list[str]) -> None:
        from playwright.sync_api import sync_playwright

        remote_debugging_url = f"http://{self.guest.host}:{self.chromium_port}"
        logger.info("Connecting to Chrome @ %s", remote_debugging_url)

        for attempt in range(30):
            if attempt > 0:
                time.sleep(5)

            with sync_playwright() as p:
                try:
                    browser = p.chromium.connect_over_cdp(remote_debugging_url)
                except Exception as exc:
                    if attempt < 29:
                        logger.error("Chrome CDP attempt %d failed: %s", attempt + 1, exc)
                        continue
                    raise

                if not browser:
                    return

                for i, url in enumerate(urls_to_open):
                    context = browser.contexts[0]
                    page = context.new_page()
                    try:
                        page.goto(url, timeout=60000)
                    except Exception:
                        logger.warning("Opening %s exceeded time limit", url)
                    logger.info("Opened tab %d: %s", i + 1, url)

                    if i == 0:
                        context.pages[0].close()

                return

    def _close_all_setup(self) -> None:
        try:
            resp = requests.post(
                self.http_server_setup_root + "/close_all",
                headers={"Content-Type": "application/json"},
                data=json.dumps({}),
            )
            if resp.status_code != 200:
                logger.error("Failed to close all windows: %s", resp.text)
        except requests.RequestException as exc:
            logger.error("close_all error: %s", exc)

    def _close_window_setup(self, window_name: str, strict: bool = False, by_class: bool = False) -> None:
        payload = json.dumps({"window_name": window_name, "strict": strict, "by_class": by_class})
        try:
            resp = requests.post(
                self.http_server_setup_root + "/close_window",
                headers={"Content-Type": "application/json"},
                data=payload,
            )
            if resp.status_code != 200:
                logger.error("Failed to close window %s: %s", window_name, resp.text)
        except requests.RequestException as exc:
            logger.error("close_window error: %s", exc)

    def _command_setup(self, command: list[str], **kwargs: Any) -> None:
        self._execute_setup(command, **kwargs)

    def _create_file_setup(self, path: str, content: str = "") -> None:
        if not path:
            raise ValueError(f"Invalid create_file path: {path!r}")
        payload = json.dumps({"path": path, "content": content})
        try:
            resp = requests.post(
                self.http_server_setup_root + "/create_file",
                headers={"Content-Type": "application/json"},
                data=payload,
            )
            if resp.status_code != 200:
                logger.error("Failed to create file %s: %s", path, resp.text)
        except requests.RequestException as exc:
            logger.error("create_file error: %s", exc)

    def _create_folder_setup(self, path: str) -> None:
        if not path:
            raise ValueError(f"Invalid create_folder path: {path!r}")
        payload = json.dumps({"path": path})
        try:
            resp = requests.post(
                self.http_server_setup_root + "/create_folder",
                headers={"Content-Type": "application/json"},
                data=payload,
            )
            if resp.status_code != 200:
                logger.error("Failed to create folder %s: %s", path, resp.text)
        except requests.RequestException as exc:
            logger.error("create_folder error: %s", exc)

    def _download_setup(self, files: list[dict[str, str]]) -> None:
        for f in files:
            url: str = f["url"]
            path: str = f["path"]
            if not url or not path:
                raise ValueError(f"Invalid download url={url!r} or path={path!r}")

            cache_path = os.path.join(
                self.cache_dir,
                "{:}_{:}".format(uuid.uuid5(uuid.NAMESPACE_URL, url), os.path.basename(path)),
            )

            if not os.path.exists(cache_path):
                logger.info("Downloading %s → %s", url, cache_path)
                for attempt in range(3):
                    try:
                        resp = requests.get(url, stream=True, timeout=300)
                        resp.raise_for_status()
                        with open(cache_path, "wb") as fp:
                            for chunk in resp.iter_content(chunk_size=8192):
                                if chunk:
                                    fp.write(chunk)
                        break
                    except requests.RequestException as exc:
                        logger.error("Download attempt %d failed: %s", attempt + 1, exc)
                        if os.path.exists(cache_path):
                            os.remove(cache_path)
                else:
                    raise requests.RequestException(f"Failed to download {url} after 3 attempts")

            form = MultipartEncoder({"file_path": path, "file_data": (os.path.basename(path), open(cache_path, "rb"))})
            resp = requests.post(
                self.http_server_setup_root + "/upload",
                headers={"Content-Type": form.content_type},
                data=form,
                timeout=600,
            )
            if resp.status_code != 200:
                raise requests.RequestException(f"Upload failed ({resp.status_code}): {resp.text}")
            logger.info("Uploaded %s to VM at %s", os.path.basename(path), path)

    def _execute_setup(
        self,
        command: Union[str, list[str]],
        stdout: str = "",
        stderr: str = "",
        shell: bool = False,
        until: dict[str, Any] | None = None,
    ) -> None:
        if not command:
            raise ValueError("Empty execute command")

        until = until or {}
        command = self._replace_screen_env(command)
        payload = json.dumps({"command": command, "shell": shell})
        headers = {"Content-Type": "application/json"}

        terminates = False
        nb_failings = 0
        while not terminates:
            try:
                resp = requests.post(self.http_server_setup_root + "/execute", headers=headers, data=payload)
                if resp.status_code == 200:
                    results = resp.json()
                    if stdout:
                        Path(self.cache_dir, stdout).write_text(results.get("output", ""))
                    if stderr:
                        Path(self.cache_dir, stderr).write_text(results.get("error", ""))
                else:
                    results = None
                    nb_failings += 1
            except requests.RequestException as exc:
                logger.error("execute error: %s", exc)
                results = None
                nb_failings += 1

            if not until:
                terminates = True
            elif results is not None:
                terminates = (
                    ("returncode" in until and results.get("returncode") == until["returncode"])
                    or ("stdout" in until and until["stdout"] in results.get("output", ""))
                    or ("stderr" in until and until["stderr"] in results.get("error", ""))
                )
            terminates = terminates or nb_failings >= 5
            if not terminates:
                time.sleep(0.3)

    def _launch_setup(self, command: Union[str, list[str]], shell: bool = False) -> None:
        if not command:
            raise ValueError("Empty launch command")
        if not shell and isinstance(command, str) and len(command.split()) > 1:
            command = command.split()
        payload = json.dumps({"command": command, "shell": shell})
        try:
            resp = requests.post(
                self.http_server_setup_root + "/launch",
                headers={"Content-Type": "application/json"},
                data=payload,
            )
            if resp.status_code != 200:
                logger.error("Failed to launch: %s", resp.text)
        except requests.RequestException as exc:
            logger.error("launch error: %s", exc)

    def _open_setup(self, path: str) -> None:
        if not path:
            raise ValueError(f"Invalid open path: {path!r}")
        payload = json.dumps({"path": path})
        try:
            resp = requests.post(
                self.http_server_setup_root + "/open_file",
                headers={"Content-Type": "application/json"},
                data=payload,
                timeout=1810,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to open {path!r}: {exc}") from exc

    def _recycle_file_setup(self, path: str) -> None:
        if not path:
            raise ValueError(f"Invalid recycle_file path: {path!r}")
        payload = json.dumps({"path": path})
        try:
            resp = requests.post(
                self.http_server_setup_root + "/recycle",
                headers={"Content-Type": "application/json"},
                data=payload,
            )
            if resp.status_code != 200:
                logger.error("Failed to recycle file %s: %s", path, resp.text)
        except requests.RequestException as exc:
            logger.error("recycle_file error: %s", exc)

    def _sleep_setup(self, seconds: float) -> None:
        time.sleep(seconds)

    def _tidy_desktop_setup(self, **kwargs: Any) -> None:
        """Minimize all windows to tidy the desktop."""
        self._close_all_setup()

    def _update_browse_history_setup(self, **config: Any) -> None:
        """Inject fake Chrome browsing history into the Windows VM."""
        cache_path = os.path.join(self.cache_dir, "history_new.sqlite")

        if not os.path.exists(cache_path):
            for attempt in range(3):
                try:
                    resp = requests.get(_HISTORY_EMPTY_DB_URL, stream=True)
                    resp.raise_for_status()
                    with open(cache_path, "wb") as fp:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                fp.write(chunk)
                    break
                except requests.RequestException as exc:
                    logger.error("History DB download attempt %d failed: %s", attempt + 1, exc)
            else:
                raise requests.RequestException(f"Failed to download history DB from {_HISTORY_EMPTY_DB_URL}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "history_empty.sqlite")
            shutil.copy(cache_path, db_path)

            epoch_start = datetime(1601, 1, 1)
            conn = sqlite3.connect(db_path)
            for item in config["history"]:
                visit_time = datetime.now() - timedelta(seconds=item["visit_time_from_now_in_seconds"])
                chrome_ts = int((visit_time - epoch_start).total_seconds() * 1_000_000)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO urls (url, title, visit_count, typed_count, last_visit_time, hidden) VALUES (?, ?, ?, ?, ?, ?)",
                    (item["url"], item["title"], 1, 0, chrome_ts, 0),
                )
                url_id = cursor.lastrowid
                cursor.execute(
                    "INSERT INTO visits (url, visit_time, from_visit, transition, segment_id, visit_duration) VALUES (?, ?, ?, ?, ?, ?)",
                    (url_id, chrome_ts, 0, 805306368, 0, 0),
                )
                conn.commit()
            conn.close()

            result = self.guest.execute_python_command(
                r"""import os; print(os.path.join(os.getenv('USERPROFILE'), 'AppData', 'Local', 'Google', 'Chrome', 'User Data', 'Default', 'History'))"""
            )
            chrome_history_path = result["output"].strip() if result else ""

            form = MultipartEncoder(
                {
                    "file_path": chrome_history_path,
                    "file_data": (os.path.basename(chrome_history_path), open(db_path, "rb")),
                }
            )
            resp = requests.post(
                self.http_server_setup_root + "/upload",
                headers={"Content-Type": form.content_type},
                data=form,
            )
            if resp.status_code != 200:
                logger.error("Failed to upload Chrome history DB: %s", resp.text)

    def _update_browse_history_edge_setup(self, **config: Any) -> None:
        """Inject fake Edge browsing history into the Windows VM."""
        cache_path = os.path.join(self.cache_dir, "history_edge_new.sqlite")

        if not os.path.exists(cache_path):
            for attempt in range(3):
                try:
                    resp = requests.get(_HISTORY_EMPTY_DB_URL, stream=True)
                    resp.raise_for_status()
                    with open(cache_path, "wb") as fp:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                fp.write(chunk)
                    break
                except requests.RequestException as exc:
                    logger.error("Edge history DB download attempt %d failed: %s", attempt + 1, exc)
            else:
                raise requests.RequestException(f"Failed to download history DB from {_HISTORY_EMPTY_DB_URL}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "history_empty.sqlite")
            shutil.copy(cache_path, db_path)

            epoch_start = datetime(1601, 1, 1)
            conn = sqlite3.connect(db_path)
            for item in config["history"]:
                visit_time = datetime.now() - timedelta(seconds=item["visit_time_from_now_in_seconds"])
                edge_ts = int((visit_time - epoch_start).total_seconds() * 1_000_000)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO urls (url, title, visit_count, typed_count, last_visit_time, hidden) VALUES (?, ?, ?, ?, ?, ?)",
                    (item["url"], item["title"], 1, 0, edge_ts, 0),
                )
                url_id = cursor.lastrowid
                cursor.execute(
                    "INSERT INTO visits (url, visit_time, from_visit, transition, segment_id, visit_duration) VALUES (?, ?, ?, ?, ?, ?)",
                    (url_id, edge_ts, 0, 805306368, 0, 0),
                )
                conn.commit()
            conn.close()

            result = self.guest.execute_python_command(
                r"""import os; print(os.path.join(os.getenv('USERPROFILE'), 'AppData', 'Local', 'Microsoft', 'Edge', 'User Data', 'Default', 'History'))"""
            )
            edge_history_path = result["output"].strip() if result else ""

            form = MultipartEncoder(
                {
                    "file_path": edge_history_path,
                    "file_data": (os.path.basename(edge_history_path), open(db_path, "rb")),
                }
            )
            resp = requests.post(
                self.http_server_setup_root + "/upload",
                headers={"Content-Type": form.content_type},
                data=form,
            )
            if resp.status_code != 200:
                logger.error("Failed to upload Edge history DB: %s", resp.text)

    def _upload_file_setup(self, files: list[dict[str, str]]) -> None:
        for f in files:
            local_path = f["local_path"]
            path = f["path"]
            if not os.path.exists(local_path):
                raise ValueError(f"Local file not found: {local_path!r}")
            last_err: Exception = RuntimeError("Upload not attempted")
            for attempt in range(3):
                try:
                    with open(local_path, "rb") as fp:
                        form = MultipartEncoder({"file_path": path, "file_data": (os.path.basename(path), fp)})
                        resp = requests.post(
                            self.http_server_setup_root + "/upload",
                            headers={"Content-Type": form.content_type},
                            data=form,
                            timeout=(10, 600),
                        )
                        if resp.status_code == 200:
                            break
                        last_err = requests.RequestException(f"Upload failed ({resp.status_code}): {resp.text}")
                except requests.RequestException as exc:
                    last_err = exc
                    time.sleep(2**attempt)
            else:
                raise last_err

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _replace_screen_env(self, command: Union[str, list[str]]) -> Union[str, list[str]]:
        replacements = {
            "{SCREEN_WIDTH}": str(self.screen_width),
            "{SCREEN_HEIGHT}": str(self.screen_height),
            "{SCREEN_WIDTH_HALF}": str(self.screen_width // 2),
            "{SCREEN_HEIGHT_HALF}": str(self.screen_height // 2),
        }
        if isinstance(command, str):
            for k, v in replacements.items():
                command = command.replace(k, v)
            return command
        return [
            token.replace("{SCREEN_WIDTH_HALF}", replacements["{SCREEN_WIDTH_HALF}"])
            .replace("{SCREEN_HEIGHT_HALF}", replacements["{SCREEN_HEIGHT_HALF}"])
            .replace("{SCREEN_WIDTH}", replacements["{SCREEN_WIDTH}"])
            .replace("{SCREEN_HEIGHT}", replacements["{SCREEN_HEIGHT}"])
            for token in command
        ]
