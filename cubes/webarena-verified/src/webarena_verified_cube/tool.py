import json
import logging
import os
import tempfile

from cube.tool import tool_action
from playwright.sync_api import BrowserContext

from agentlab2.tools.playwright import PlaywrightConfig, SyncPlaywrightTool
from webarena_verified.types.agent_response import FinalAgentResponse, MainObjectiveType, Status

logger = logging.getLogger(__name__)


class WebArenaToolConfig(PlaywrightConfig):
    def make(self, container=None) -> "WebArenaSyncPlaywrightTool":
        return WebArenaSyncPlaywrightTool(self)


class WebArenaSyncPlaywrightTool(SyncPlaywrightTool):
    """Playwright tool extended with HAR recording and submit_response action for WebArena."""

    def __init__(self, config: PlaywrightConfig) -> None:
        super().__init__(config)
        # super().__init__ creates self._page = self._browser.new_page().
        # We replace it with a page from a HAR-recording context.
        self._page.close()
        self._submitted_response: FinalAgentResponse | None = None
        self._har_path: str = _new_har_path()
        self._context: BrowserContext = self._browser.new_context(record_har_path=self._har_path)
        self._page = self._context.new_page()

    def reset(self) -> None:
        self._context.close()
        _delete_path(self._har_path)
        self._submitted_response = None
        self._har_path = _new_har_path()
        self._context = self._browser.new_context(record_har_path=self._har_path)
        self._page = self._context.new_page()

    def close(self) -> None:
        self._context.close()
        _delete_path(self._har_path)
        self._browser.close()
        self._pw.stop()

    def get_har(self) -> list[dict]:
        """Close current context (flushing HAR), read entries, then open a fresh context."""
        self._context.close()
        entries = _read_har(self._har_path)
        _delete_path(self._har_path)
        self._har_path = _new_har_path()
        self._context = self._browser.new_context(record_har_path=self._har_path)
        self._page = self._context.new_page()
        return entries

    def get_submitted_response(self) -> FinalAgentResponse | None:
        return self._submitted_response

    @tool_action
    def submit_response(
        self,
        task_type: str,
        status: str,
        retrieved_data: list | None = None,
        error_details: str | None = None,
    ) -> str:
        """Submit your final response for the task.

        Args:
            task_type: The type of task performed. Must be one of: RETRIEVE, MUTATE, NAVIGATE.
                - RETRIEVE: The main objective was to retrieve or look up information.
                - MUTATE: The main objective was to create, update, or delete data or state.
                - NAVIGATE: The main objective was to navigate to a specific page or location.
            status: The outcome of the task execution. Must be one of: SUCCESS,
                ACTION_NOT_ALLOWED_ERROR, NOT_FOUND_ERROR, PERMISSION_DENIED_ERROR,
                DATA_VALIDATION_ERROR, UNKNOWN_ERROR.
            retrieved_data: Array of retrieved items for RETRIEVE tasks, null for MUTATE/NAVIGATE.
                All items must be the same type. Use numbers for counts/amounts, booleans for
                true/false values. Returns empty list if no items were found.
            error_details: Null when status is SUCCESS. Otherwise, concisely explains the failure.
        """
        self._submitted_response = FinalAgentResponse(
            task_type=MainObjectiveType(task_type.upper()),
            status=Status(status.upper()),
            retrieved_data=retrieved_data,
            error_details=error_details,
        )
        return f"Response submitted: task_type={task_type}, status={status}."


def _new_har_path() -> str:
    with tempfile.NamedTemporaryFile(suffix=".har", delete=False) as f:
        return f.name


def _delete_path(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _read_har(path: str) -> list[dict]:
    try:
        with open(path) as f:
            har = json.load(f)
        return har["log"]["entries"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        logger.warning(f"Failed to read HAR file at {path}, returning empty list.")
        return []
