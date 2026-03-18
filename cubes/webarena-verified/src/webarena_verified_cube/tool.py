import tempfile

from cube.tool import ToolConfig, tool_action
from cube_browser_tool import PlaywrightConfig, SyncPlaywrightTool

from cube_harness.tool import ToolWithTelemetry
from cube_harness.tools.toolbox import ToolboxConfig
from webarena_verified.types.agent_response import FinalAgentResponse, MainObjectiveType, PublicResultItem, Status


class HarPlaywrightConfig(PlaywrightConfig):
    har_path: str = ""

    def make(self, container=None) -> SyncPlaywrightTool:
        with tempfile.NamedTemporaryFile(suffix=".har", delete=False) as f:
            har_path = f.name
        browser_with_har = self.browser.model_copy(
            update={"pw_extra_kwargs": {**self.browser.pw_extra_kwargs, "record_har_path": har_path}}
        )
        config_with_har = self.model_copy(update={"browser": browser_with_har, "har_path": har_path})
        session = browser_with_har.make()
        return SyncPlaywrightTool(config=config_with_har, session=session)


class SubmitResponseConfig(ToolConfig):
    def make(self, container=None) -> "SubmitResponseTool":
        return SubmitResponseTool()


class SubmitResponseTool(ToolWithTelemetry):
    """Tool providing the submit_response action for WebArena tasks."""

    def __init__(self) -> None:
        self._submitted_response: FinalAgentResponse | None = None

    def reset(self) -> None:
        self._submitted_response = None

    def close(self) -> None:
        pass

    def get_submitted_response(self) -> FinalAgentResponse | None:
        return self._submitted_response

    @tool_action
    def submit_response(
        self,
        task_type: str,
        status: str,
        retrieved_data: list[PublicResultItem] | None = None,
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


class WebArenaToolConfig(ToolboxConfig):
    tool_configs: list[ToolConfig] = [HarPlaywrightConfig(), SubmitResponseConfig()]
