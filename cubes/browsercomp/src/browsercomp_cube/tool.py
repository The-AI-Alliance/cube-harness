"""SubmitAnswerTool for BrowseComp."""

from cube.tool import Tool, ToolConfig, tool_action


class SubmitAnswerToolConfig(ToolConfig):
    """Configuration for SubmitAnswerTool."""

    def make(self, container=None) -> "SubmitAnswerTool":
        return SubmitAnswerTool()


class SubmitAnswerTool(Tool):
    """Tool for submitting a final answer to a BrowseComp research question."""

    def __init__(self) -> None:
        self._submitted_answer: str | None = None

    def reset(self) -> None:
        self._submitted_answer = None

    @property
    def last_answer(self) -> str | None:
        return self._submitted_answer

    @tool_action
    def submit_answer(self, answer: str) -> str:
        """Submit your final answer to the research question.

        Args:
            answer: Your final answer. Format as:
                Explanation: <brief reasoning>
                Exact Answer: <the precise answer>
                Confidence: <0-100>
        """
        self._submitted_answer = answer
        return f"Answer submitted: {answer}"
