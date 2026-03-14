"""Tests for cube.tool.Tool used in cube_harness."""

from unittest.mock import MagicMock, patch

import pytest
from cube.core import Action, ActionSchema, Observation, StepError
from cube.tool import Tool, tool_action
from opentelemetry.trace import SpanKind

from cube_harness.tool import AsyncToolWithTelemetry


class TestAbstractTool:
    """Tests for AbstractTool base class."""

    def test_abstract_tool_methods(self, mock_tool):
        """Test that AbstractTool methods work as expected."""
        # reset() should not raise
        mock_tool.reset()
        assert mock_tool.click_count == 0

        # close() should not raise
        mock_tool.close()


class TestTool:
    """Tests for Tool class."""

    def test_tool_actions(self, mock_tool):
        """Test getting actions from tool."""
        actions = mock_tool.action_set
        assert len(actions) == 2
        action_names = {a.name for a in actions}
        assert "click" in action_names
        assert "type_text" in action_names

    def test_tool_action_schema_format(self, mock_tool):
        """Test that action schemas have correct format."""
        actions = mock_tool.action_set
        click_action = next(a for a in actions if a.name == "click")

        assert isinstance(click_action, ActionSchema)
        assert "Click on an element" in click_action.description
        assert "element_id" in click_action.parameters.get("properties", {})

    def test_tool_execute_action_click(self, mock_tool):
        """Test executing click action."""
        action = Action(name="click", arguments={"element_id": "button_1"})
        result = mock_tool.execute_action(action)

        assert result.contents[0].data == "Clicked on button_1"
        assert mock_tool.click_count == 1

    def test_tool_execute_action_type_text(self, mock_tool):
        """Test executing type_text action."""
        action = Action(name="type_text", arguments={"element_id": "input_1", "text": "Hello"})
        result = mock_tool.execute_action(action)

        assert result.contents[0].data == "Typed 'Hello' into input_1"
        assert mock_tool.typed_texts == [("input_1", "Hello")]

    def test_tool_execute_action_returns_success_on_none(self, mock_tool):
        """Test that execute_action returns 'Success' when method returns None."""

        # Add a method that returns None
        class ExtendedTool(Tool):
            @tool_action
            def click(self, element_id: str) -> str:
                """Click.

                Args:
                    element_id: Element.

                Returns:
                    Result.
                """
                return f"Clicked {element_id}"

            @tool_action
            def type_text(self, element_id: str, text: str) -> str:
                """Type.

                Args:
                    element_id: Element.
                    text: Text.

                Returns:
                    Result.
                """
                return f"Typed {text}"

            @tool_action
            def noop(self) -> None:
                """Do nothing.

                Returns:
                    Nothing.
                """
                pass

        tool = ExtendedTool()
        action = Action(name="noop", arguments={})
        result = tool.execute_action(action)
        assert isinstance(result, Observation)
        assert result.contents[0].data == "Success"

    def test_tool_execute_action_error_handling(self, mock_tool):
        """Test that execute_action handles errors gracefully."""

        # Override click to raise an error
        def raise_error(element_id: str) -> str:
            """Click stub that always raises."""
            raise ValueError("Element not found")

        original_click = mock_tool.click
        mock_tool.click = raise_error

        action = Action(name="click", arguments={"element_id": "nonexistent"})
        result = mock_tool.execute_action(action)

        assert isinstance(result, StepError)
        assert "Element not found" in result.exception_str

        # Restore
        mock_tool.click = original_click

    def test_tool_get_action_method_valid(self, mock_tool):
        """Test getting valid action method."""
        action = Action(name="click", arguments={})
        method = mock_tool.get_action_method(action)
        assert callable(method)
        assert method == mock_tool.click

    def test_tool_get_action_method_invalid_action_space(self, mock_tool):
        """Test getting method for action not in action space."""
        action = Action(name="invalid_action", arguments={})
        with pytest.raises(ValueError, match="does not exist in"):
            mock_tool.get_action_method(action)

    def test_tool_get_action_method_not_implemented(self):
        """Test getting method that's in action space but not implemented."""

        class PartialTool(Tool):
            @tool_action
            def implemented(self) -> str:
                """Implemented method.

                Returns:
                    Result.
                """
                return "done"

            # not_implemented exists but has no @tool_action so it is not a registered action
            def not_implemented(self) -> str:
                """Not decorated."""
                return "oops"

        tool = PartialTool()
        action = Action(name="not_implemented", arguments={})
        with pytest.raises(ValueError, match="is not decorated with @tool_action"):
            tool.get_action_method(action)

    def test_tool_reset(self, mock_tool):
        """Test tool reset."""
        # Modify state
        mock_tool.click_count = 5
        mock_tool.typed_texts = [("a", "b")]

        # Reset
        mock_tool.reset()

        assert mock_tool.click_count == 0
        assert mock_tool.typed_texts == []

    def test_tool_multiple_action_executions(self, mock_tool):
        """Test multiple action executions."""
        actions = [
            Action(name="click", arguments={"element_id": "btn1"}),
            Action(name="click", arguments={"element_id": "btn2"}),
            Action(name="type_text", arguments={"element_id": "input", "text": "test"}),
        ]

        results = [mock_tool.execute_action(a).contents[0].data for a in actions]

        assert mock_tool.click_count == 2
        assert len(mock_tool.typed_texts) == 1
        assert "btn1" in results[0]
        assert "btn2" in results[1]
        assert "test" in results[2]


class TestToolExecutionSpans:
    """Tests for tool execution OpenTelemetry spans following GenAI conventions.

    Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/
    """

    @patch("cube_harness.metrics.tracer._tool_tracer")
    def test_execute_action_creates_span(self, mock_tracer, mock_tool) -> None:
        """Test that execute_action creates a span with correct name and kind."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        action = Action(name="click", arguments={"element_id": "btn1"})
        mock_tool.execute_action(action)

        mock_tracer.start_as_current_span.assert_called_once()
        call_args = mock_tracer.start_as_current_span.call_args
        assert call_args[0][0] == "execute_tool click"

        assert call_args[1]["kind"] == SpanKind.INTERNAL

    @patch("cube_harness.metrics.tracer._tool_tracer")
    def test_execute_action_sets_required_attributes(self, mock_tracer, mock_tool) -> None:
        """Test that execute_action sets required GenAI tool attributes."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        action = Action(id="call_123", name="click", arguments={"element_id": "btn1"})
        mock_tool.execute_action(action)

        set_attr_calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
        assert set_attr_calls["gen_ai.tool.name"] == "click"
        assert set_attr_calls["gen_ai.tool.call.id"] == "call_123"
        assert set_attr_calls["gen_ai.tool.call.arguments"] == '{"element_id": "btn1"}'
        assert set_attr_calls["gen_ai.tool.call.result"] == "Clicked on btn1"

    @patch("cube_harness.metrics.tracer._tool_tracer")
    def test_execute_action_traces_error_result(self, mock_tracer, mock_tool) -> None:
        """Test that error results are traced."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        def raise_error(element_id: str) -> str:
            """Click stub that always raises."""
            raise ValueError("Element not found")

        mock_tool.click = raise_error

        action = Action(name="click", arguments={"element_id": "nonexistent"})
        mock_tool.execute_action(action)

        set_attr_calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
        assert "Error executing action click" in set_attr_calls["gen_ai.tool.call.result"]
        assert "Element not found" in set_attr_calls["gen_ai.tool.call.result"]


class MockAsyncTool(AsyncToolWithTelemetry):
    """Async mock tool for telemetry tests."""

    def __init__(self):
        self.click_count = 0

    @tool_action
    async def click(self, element_id: str) -> str:
        """Click on an element.

        Args:
            element_id: The element to click.

        Returns:
            Click confirmation message.
        """
        self.click_count += 1
        return f"Clicked on {element_id}"


@pytest.fixture
def mock_async_tool() -> MockAsyncTool:
    return MockAsyncTool()


class TestAsyncToolExecutionSpans:
    """Tests for AsyncToolWithTelemetry OpenTelemetry spans."""

    @patch("cube_harness.metrics.tracer._tool_tracer")
    @pytest.mark.asyncio
    async def test_execute_action_creates_span(self, mock_tracer, mock_async_tool) -> None:
        """Test that execute_action creates a span with correct name and kind."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        action = Action(name="click", arguments={"element_id": "btn1"})
        await mock_async_tool.execute_action(action)

        mock_tracer.start_as_current_span.assert_called_once()
        call_args = mock_tracer.start_as_current_span.call_args
        assert call_args[0][0] == "execute_tool click"
        assert call_args[1]["kind"] == SpanKind.INTERNAL

    @patch("cube_harness.metrics.tracer._tool_tracer")
    @pytest.mark.asyncio
    async def test_execute_action_sets_required_attributes(self, mock_tracer, mock_async_tool) -> None:
        """Test that execute_action sets required GenAI tool attributes."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        action = Action(id="call_123", name="click", arguments={"element_id": "btn1"})
        await mock_async_tool.execute_action(action)

        set_attr_calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
        assert set_attr_calls["gen_ai.tool.name"] == "click"
        assert set_attr_calls["gen_ai.tool.call.id"] == "call_123"
        assert set_attr_calls["gen_ai.tool.call.arguments"] == '{"element_id": "btn1"}'
        assert set_attr_calls["gen_ai.tool.call.result"] == "Clicked on btn1"

    @patch("cube_harness.metrics.tracer._tool_tracer")
    @pytest.mark.asyncio
    async def test_execute_action_traces_error_result(self, mock_tracer, mock_async_tool) -> None:
        """Test that error results are traced."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        async def raise_error(element_id: str) -> str:
            """Click stub that always raises."""
            raise ValueError("Element not found")

        mock_async_tool.click = raise_error

        action = Action(name="click", arguments={"element_id": "nonexistent"})
        await mock_async_tool.execute_action(action)

        set_attr_calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
        assert "Error executing action click" in set_attr_calls["gen_ai.tool.call.result"]
        assert "Element not found" in set_attr_calls["gen_ai.tool.call.result"]
