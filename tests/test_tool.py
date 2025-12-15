"""Tests for agentlab2.tool module."""

from typing import Protocol, runtime_checkable

import pytest

from agentlab2.core import Action, ActionSchema
from agentlab2.tool import Tool


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
        actions = mock_tool.actions()
        assert len(actions) == 2
        action_names = {a.name for a in actions}
        assert "click" in action_names
        assert "type_text" in action_names

    def test_tool_action_schema_format(self, mock_tool):
        """Test that action schemas have correct format."""
        actions = mock_tool.actions()
        click_action = next(a for a in actions if a.name == "click")

        assert isinstance(click_action, ActionSchema)
        assert "Click on an element" in click_action.description
        assert "element_id" in click_action.parameters.get("properties", {})

    def test_tool_execute_action_click(self, mock_tool):
        """Test executing click action."""
        action = Action(name="click", arguments={"element_id": "button_1"})
        result = mock_tool.execute_action(action)

        assert result == "Clicked on button_1"
        assert mock_tool.click_count == 1

    def test_tool_execute_action_type_text(self, mock_tool):
        """Test executing type_text action."""
        action = Action(name="type_text", arguments={"element_id": "input_1", "text": "Hello"})
        result = mock_tool.execute_action(action)

        assert result == "Typed 'Hello' into input_1"
        assert mock_tool.typed_texts == [("input_1", "Hello")]

    def test_tool_execute_action_returns_success_on_none(self, mock_tool):
        """Test that execute_action returns 'Success' when method returns None."""

        # Add a method that returns None
        @runtime_checkable
        class ExtendedActionSpace(Protocol):
            def click(self, element_id: str) -> str: ...
            def type_text(self, element_id: str, text: str) -> str: ...
            def noop(self) -> None: ...

        class ExtendedTool(Tool):
            action_space = ExtendedActionSpace

            def click(self, element_id: str) -> str:
                """Click.

                Args:
                    element_id: Element.

                Returns:
                    Result.
                """
                return f"Clicked {element_id}"

            def type_text(self, element_id: str, text: str) -> str:
                """Type.

                Args:
                    element_id: Element.
                    text: Text.

                Returns:
                    Result.
                """
                return f"Typed {text}"

            def noop(self) -> None:
                """Do nothing.

                Returns:
                    Nothing.
                """
                pass

        tool = ExtendedTool()
        action = Action(name="noop", arguments={})
        result = tool.execute_action(action)
        assert result == "Success"

    def test_tool_execute_action_error_handling(self, mock_tool):
        """Test that execute_action handles errors gracefully."""

        # Override click to raise an error
        def raise_error(element_id: str) -> str:
            raise ValueError("Element not found")

        original_click = mock_tool.click
        mock_tool.click = raise_error

        action = Action(name="click", arguments={"element_id": "nonexistent"})
        result = mock_tool.execute_action(action)

        assert "Error executing action click" in result
        assert "Element not found" in result

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
        with pytest.raises(ValueError, match="is not a part of"):
            mock_tool.get_action_method(action)

    def test_tool_get_action_method_not_implemented(self):
        """Test getting method that's in action space but not implemented."""

        @runtime_checkable
        class PartialActionSpace(Protocol):
            def implemented(self) -> str: ...
            def not_implemented(self) -> str: ...

        class PartialTool(Tool):
            action_space = PartialActionSpace

            def implemented(self) -> str:
                """Implemented method.

                Returns:
                    Result.
                """
                return "done"

            # not_implemented is missing

        tool = PartialTool()
        action = Action(name="not_implemented", arguments={})
        with pytest.raises(ValueError, match="is not implemented"):
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

        results = [mock_tool.execute_action(a) for a in actions]

        assert mock_tool.click_count == 2
        assert len(mock_tool.typed_texts) == 1
        assert "btn1" in results[0]
        assert "btn2" in results[1]
        assert "test" in results[2]
