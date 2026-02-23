"""Tests for agentlab2.environment module."""

import pytest

from agentlab2.core import Action, ActionSchema, Content, EnvironmentOutput, Observation
from agentlab2.environment import STOP_ACTION, EnvConfig, Environment
from agentlab2.tools.toolbox import Toolbox
from tests.conftest import MockTool


class TestStopAction:
    """Tests for STOP_ACTION constant."""

    def test_stop_action_name(self):
        """Test STOP_ACTION has correct name."""
        assert STOP_ACTION.name == "final_step"

    def test_stop_action_is_action_schema(self):
        """Test STOP_ACTION is an ActionSchema."""
        assert isinstance(STOP_ACTION, ActionSchema)


class TestToolboxEnv:
    """Tests for ToolboxEnv class."""

    def test_toolbox_env_creation(self, mock_task, mock_tool):
        """Test ToolboxEnv creation."""
        tool_env = Environment(task=mock_task, tool=mock_tool)
        assert tool_env.task == mock_task
        assert tool_env.tool == mock_tool

    def test_toolbox_env_actions(self, mock_task, mock_tool):
        """Test getting actions from ToolboxEnv."""
        mock_tool_env = Environment(task=mock_task, tool=mock_tool)
        actions = mock_tool_env.action_set

        # Should have actions from mock_tool, filtered by task
        assert len(actions) == 2
        action_names = {a.name for a in actions}
        assert "click" in action_names
        assert "type_text" in action_names

    def test_toolbox_env_setup(self, mock_tool_env, mock_task):
        """Test ToolboxEnv setup."""
        env_output = mock_tool_env.setup()

        assert mock_task.setup_called
        assert isinstance(env_output, EnvironmentOutput)
        assert mock_task.goal in env_output.obs.contents[0].data

    def test_toolbox_env_step_single_action(self, mock_tool_env, mock_tool):
        """Test stepping with a single action."""
        mock_tool_env.setup()

        action = Action(id="a1", name="click", arguments={"element_id": "btn1"})
        env_output = mock_tool_env.step(action)

        assert isinstance(env_output, EnvironmentOutput)
        assert len(env_output.obs.contents) == 1
        assert "Clicked on btn1" in str(env_output.obs.contents[0].data)
        assert mock_tool.click_count == 1

    def test_toolbox_env_step_multiple_actions(self, mock_tool_env, mock_tool):
        """Test stepping with multiple actions."""
        mock_tool_env.setup()

        actions = [
            Action(id="a1", name="click", arguments={"element_id": "btn1"}),
            Action(id="a2", name="type_text", arguments={"element_id": "input1", "text": "test"}),
        ]
        env_output = mock_tool_env.step(actions)

        assert len(env_output.obs.contents) == 2
        assert mock_tool.click_count == 1
        assert len(mock_tool.typed_texts) == 1

    def test_toolbox_env_step_stop_action(self, mock_tool_env):
        """Test stepping with stop action."""
        mock_tool_env.setup()

        action = Action(name="final_step", arguments={})
        env_output = mock_tool_env.step(action)

        assert env_output.done is True
        assert "finished" in env_output.obs.contents[0].data.lower()

    def test_toolbox_env_step_stop_action_stops_further_actions(self, mock_tool_env, mock_tool):
        """Test that stop action stops processing further actions."""
        mock_tool_env.setup()

        actions = [
            Action(name="final_step", arguments={}),
            Action(name="click", arguments={"element_id": "btn1"}),  # Should not execute
        ]
        env_output = mock_tool_env.step(actions)

        assert env_output.done is True
        assert mock_tool.click_count == 0  # Click should not have executed

    def test_toolbox_env_step_unsupported_action(self, mock_tool_env):
        """Test stepping with unsupported action returns error observation."""
        mock_tool_env.setup()

        action = Action(name="nonexistent_action", arguments={})
        env_output = mock_tool_env.step(action)
        assert env_output.obs.has_error is True
        assert "nonexistent_action" in env_output.obs.contents[0].data

    def test_toolbox_env_step_validates_when_done(self, mock_tool_env, mock_task):
        """Test that validation is called when done."""
        mock_tool_env.setup()

        action = Action(name="final_step", arguments={})
        env_output = mock_tool_env.step(action)

        assert mock_task.validate_called
        assert env_output.reward == 1.0  # MockTask returns 1.0

    def test_toolbox_env_step_validates_per_step(self, mock_task, mock_tool):
        """Test validation per step when enabled."""
        mock_task.validate_per_step = True
        env = Environment(task=mock_task, tool=mock_tool)
        env.setup()

        action = Action(id="a1", name="click", arguments={"element_id": "btn1"})
        env_output = env.step(action)

        assert mock_task.validate_called
        assert env_output.reward == 1.0

    def test_toolbox_env_close(self, mock_tool_env, mock_task):
        """Test ToolboxEnv close."""
        mock_tool_env.setup()
        mock_tool_env.close()

        assert mock_task.teardown_called

    def test_toolbox_env_tool_call_id_in_content(self, mock_tool_env):
        """Test that tool_call_id is preserved in content."""
        mock_tool_env.setup()

        action = Action(id="call_123", name="click", arguments={"element_id": "btn1"})
        env_output = mock_tool_env.step(action)

        assert env_output.obs.contents[0].tool_call_id == "call_123"

    def test_toolbox_env_multiple_tools(self, mock_task):
        """Test ToolboxEnv with multiple tools."""

        tool1 = MockTool()
        tool2 = MockTool()

        env = Environment(task=mock_task, tool=Toolbox(tools=[tool1, tool2]))
        actions = env.action_set

        # Both tools have same actions, so we should see them (from first match)
        action_names = {a.name for a in actions}
        assert "click" in action_names

    def test_toolbox_env_task_finished(self, mock_task, mock_tool):
        """Test that task.finished() is checked."""

        class FinishingTask(type(mock_task)):
            def finished(self):
                return True

        task = FinishingTask()
        env = Environment(task=task, tool=mock_tool)
        env.setup()

        action = Action(id="a1", name="click", arguments={"element_id": "btn1"})
        env_output = env.step(action)

        assert env_output.done is True

    def test_toolbox_env_obs_postprocess(self, mock_task, mock_tool):
        """Test that obs_postprocess is called."""

        class PostprocessTask(type(mock_task)):
            def obs_postprocess(self, obs: Observation) -> Observation:
                # Add a marker to observation
                obs.contents.append(Content(data="postprocessed"))
                return obs

        task = PostprocessTask()
        env = Environment(task=task, tool=mock_tool)
        env.setup()

        action = Action(id="a1", name="click", arguments={"element_id": "btn1"})
        env_output = env.step(action)

        assert any("postprocessed" in str(c.data) for c in env_output.obs.contents)


class TestTask:
    """Tests for Task abstract class behavior through MockTask."""

    def test_task_id(self, mock_task):
        """Test task has id."""
        assert mock_task.id == "mock_task_1"

    def test_task_setup(self, mock_tool_env, mock_task):
        """Test task setup."""
        obs, info = mock_task.setup(mock_tool_env)

        assert obs.contents[0].data == "Complete the test task"
        assert info == {"task_type": "mock"}

    def test_task_teardown(self, mock_tool_env, mock_task):
        """Test task teardown."""
        mock_task.teardown()
        assert mock_task.teardown_called

    def test_task_validate(self, mock_tool_env, mock_task):
        """Test task validation."""
        obs = Observation.from_text("done")
        reward, info = mock_task.validate_task(mock_tool_env, obs)

        assert reward == 1.0
        assert info == {"success": True}

    def test_task_filter_actions(self, mock_task):
        """Test task filter_actions passes through all actions."""
        actions = [
            ActionSchema(name="a1", description="Action 1"),
            ActionSchema(name="a2", description="Action 2"),
        ]
        filtered = mock_task.filter_actions(actions)
        assert filtered == actions

    def test_task_cheat_not_implemented(self, mock_task):
        """Test task cheat raises NotImplementedError by default."""
        with pytest.raises(NotImplementedError):
            mock_task.cheat()

    def test_task_finished_default(self, mock_tool_env, mock_task):
        """Test task finished returns False by default."""
        assert mock_task.finished() is False

    def test_task_obs_postprocess_default(self, mock_task):
        """Test obs_postprocess returns obs unchanged by default."""
        obs = Observation.from_text("test")
        result = mock_task.obs_postprocess(obs)
        assert result == obs


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig through MockEnvironmentConfig."""

    def test_env_config_make(self, mock_task, mock_tool_config):
        """Test creating environment from config."""
        env_config = EnvConfig(task=mock_task, tool_config=mock_tool_config)
        env = env_config.make()
        assert isinstance(env, Environment)
        assert env.task == mock_task
        assert type(env.tool) is MockTool
