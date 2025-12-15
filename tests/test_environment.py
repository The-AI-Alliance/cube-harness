"""Tests for agentlab2.environment module."""

import pytest

from agentlab2.core import Action, ActionSchema, Content, EnvironmentOutput, Observation
from agentlab2.environment import STOP_ACTION, ToolboxEnv
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
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        assert env.task == mock_task
        assert len(env.tools) == 1

    def test_toolbox_env_actions(self, mock_task, mock_tool):
        """Test getting actions from ToolboxEnv."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        actions = env.action_set()

        # Should have actions from mock_tool, filtered by task
        assert len(actions) == 2
        action_names = {a.name for a in actions}
        assert "click" in action_names
        assert "type_text" in action_names

    def test_toolbox_env_setup(self, mock_task, mock_tool):
        """Test ToolboxEnv setup."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        env_output = env.setup()

        assert mock_task.setup_called
        assert isinstance(env_output, EnvironmentOutput)
        assert mock_task.goal in env_output.obs.contents[0].data

    def test_toolbox_env_step_single_action(self, mock_task, mock_tool):
        """Test stepping with a single action."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        env.setup()

        action = Action(id="a1", name="click", arguments={"element_id": "btn1"})
        env_output = env.step(action)

        assert isinstance(env_output, EnvironmentOutput)
        assert len(env_output.obs.contents) == 1
        assert "Clicked on btn1" in env_output.obs.contents[0].data
        assert mock_tool.click_count == 1

    def test_toolbox_env_step_multiple_actions(self, mock_task, mock_tool):
        """Test stepping with multiple actions."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        env.setup()

        actions = [
            Action(id="a1", name="click", arguments={"element_id": "btn1"}),
            Action(id="a2", name="type_text", arguments={"element_id": "input1", "text": "test"}),
        ]
        env_output = env.step(actions)

        assert len(env_output.obs.contents) == 2
        assert mock_tool.click_count == 1
        assert len(mock_tool.typed_texts) == 1

    def test_toolbox_env_step_stop_action(self, mock_task, mock_tool):
        """Test stepping with stop action."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        env.setup()

        action = Action(name="final_step", arguments={})
        env_output = env.step(action)

        assert env_output.done is True
        assert "finished" in env_output.obs.contents[0].data.lower()

    def test_toolbox_env_step_stop_action_stops_further_actions(self, mock_task, mock_tool):
        """Test that stop action stops processing further actions."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        env.setup()

        actions = [
            Action(name="final_step", arguments={}),
            Action(name="click", arguments={"element_id": "btn1"}),  # Should not execute
        ]
        env_output = env.step(actions)

        assert env_output.done is True
        assert mock_tool.click_count == 0  # Click should not have executed

    def test_toolbox_env_step_unsupported_action(self, mock_task, mock_tool):
        """Test stepping with unsupported action raises error."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        env.setup()

        action = Action(name="nonexistent_action", arguments={})
        with pytest.raises(ValueError, match="is not supported"):
            env.step(action)

    def test_toolbox_env_step_validates_when_done(self, mock_task, mock_tool):
        """Test that validation is called when done."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        env.setup()

        action = Action(name="final_step", arguments={})
        env_output = env.step(action)

        assert mock_task.validate_called
        assert env_output.reward == 1.0  # MockTask returns 1.0

    def test_toolbox_env_step_validates_per_step(self, mock_task, mock_tool):
        """Test validation per step when enabled."""
        mock_task.validate_per_step = True
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        env.setup()

        action = Action(id="a1", name="click", arguments={"element_id": "btn1"})
        env_output = env.step(action)

        assert mock_task.validate_called
        assert env_output.reward == 1.0

    def test_toolbox_env_is_stop_action(self, mock_task, mock_tool):
        """Test is_stop_action method."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])

        stop = Action(name="final_step", arguments={})
        not_stop = Action(name="click", arguments={})

        assert env.is_stop_action(stop) is True
        assert env.is_stop_action(not_stop) is False

    def test_toolbox_env_close(self, mock_task, mock_tool):
        """Test ToolboxEnv close."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        env.setup()
        env.close()

        assert mock_task.teardown_called

    def test_toolbox_env_tool_call_id_in_content(self, mock_task, mock_tool):
        """Test that tool_call_id is preserved in content."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        env.setup()

        action = Action(id="call_123", name="click", arguments={"element_id": "btn1"})
        env_output = env.step(action)

        assert env_output.obs.contents[0].tool_call_id == "call_123"

    def test_toolbox_env_multiple_tools(self, mock_task):
        """Test ToolboxEnv with multiple tools."""

        tool1 = MockTool()
        tool2 = MockTool()

        env = ToolboxEnv(task=mock_task, tools=[tool1, tool2])
        actions = env.action_set()

        # Both tools have same actions, so we should see them (from first match)
        action_names = {a.name for a in actions}
        assert "click" in action_names

    def test_toolbox_env_task_finished(self, mock_task, mock_tool):
        """Test that task.finished() is checked."""

        class FinishingTask(type(mock_task)):
            def finished(self, env):
                return True

        task = FinishingTask()
        env = ToolboxEnv(task=task, tools=[mock_tool])
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
        env = ToolboxEnv(task=task, tools=[mock_tool])
        env.setup()

        action = Action(id="a1", name="click", arguments={"element_id": "btn1"})
        env_output = env.step(action)

        assert any("postprocessed" in str(c.data) for c in env_output.obs.contents)


class TestTask:
    """Tests for Task abstract class behavior through MockTask."""

    def test_task_id(self, mock_task):
        """Test task has id."""
        assert mock_task.id == "mock_task_1"

    def test_task_setup(self, mock_task, mock_tool):
        """Test task setup."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        goal, info = mock_task.setup(env)

        assert goal == "Complete the test task"
        assert info == {"task_type": "mock"}

    def test_task_teardown(self, mock_task, mock_tool):
        """Test task teardown."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        mock_task.teardown(env)
        assert mock_task.teardown_called

    def test_task_validate(self, mock_task, mock_tool):
        """Test task validation."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        obs = Observation.from_text("done")
        reward, info = mock_task.validate_task(env, obs)

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

    def test_task_finished_default(self, mock_task, mock_tool):
        """Test task finished returns False by default."""
        env = ToolboxEnv(task=mock_task, tools=[mock_tool])
        assert mock_task.finished(env) is False

    def test_task_obs_postprocess_default(self, mock_task):
        """Test obs_postprocess returns obs unchanged by default."""
        obs = Observation.from_text("test")
        result = mock_task.obs_postprocess(obs)
        assert result == obs


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig through MockEnvironmentConfig."""

    def test_env_config_make(self, mock_env_config, mock_task):
        """Test creating environment from config."""
        mock_env_config._task = mock_task
        env = mock_env_config.make()
        assert isinstance(env, ToolboxEnv)
        assert env.task == mock_task
