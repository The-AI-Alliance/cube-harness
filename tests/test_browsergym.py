"""Tests for agentlab2.tools.browsergym module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from agentlab2.core import Action, Observation
from agentlab2.tools.browsergym import BrowsergymConfig, BrowsergymTool


class TestBrowsergymConfig:
    """Tests for BrowsergymConfig."""

    def test_default_config_values(self) -> None:
        """Test that default configuration values are set correctly."""
        config = BrowsergymConfig()

        assert config.headless is True
        assert config.use_html is True
        assert config.use_axtree is True
        assert config.use_screenshot is True
        assert config.prune_html is True
        assert config.max_wait == 60
        assert config.task_entrypoint is None
        assert config.task_kwargs == {}

    def test_custom_config_values(self) -> None:
        """Test that custom configuration values are applied."""
        config = BrowsergymConfig(
            headless=False,
            use_html=False,
            use_axtree=False,
            use_screenshot=False,
            max_wait=30,
            viewport={"width": 1920, "height": 1080},
        )

        assert config.headless is False
        assert config.use_html is False
        assert config.use_axtree is False
        assert config.use_screenshot is False
        assert config.max_wait == 30
        assert config.viewport == {"width": 1920, "height": 1080}

    def test_make_creates_tool_instance(self) -> None:
        """Test that make() creates a proper BrowsergymTool instance."""
        config = BrowsergymConfig()
        tool = config.make()

        assert isinstance(tool, BrowsergymTool)
        assert tool.config is config

    def test_make_passes_config_to_tool(self) -> None:
        """Test that make() passes configuration to the tool."""
        config = BrowsergymConfig(headless=False, max_wait=120)
        tool = config.make()

        assert tool.config.headless is False
        assert tool.config.max_wait == 120


class TestBrowsergymToolInitialization:
    """Tests for BrowsergymTool initialization and lifecycle."""

    def test_tool_init_sets_config(self) -> None:
        """Test that tool initialization sets the config."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        assert tool.config is config
        assert tool._env is None
        assert tool._last_obs is None
        assert tool._last_info is None

    def test_ensure_env_raises_when_not_initialized(self) -> None:
        """Test that _ensure_env raises RuntimeError when env not initialized."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        with pytest.raises(RuntimeError, match="BrowserGym environment is not initialized"):
            tool._ensure_env()

    def test_env_property_raises_when_not_initialized(self) -> None:
        """Test that env property raises RuntimeError when not initialized."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        with pytest.raises(RuntimeError, match="BrowserGym environment is not initialized"):
            _ = tool.env

    def test_page_property_raises_when_not_initialized(self) -> None:
        """Test that page property raises RuntimeError when not initialized."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        with pytest.raises(RuntimeError, match="BrowserGym environment is not initialized"):
            _ = tool.page

    def test_page_obs_raises_when_not_initialized(self) -> None:
        """Test that page_obs raises RuntimeError when no observation available."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        with pytest.raises(RuntimeError, match="BrowserGym environment is not initialized"):
            tool.page_obs()

    def test_goto_raises_when_not_initialized(self) -> None:
        """Test that goto raises RuntimeError when env not initialized."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        with pytest.raises(RuntimeError, match="BrowserGym environment is not initialized"):
            tool.goto("https://example.com")

    def test_evaluate_js_raises_when_not_initialized(self) -> None:
        """Test that evaluate_js raises RuntimeError when env not initialized."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        with pytest.raises(RuntimeError, match="BrowserGym environment is not initialized"):
            tool.evaluate_js("return 1+1")


class TestBrowsergymToolObservationConversion:
    """Tests for _bgym_obs_to_agentlab_obs conversion."""

    def test_empty_observation(self) -> None:
        """Test conversion of empty BrowserGym observation."""
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=False)
        tool = BrowsergymTool(config)

        bgym_obs: dict = {}
        obs = tool._bgym_obs_to_agentlab_obs(bgym_obs)

        assert isinstance(obs, Observation)
        assert len(obs.contents) == 0

    @patch("agentlab2.tools.browsergym.flatten_dom_to_str")
    def test_html_observation(self, mock_flatten_dom: MagicMock) -> None:
        """Test conversion of HTML observation."""
        mock_flatten_dom.return_value = "<html><body>Test</body></html>"

        config = BrowsergymConfig(use_html=True, use_axtree=False, use_screenshot=False, prune_html=False)
        tool = BrowsergymTool(config)

        dom_obj = {"documents": [], "strings": []}
        bgym_obs = {"dom_object": dom_obj}
        obs = tool._bgym_obs_to_agentlab_obs(bgym_obs)

        assert len(obs.contents) == 1
        assert obs.contents[0].name == "pruned_html"
        assert obs.contents[0].data == "<html><body>Test</body></html>"

    @patch("agentlab2.tools.browsergym.prune_html")
    @patch("agentlab2.tools.browsergym.flatten_dom_to_str")
    def test_html_observation_with_pruning(self, mock_flatten_dom: MagicMock, mock_prune_html: MagicMock) -> None:
        """Test that HTML is pruned when prune_html is True."""
        mock_flatten_dom.return_value = "<html><body>Full HTML</body></html>"
        mock_prune_html.return_value = "<body>Pruned</body>"

        config = BrowsergymConfig(use_html=True, use_axtree=False, use_screenshot=False, prune_html=True)
        tool = BrowsergymTool(config)

        bgym_obs = {"dom_object": {"documents": [], "strings": []}}
        obs = tool._bgym_obs_to_agentlab_obs(bgym_obs)

        mock_prune_html.assert_called_once_with("<html><body>Full HTML</body></html>")
        assert obs.contents[0].data == "<body>Pruned</body>"

    @patch("agentlab2.tools.browsergym.flatten_axtree_to_str")
    def test_axtree_observation(self, mock_flatten_axtree: MagicMock) -> None:
        """Test conversion of accessibility tree observation."""
        mock_flatten_axtree.return_value = "[a1] button 'Submit'"

        config = BrowsergymConfig(use_html=False, use_axtree=True, use_screenshot=False)
        tool = BrowsergymTool(config)

        axtree_obj = {"nodes": []}
        bgym_obs = {"axtree_object": axtree_obj}
        obs = tool._bgym_obs_to_agentlab_obs(bgym_obs)

        assert len(obs.contents) == 1
        assert obs.contents[0].name == "axtree_txt"
        assert obs.contents[0].data == "[a1] button 'Submit'"

    def test_axtree_observation_empty_object(self) -> None:
        """Test that empty axtree_object is skipped."""
        config = BrowsergymConfig(use_html=False, use_axtree=True, use_screenshot=False)
        tool = BrowsergymTool(config)

        bgym_obs = {"axtree_object": None}
        obs = tool._bgym_obs_to_agentlab_obs(bgym_obs)

        assert len(obs.contents) == 0

    def test_screenshot_observation_from_numpy(self) -> None:
        """Test conversion of numpy array screenshot."""
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=True)
        tool = BrowsergymTool(config)

        # Create a 100x100 RGB numpy array
        screenshot_array = np.zeros((100, 100, 3), dtype=np.uint8)
        screenshot_array[:, :, 0] = 255  # Red channel

        bgym_obs = {"screenshot": screenshot_array}
        obs = tool._bgym_obs_to_agentlab_obs(bgym_obs)

        assert len(obs.contents) == 1
        assert obs.contents[0].name == "screenshot"
        assert isinstance(obs.contents[0].data, Image.Image)
        assert obs.contents[0].data.size == (100, 100)

    def test_screenshot_observation_from_pil_image(self) -> None:
        """Test conversion of PIL Image screenshot."""
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=True)
        tool = BrowsergymTool(config)

        screenshot_img = Image.new("RGB", (200, 150), color="blue")
        bgym_obs = {"screenshot": screenshot_img}
        obs = tool._bgym_obs_to_agentlab_obs(bgym_obs)

        assert len(obs.contents) == 1
        assert obs.contents[0].name == "screenshot"
        assert obs.contents[0].data is screenshot_img

    @patch("agentlab2.tools.browsergym.flatten_axtree_to_str")
    @patch("agentlab2.tools.browsergym.flatten_dom_to_str")
    def test_full_observation(self, mock_flatten_dom: MagicMock, mock_flatten_axtree: MagicMock) -> None:
        """Test conversion with all observation types enabled."""
        mock_flatten_dom.return_value = "<html>...</html>"
        mock_flatten_axtree.return_value = "[a1] button"

        config = BrowsergymConfig(use_html=True, use_axtree=True, use_screenshot=True, prune_html=False)
        tool = BrowsergymTool(config)

        screenshot_img = Image.new("RGB", (100, 100), color="green")
        bgym_obs = {
            "dom_object": {"documents": []},
            "axtree_object": {"nodes": []},
            "screenshot": screenshot_img,
        }
        obs = tool._bgym_obs_to_agentlab_obs(bgym_obs)

        assert len(obs.contents) == 3
        content_names = {c.name for c in obs.contents}
        assert content_names == {"pruned_html", "axtree_txt", "screenshot"}


class TestBrowsergymToolActionMethods:
    """Tests for action method implementations."""

    def _create_tool_with_mock_env(self) -> BrowsergymTool:
        """Helper to create a tool with a mocked environment."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        # Mock the environment and step method
        mock_env = MagicMock()
        mock_env.step.return_value = ({}, 0.0, False, False, {})
        tool._env = mock_env
        tool._last_obs = {}

        return tool

    def test_browser_click_action_string(self) -> None:
        """Test that browser_click constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        tool.browser_click("a51")

        tool._env.step.assert_called_once_with('click(bid="a51")')

    def test_browser_type_action_string(self) -> None:
        """Test that browser_type constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        tool.browser_type("b12", "Hello World")

        tool._env.step.assert_called_once_with('fill(bid="b12", value="Hello World")')

    def test_browser_type_escapes_quotes(self) -> None:
        """Test that browser_type properly escapes quotes in text."""
        tool = self._create_tool_with_mock_env()

        tool.browser_type("c1", 'Say "Hello"')

        tool._env.step.assert_called_once_with('fill(bid="c1", value="Say \\"Hello\\"")')

    def test_browser_type_escapes_backslashes(self) -> None:
        """Test that browser_type properly escapes backslashes."""
        tool = self._create_tool_with_mock_env()

        tool.browser_type("d1", "path\\to\\file")

        tool._env.step.assert_called_once_with('fill(bid="d1", value="path\\\\to\\\\file")')

    def test_browser_press_key_action_string(self) -> None:
        """Test that browser_press_key constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        tool.browser_press_key("Enter")

        tool._env.step.assert_called_once_with('keyboard_press("Enter")')

    def test_browser_drag_action_string(self) -> None:
        """Test that browser_drag constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        tool.browser_drag("e1", "f2")

        tool._env.step.assert_called_once_with('drag_and_drop(from_bid="e1", to_bid="f2")')

    def test_browser_hover_action_string(self) -> None:
        """Test that browser_hover constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        tool.browser_hover("g3")

        tool._env.step.assert_called_once_with('hover(bid="g3")')

    def test_browser_select_option_action_string(self) -> None:
        """Test that browser_select_option constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        tool.browser_select_option("h4", "option1")

        tool._env.step.assert_called_once_with('select_option(bid="h4", options="option1")')

    def test_browser_select_option_escapes_quotes(self) -> None:
        """Test that browser_select_option escapes quotes in value."""
        tool = self._create_tool_with_mock_env()

        tool.browser_select_option("i5", 'value "with" quotes')

        tool._env.step.assert_called_once_with('select_option(bid="i5", options="value \\"with\\" quotes")')

    def test_browser_mouse_click_xy_action_string(self) -> None:
        """Test that browser_mouse_click_xy constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        tool.browser_mouse_click_xy(100, 200)

        tool._env.step.assert_called_once_with("mouse_click(x=100, y=200)")

    def test_browser_wait_action_string(self) -> None:
        """Test that browser_wait constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        tool.browser_wait(5)

        tool._env.step.assert_called_once_with("noop(wait_ms=5000)")

    def test_browser_wait_respects_max_wait(self) -> None:
        """Test that browser_wait clamps to max_wait value."""
        config = BrowsergymConfig(max_wait=10)
        tool = BrowsergymTool(config)

        mock_env = MagicMock()
        mock_env.step.return_value = ({}, 0.0, False, False, {})
        tool._env = mock_env
        tool._last_obs = {}

        tool.browser_wait(120)  # Request 120 seconds, but max_wait is 10

        tool._env.step.assert_called_once_with("noop(wait_ms=10000)")

    def test_browser_back_action_string(self) -> None:
        """Test that browser_back constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        tool.browser_back()

        tool._env.step.assert_called_once_with("go_back()")

    def test_browser_forward_action_string(self) -> None:
        """Test that browser_forward constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        tool.browser_forward()

        tool._env.step.assert_called_once_with("go_forward()")

    def test_noop_action_string(self) -> None:
        """Test that noop constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        tool.noop()

        tool._env.step.assert_called_once_with("noop()")


class TestBrowsergymToolStepResults:
    """Tests for action step results and state updates."""

    def test_execute_bgym_step_returns_success(self) -> None:
        """Test that _execute_bgym_step returns success message."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        mock_env = MagicMock()
        mock_env.step.return_value = ({}, 0.0, False, False, {})
        tool._env = mock_env
        tool._last_obs = {}

        result = tool._execute_bgym_step("noop()")

        assert result == "Success"

    def test_execute_bgym_step_indicates_termination(self) -> None:
        """Test that _execute_bgym_step indicates episode termination."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        mock_env = MagicMock()
        mock_env.step.return_value = ({}, 1.0, True, False, {})  # terminated=True
        tool._env = mock_env
        tool._last_obs = {}

        result = tool._execute_bgym_step("noop()")

        assert "terminated" in result.lower()

    def test_execute_bgym_step_indicates_truncation(self) -> None:
        """Test that _execute_bgym_step indicates episode truncation."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        mock_env = MagicMock()
        mock_env.step.return_value = ({}, 0.0, False, True, {})  # truncated=True
        tool._env = mock_env
        tool._last_obs = {}

        result = tool._execute_bgym_step("noop()")

        assert "truncated" in result.lower()

    def test_execute_bgym_step_updates_last_obs(self) -> None:
        """Test that _execute_bgym_step updates _last_obs."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        new_obs = {"screenshot": np.zeros((10, 10, 3))}
        mock_env = MagicMock()
        mock_env.step.return_value = (new_obs, 0.0, False, False, {})
        tool._env = mock_env
        tool._last_obs = {}

        tool._execute_bgym_step("noop()")

        assert tool._last_obs is new_obs

    def test_execute_bgym_step_updates_last_info(self) -> None:
        """Test that _execute_bgym_step updates _last_info."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        new_info = {"step_count": 5}
        mock_env = MagicMock()
        mock_env.step.return_value = ({}, 0.0, False, False, new_info)
        tool._env = mock_env
        tool._last_obs = {}

        tool._execute_bgym_step("noop()")

        assert tool._last_info is new_info


class TestBrowsergymToolExecuteAction:
    """Tests for execute_action integration."""

    @patch("agentlab2.tools.browsergym.flatten_dom_to_str")
    def test_execute_action_returns_combined_observation(self, mock_flatten_dom: MagicMock) -> None:
        """Test that execute_action combines action result and page observation."""
        mock_flatten_dom.return_value = "<html>...</html>"

        config = BrowsergymConfig(use_html=True, use_axtree=False, use_screenshot=False, prune_html=False)
        tool = BrowsergymTool(config)

        mock_env = MagicMock()
        mock_env.step.return_value = ({"dom_object": {}}, 0.0, False, False, {})
        tool._env = mock_env
        tool._last_obs = {"dom_object": {}}

        action = Action(name="browser_click", arguments={"bid": "a1"})
        obs = tool.execute_action(action)

        assert isinstance(obs, Observation)
        # Should have action result + HTML content
        assert len(obs.contents) >= 1


class TestBrowsergymToolLifecycle:
    """Tests for tool lifecycle methods."""

    @patch("agentlab2.tools.browsergym.BrowserEnv")
    def test_reset_creates_new_env(self, mock_browser_env_cls: MagicMock) -> None:
        """Test that reset creates a new BrowserEnv."""
        mock_env = MagicMock()
        mock_env.reset.return_value = ({}, {})
        mock_browser_env_cls.return_value = mock_env

        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        tool.reset()

        mock_browser_env_cls.assert_called_once()
        mock_env.reset.assert_called_once()
        assert tool._env is mock_env

    @patch("agentlab2.tools.browsergym.BrowserEnv")
    def test_reset_closes_existing_env(self, mock_browser_env_cls: MagicMock) -> None:
        """Test that reset closes existing environment before creating new one."""
        old_env = MagicMock()
        new_env = MagicMock()
        new_env.reset.return_value = ({}, {})
        mock_browser_env_cls.return_value = new_env

        config = BrowsergymConfig()
        tool = BrowsergymTool(config)
        tool._env = old_env

        tool.reset()

        old_env.close.assert_called_once()
        assert tool._env is new_env

    def test_close_cleans_up_env(self) -> None:
        """Test that close cleans up the environment."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        mock_env = MagicMock()
        tool._env = mock_env
        tool._last_obs = {"some": "data"}
        tool._last_info = {"info": "data"}

        tool.close()

        mock_env.close.assert_called_once()
        assert tool._env is None
        assert tool._last_obs is None
        assert tool._last_info is None

    def test_close_handles_exception(self) -> None:
        """Test that close handles exceptions gracefully."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        mock_env = MagicMock()
        mock_env.close.side_effect = Exception("Close failed")
        tool._env = mock_env
        tool._last_obs = {"some": "data"}

        # Should not raise
        tool.close()

        # State should still be cleaned up
        assert tool._env is None
        assert tool._last_obs is None

    def test_close_noop_when_no_env(self) -> None:
        """Test that close is safe to call when no env exists."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        # Should not raise
        tool.close()


class TestBrowsergymToolActionSet:
    """Tests for action set property."""

    def test_action_set_contains_expected_actions(self) -> None:
        """Test that action_set contains all BidBrowserActionSpace methods."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        action_names = {a.name for a in tool.action_set}

        expected_actions = {
            "browser_press_key",
            "browser_type",
            "browser_click",
            "browser_drag",
            "browser_hover",
            "browser_select_option",
            "browser_mouse_click_xy",
            "browser_wait",
            "browser_back",
            "browser_forward",
            "noop",
        }

        assert expected_actions.issubset(action_names)
