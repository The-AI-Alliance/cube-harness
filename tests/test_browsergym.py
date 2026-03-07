"""Tests for agentlab2.tools.browsergym module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from cube.core import Action, Observation
from PIL import Image

from agentlab2.tools.browser_session import PlaywrightSession, PlaywrightSessionConfig
from agentlab2.tools.browsergym import BrowsergymConfig, BrowsergymTool


class TestBrowsergymConfig:
    """Tests for BrowsergymConfig."""

    def test_default_config_values(self) -> None:
        """Test that default configuration values are set correctly."""
        config = BrowsergymConfig()

        assert config.browser_config.headless is True
        assert config.use_html is True
        assert config.use_axtree is True
        assert config.use_screenshot is True
        assert config.prune_html is True
        assert config.max_wait == 60

    def test_custom_config_values(self) -> None:
        """Test that custom configuration values are applied."""
        config = BrowsergymConfig(
            browser_config=PlaywrightSessionConfig(headless=False, viewport={"width": 1920, "height": 1080}),
            use_html=False,
            use_axtree=False,
            use_screenshot=False,
            max_wait=30,
        )

        assert config.browser_config.headless is False
        assert config.use_html is False
        assert config.use_axtree is False
        assert config.use_screenshot is False
        assert config.max_wait == 30
        assert config.browser_config.viewport == {"width": 1920, "height": 1080}

    def test_make_creates_tool_instance(self) -> None:
        """Test that make() creates a proper BrowsergymTool instance."""
        config = BrowsergymConfig()
        tool = config.make()

        assert isinstance(tool, BrowsergymTool)
        assert tool.config is config

    def test_make_passes_config_to_tool(self) -> None:
        """Test that make() passes configuration to the tool."""
        config = BrowsergymConfig(browser_config=PlaywrightSessionConfig(headless=False), max_wait=120)
        tool = config.make()

        assert tool.config.browser_config.headless is False
        assert tool.config.max_wait == 120


class TestBrowsergymToolInitialization:
    """Tests for BrowsergymTool initialization and lifecycle."""

    def test_tool_init_sets_config(self) -> None:
        """Test that tool initialization sets the config."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        assert tool.config is config
        assert tool.session is None
        assert tool._last_obs is None
        assert tool._last_info is None

    def test_ensure_page_raises_when_not_initialized(self) -> None:
        """Test that _ensure_page raises RuntimeError when page not initialized."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        with pytest.raises(RuntimeError, match="Browser is not initialized"):
            tool._ensure_page()

    def test_page_property_raises_when_not_initialized(self) -> None:
        """Test that page property raises RuntimeError when not initialized."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        with pytest.raises(RuntimeError, match="Browser is not initialized"):
            _ = tool.page

    def test_page_obs_raises_when_not_initialized(self) -> None:
        """Test that page_obs raises RuntimeError when no observation available."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        with pytest.raises(RuntimeError, match="Browser is not initialized"):
            tool.page_obs()

    def test_evaluate_js_raises_when_not_initialized(self) -> None:
        """Test that evaluate_js raises RuntimeError when env not initialized."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        with pytest.raises(RuntimeError, match="Browser is not initialized"):
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

    def test_focused_element_observation(self) -> None:
        """Test conversion of focused_element_bid observation field."""
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=False)
        tool = BrowsergymTool(config)

        bgym_obs = {"focused_element_bid": "a123"}
        obs = tool._bgym_obs_to_agentlab_obs(bgym_obs)

        assert len(obs.contents) == 1
        assert obs.contents[0].name == "focused_element"
        assert obs.contents[0].data == "a123"

    def test_focused_element_observation_empty(self) -> None:
        """Test that empty focused_element_bid is not added."""
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=False)
        tool = BrowsergymTool(config)

        bgym_obs = {"focused_element_bid": None}
        obs = tool._bgym_obs_to_agentlab_obs(bgym_obs)

        assert len(obs.contents) == 0

    def test_last_action_error_observation(self) -> None:
        """Test conversion of last_action_error observation field."""
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=False)
        tool = BrowsergymTool(config)

        bgym_obs = {"last_action_error": "Element not found: bid='xyz'"}
        obs = tool._bgym_obs_to_agentlab_obs(bgym_obs)

        assert len(obs.contents) == 1
        assert obs.contents[0].name == "last_action_error"
        assert obs.contents[0].data == "Element not found: bid='xyz'"

    def test_last_action_error_observation_empty(self) -> None:
        """Test that empty last_action_error is not added."""
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=False)
        tool = BrowsergymTool(config)

        bgym_obs = {"last_action_error": None}
        obs = tool._bgym_obs_to_agentlab_obs(bgym_obs)

        assert len(obs.contents) == 0

    def test_last_action_error_observation_empty_string(self) -> None:
        """Test that empty string last_action_error is not added."""
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=False)
        tool = BrowsergymTool(config)

        bgym_obs = {"last_action_error": ""}
        obs = tool._bgym_obs_to_agentlab_obs(bgym_obs)

        assert len(obs.contents) == 0


class TestBrowsergymToolActionMethods:
    """Tests for action method implementations."""

    def _create_tool_with_mock_env(self) -> BrowsergymTool:
        """Helper to create a tool with a mocked page for action testing."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)
        tool.session = PlaywrightSession(page=MagicMock(), context=MagicMock(), cdp_url="http://localhost:9222")
        return tool

    def test_browser_click_action_string(self) -> None:
        """Test that browser_click constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        with (
            patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step,
            patch.object(tool, "_get_checkbox_state", return_value=None),
        ):
            tool.browser_click("a51")

        mock_step.assert_called_once_with('click(bid="a51")')

    def test_browser_type_action_string(self) -> None:
        """Test that browser_type constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_type("b12", "Hello World")

        mock_step.assert_called_once_with('fill(bid="b12", value="Hello World")')

    def test_browser_type_escapes_quotes(self) -> None:
        """Test that browser_type properly escapes quotes in text."""
        tool = self._create_tool_with_mock_env()

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_type("c1", 'Say "Hello"')

        mock_step.assert_called_once_with('fill(bid="c1", value="Say \\"Hello\\"")')

    def test_browser_type_escapes_backslashes(self) -> None:
        """Test that browser_type properly escapes backslashes."""
        tool = self._create_tool_with_mock_env()

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_type("d1", "path\\to\\file")

        mock_step.assert_called_once_with('fill(bid="d1", value="path\\\\to\\\\file")')

    def test_browser_press_key_action_string(self) -> None:
        """Test that browser_press_key constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_press_key("Enter")

        mock_step.assert_called_once_with('keyboard_press("Enter")')

    def test_browser_drag_action_string(self) -> None:
        """Test that browser_drag constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_drag("e1", "f2")

        mock_step.assert_called_once_with('drag_and_drop(from_bid="e1", to_bid="f2")')

    def test_browser_hover_action_string(self) -> None:
        """Test that browser_hover constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_hover("g3")

        mock_step.assert_called_once_with('hover(bid="g3")')

    def test_browser_select_option_action_string(self) -> None:
        """Test that browser_select_option constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_select_option("h4", "option1")

        mock_step.assert_called_once_with('select_option(bid="h4", options="option1")')

    def test_browser_select_option_escapes_quotes(self) -> None:
        """Test that browser_select_option escapes quotes in value."""
        tool = self._create_tool_with_mock_env()

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_select_option("i5", 'value "with" quotes')

        mock_step.assert_called_once_with('select_option(bid="i5", options="value \\"with\\" quotes")')

    def test_browser_mouse_click_xy_action_string(self) -> None:
        """Test that browser_mouse_click_xy constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_mouse_click_xy(100, 200)

        mock_step.assert_called_once_with("mouse_click(x=100, y=200)")

    def test_browser_wait_action_string(self) -> None:
        """Test that browser_wait constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_wait(5)

        mock_step.assert_called_once_with("noop(wait_ms=5000)")

    def test_browser_wait_respects_max_wait(self) -> None:
        """Test that browser_wait clamps to max_wait value."""
        config = BrowsergymConfig(max_wait=10)
        tool = BrowsergymTool(config)
        tool.session = PlaywrightSession(page=MagicMock(), context=MagicMock(), cdp_url="http://localhost:9222")

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_wait(120)  # Request 120 seconds, but max_wait is 10

        mock_step.assert_called_once_with("noop(wait_ms=10000)")

    def test_browser_back_action_string(self) -> None:
        """Test that browser_back constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_back()

        mock_step.assert_called_once_with("go_back()")

    def test_browser_forward_action_string(self) -> None:
        """Test that browser_forward constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_forward()

        mock_step.assert_called_once_with("go_forward()")

    def test_noop_action_string(self) -> None:
        """Test that noop constructs correct action string."""
        tool = self._create_tool_with_mock_env()

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.noop()

        mock_step.assert_called_once_with("noop()")


class TestBrowsergymToolStepResults:
    """Tests for action step results and state updates."""

    def _create_tool_with_mock_page(self) -> BrowsergymTool:
        """Helper to create a tool with a mocked page."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)
        tool.session = PlaywrightSession(page=MagicMock(), context=MagicMock(), cdp_url="http://localhost:9222")
        tool._action_set = MagicMock()
        return tool

    def test_execute_bgym_step_returns_success(self) -> None:
        """Test that _execute_bgym_step returns success message."""
        tool = self._create_tool_with_mock_page()

        with (
            patch("agentlab2.tools.browsergym.execute_python_code"),
            patch.object(tool, "_extract_bgym_obs", return_value={}),
        ):
            result = tool._execute_bgym_step("noop()")

        assert result == "Success"

    def test_execute_bgym_step_returns_failure_on_exception(self) -> None:
        """Test that _execute_bgym_step returns failure message on exception."""
        tool = self._create_tool_with_mock_page()

        with (
            patch("agentlab2.tools.browsergym.execute_python_code", side_effect=Exception("Some error")),
            patch.object(tool, "_extract_bgym_obs", return_value={}),
        ):
            result = tool._execute_bgym_step("noop()")

        assert "Failed" in result

    def test_execute_bgym_step_updates_last_obs(self) -> None:
        """Test that _execute_bgym_step updates _last_obs."""
        tool = self._create_tool_with_mock_page()
        new_obs = {"screenshot": np.zeros((10, 10, 3))}

        with (
            patch("agentlab2.tools.browsergym.execute_python_code"),
            patch.object(tool, "_extract_bgym_obs", return_value=new_obs),
        ):
            tool._execute_bgym_step("noop()")

        assert tool._last_obs is new_obs

    def test_execute_bgym_step_updates_last_info(self) -> None:
        """Test that _execute_bgym_step updates _last_info with source=action."""
        tool = self._create_tool_with_mock_page()

        with (
            patch("agentlab2.tools.browsergym.execute_python_code"),
            patch.object(tool, "_extract_bgym_obs", return_value={}),
        ):
            tool._execute_bgym_step("noop()")

        assert tool._last_info is not None
        assert tool._last_info["source"] == "action"


class TestBrowsergymToolExecuteAction:
    """Tests for execute_action integration."""

    def test_execute_action_returns_combined_observation(self) -> None:
        """Test that execute_action combines action result and page observation."""
        config = BrowsergymConfig(use_html=True, use_axtree=False, use_screenshot=False, prune_html=False)
        tool = BrowsergymTool(config)
        tool.session = PlaywrightSession(page=MagicMock(), context=MagicMock(), cdp_url="http://localhost:9222")

        action = Action(name="browser_click", arguments={"bid": "a1"})

        with (
            patch.object(tool, "_get_checkbox_state", return_value=None),
            patch.object(tool, "_execute_bgym_step", return_value="Success"),
            patch.object(tool, "page_obs", return_value=Observation()),
        ):
            obs = tool.execute_action(action)

        assert isinstance(obs, Observation)
        # Should have at least the action result content
        assert len(obs.contents) >= 1


class TestBrowsergymToolLifecycle:
    """Tests for tool lifecycle methods."""

    def test_reset_initializes_browser(self) -> None:
        """Test that reset calls runtime creation and observation extraction."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        with (
            patch.object(tool, "_close_runtime") as mock_close,
            patch.object(tool, "_create_runtime") as mock_create,
            patch.object(tool, "_wait_dom_loaded") as mock_wait,
            patch.object(tool, "_extract_bgym_obs", return_value={}) as mock_extract,
        ):
            tool.reset()

        mock_close.assert_called_once()
        mock_create.assert_called_once()
        mock_wait.assert_called_once()
        mock_extract.assert_called_once()

    def test_reset_closes_existing_runtime(self) -> None:
        """Test that reset closes existing runtime before creating new one."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)
        tool.session = PlaywrightSession(page=MagicMock(), context=MagicMock(), cdp_url="http://localhost:9222")

        with (
            patch.object(tool, "_close_runtime") as mock_close,
            patch.object(tool, "_create_runtime"),
            patch.object(tool, "_wait_dom_loaded"),
            patch.object(tool, "_extract_bgym_obs", return_value={}),
        ):
            tool.reset()

        mock_close.assert_called_once()

    def test_close_cleans_up_browser(self) -> None:
        """Test that close cleans up the browser resources."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        tool.session = PlaywrightSession(page=MagicMock(), context=MagicMock(), cdp_url="http://localhost:9222")
        tool._last_obs = {"some": "data"}
        tool._last_info = {"info": "data"}

        tool.close()

        assert tool.session is None
        assert tool._last_obs is None
        assert tool._last_info is None

    def test_close_handles_exception(self) -> None:
        """Test that close handles exceptions gracefully."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        mock_context = MagicMock()
        mock_context.close.side_effect = Exception("Close failed")
        tool.session = PlaywrightSession(page=MagicMock(), context=mock_context, cdp_url="http://localhost:9222")
        tool._last_obs = {"some": "data"}

        # Should not raise
        tool.close()

        # State should still be cleaned up
        assert tool.session is None
        assert tool._last_obs is None

    def test_close_noop_when_no_browser(self) -> None:
        """Test that close is safe to call when no browser exists."""
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


class TestBrowsergymToolCheckboxJsFallback:
    """Tests for checkbox/radio JS fallback in browser_click."""

    def _create_tool_with_mock_page(self) -> BrowsergymTool:
        """Helper to create a tool with a mocked page.

        Creates a mock that handles the frame navigation chain:
        1. _get_frame_for_bid(bid) parses BID and navigates to iframe
        2. For BID "a123": page.get_by_test_id("a") -> frame_locator(":scope") -> frame
        3. frame.get_by_test_id("a123") -> element locator
        """
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        mock_page = MagicMock()

        # Create the element locator (final target)
        mock_element_locator = MagicMock()
        mock_element_locator.count.return_value = 1

        # Create the iframe locator that returns the element locator
        mock_iframe_locator = MagicMock()
        mock_iframe_locator.count.return_value = 1

        # Create the frame (returned by frame_locator)
        mock_frame = MagicMock()
        mock_frame.get_by_test_id.return_value = mock_element_locator

        # Chain: iframe_locator.frame_locator(":scope") -> frame
        mock_iframe_locator.frame_locator.return_value = mock_frame

        # page.get_by_test_id returns either iframe locator or element locator
        # depending on whether it's called with iframe bid or element bid
        mock_page.get_by_test_id.return_value = mock_iframe_locator

        # Store references for tests to configure return values
        mock_page._mock_element_locator = mock_element_locator
        mock_page._mock_frame = mock_frame

        tool.session = PlaywrightSession(page=mock_page, context=MagicMock(), cdp_url="http://localhost:9222")
        tool._last_obs = {}

        return tool

    def test_get_checkbox_state_returns_true_when_checked(self) -> None:
        """Test that _get_checkbox_state returns True for checked checkbox."""
        tool = self._create_tool_with_mock_page()
        # Configure evaluate to return checkbox state
        tool.page._mock_element_locator.evaluate.return_value = {
            "found": True,
            "isCheckbox": True,
            "checked": True,
        }

        result = tool._get_checkbox_state("a123")

        assert result is True
        tool.page._mock_element_locator.evaluate.assert_called_once()

    def test_get_checkbox_state_returns_false_when_unchecked(self) -> None:
        """Test that _get_checkbox_state returns False for unchecked checkbox."""
        tool = self._create_tool_with_mock_page()
        tool.page._mock_element_locator.evaluate.return_value = {
            "found": True,
            "isCheckbox": True,
            "checked": False,
        }

        result = tool._get_checkbox_state("a123")

        assert result is False

    def test_get_checkbox_state_returns_none_for_non_checkbox(self) -> None:
        """Test that _get_checkbox_state returns None for non-checkbox elements."""
        tool = self._create_tool_with_mock_page()
        tool.page._mock_element_locator.evaluate.return_value = {
            "found": True,
            "isCheckbox": False,
            "tagName": "DIV",
        }

        result = tool._get_checkbox_state("b456")

        assert result is None

    def test_get_checkbox_state_returns_none_when_element_not_found(self) -> None:
        """Test that _get_checkbox_state returns None when element not found."""
        tool = self._create_tool_with_mock_page()
        tool.page._mock_element_locator.count.return_value = 0

        result = tool._get_checkbox_state("c789")

        assert result is None

    def test_get_checkbox_state_returns_none_on_exception(self) -> None:
        """Test that _get_checkbox_state returns None when evaluate raises."""
        tool = self._create_tool_with_mock_page()
        tool.page._mock_element_locator.evaluate.side_effect = Exception("JS error")

        result = tool._get_checkbox_state("c789")

        assert result is None

    def test_toggle_checkbox_js_sets_checked_true(self) -> None:
        """Test that _toggle_checkbox_js sets checkbox to checked."""
        tool = self._create_tool_with_mock_page()
        tool.page._mock_element_locator.evaluate.return_value = "toggled_checkbox"

        tool._toggle_checkbox_js("a123", True)

        tool.page._mock_element_locator.evaluate.assert_called_once()
        call_args = tool.page._mock_element_locator.evaluate.call_args
        js_code = call_args[0][0]
        checked_arg = call_args[0][1]
        assert "elem.checked = checked" in js_code
        assert checked_arg is True

    def test_toggle_checkbox_js_sets_checked_false(self) -> None:
        """Test that _toggle_checkbox_js sets checkbox to unchecked."""
        tool = self._create_tool_with_mock_page()
        tool.page._mock_element_locator.evaluate.return_value = "toggled_checkbox"

        tool._toggle_checkbox_js("a123", False)

        call_args = tool.page._mock_element_locator.evaluate.call_args
        checked_arg = call_args[0][1]
        assert checked_arg is False

    def test_toggle_checkbox_js_dispatches_events(self) -> None:
        """Test that _toggle_checkbox_js dispatches click, change, and input events."""
        tool = self._create_tool_with_mock_page()
        tool.page._mock_element_locator.evaluate.return_value = "toggled_checkbox"

        tool._toggle_checkbox_js("a123", True)

        call_args = tool.page._mock_element_locator.evaluate.call_args
        js_code = call_args[0][0]
        assert "dispatchEvent(new Event('click'" in js_code
        assert "dispatchEvent(new Event('change'" in js_code
        assert "dispatchEvent(new Event('input'" in js_code

    def test_toggle_checkbox_js_handles_exception_gracefully(self) -> None:
        """Test that _toggle_checkbox_js doesn't raise on exception."""
        tool = self._create_tool_with_mock_page()
        tool.page._mock_element_locator.evaluate.side_effect = Exception("JS error")

        # Should not raise
        tool._toggle_checkbox_js("a123", True)

    def test_browser_click_no_fallback_for_non_checkbox(self) -> None:
        """Test that browser_click doesn't use JS fallback for non-checkbox elements."""
        tool = self._create_tool_with_mock_page()
        # _get_checkbox_state returns None for non-checkboxes
        tool.page._mock_element_locator.evaluate.return_value = {
            "found": True,
            "isCheckbox": False,
            "tagName": "BUTTON",
        }

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_click("a100")

        # Should only call _execute_bgym_step once (no JS fallback)
        mock_step.assert_called_once_with('click(bid="a100")')
        # evaluate called once for _get_checkbox_state
        assert tool.page._mock_element_locator.evaluate.call_count == 1

    def test_browser_click_no_fallback_when_native_click_works(self) -> None:
        """Test that browser_click doesn't use JS fallback when native click toggles state."""
        tool = self._create_tool_with_mock_page()
        # First call: state before (unchecked), Second call: state after (checked)
        tool.page._mock_element_locator.evaluate.side_effect = [
            {"found": True, "isCheckbox": True, "checked": False},
            {"found": True, "isCheckbox": True, "checked": True},
        ]

        with patch.object(tool, "_execute_bgym_step", return_value="Success") as mock_step:
            tool.browser_click("a200")

        mock_step.assert_called_once_with('click(bid="a200")')
        # Two evaluate calls: before and after state check
        assert tool.page._mock_element_locator.evaluate.call_count == 2

    def test_browser_click_fallback_unchecks_when_already_checked(self) -> None:
        """Test that browser_click JS fallback unchecks when checkbox was checked."""
        tool = self._create_tool_with_mock_page()
        # State before: checked, State after: still checked (click didn't work)
        # Then _toggle_checkbox_js call, then state_after_js check
        tool.page._mock_element_locator.evaluate.side_effect = [
            {"found": True, "isCheckbox": True, "checked": True},  # state_before
            {"found": True, "isCheckbox": True, "checked": True},  # state_after (same → fallback)
            "toggled_checkbox",  # _toggle_checkbox_js
            {"found": True, "isCheckbox": True, "checked": False},  # state_after_js
        ]

        with patch.object(tool, "_execute_bgym_step", return_value="Success"):
            tool.browser_click("a400")

        # Verify JS toggle was called with False (to uncheck)
        js_toggle_call = tool.page._mock_element_locator.evaluate.call_args_list[2]
        checked_arg = js_toggle_call[0][1]
        assert checked_arg is False
