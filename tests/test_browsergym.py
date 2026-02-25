"""Tests for agentlab2.tools.browsergym module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from agentlab2.core import Action, Observation
from agentlab2.tools.browsergym import BrowsergymConfig, BrowsergymTool


def _tool_with_mock_page(config: BrowsergymConfig | None = None) -> BrowsergymTool:
    tool = BrowsergymTool(config or BrowsergymConfig())
    page = MagicMock()
    page.keyboard = MagicMock()
    page.mouse = MagicMock()
    tool._page = page
    tool._context = MagicMock()
    return tool


class TestBrowsergymConfig:
    def test_default_config_values(self) -> None:
        config = BrowsergymConfig()
        assert config.headless is True
        assert config.use_html is True
        assert config.use_axtree is True
        assert config.use_screenshot is True
        assert config.prune_html is True
        assert config.max_wait == 60

    def test_make_creates_tool_instance(self) -> None:
        config = BrowsergymConfig()
        tool = config.make()
        assert isinstance(tool, BrowsergymTool)
        assert tool.config is config


class TestBrowsergymToolInitialization:
    def test_page_property_raises_when_not_initialized(self) -> None:
        tool = BrowsergymTool(BrowsergymConfig())
        with pytest.raises(RuntimeError, match="Browser is not initialized"):
            _ = tool.page


class TestBrowsergymToolObservationConversion:
    @patch("agentlab2.tools.browsergym.flatten_dom_to_str")
    def test_html_observation(self, mock_flatten_dom: MagicMock) -> None:
        mock_flatten_dom.return_value = "<html><body>Test</body></html>"
        tool = BrowsergymTool(BrowsergymConfig(use_html=True, use_axtree=False, use_screenshot=False, prune_html=False))
        obs = tool._bgym_obs_to_agentlab_obs({"dom_object": {"documents": [], "strings": []}})
        assert isinstance(obs, Observation)
        assert len(obs.contents) == 1
        assert obs.contents[0].name == "pruned_html"

    def test_screenshot_observation_from_numpy(self) -> None:
        tool = BrowsergymTool(BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=True))
        screenshot_array = np.zeros((100, 100, 3), dtype=np.uint8)
        obs = tool._bgym_obs_to_agentlab_obs({"screenshot": screenshot_array})
        assert len(obs.contents) == 1
        assert isinstance(obs.contents[0].data, Image.Image)


class TestBrowsergymToolActionMethods:
    def test_browser_press_key_uses_playwright_keyboard(self) -> None:
        tool = _tool_with_mock_page()
        tool.browser_press_key("Enter")
        tool.page.keyboard.press.assert_called_once_with("Enter")

    def test_browser_type_uses_bid_locator(self) -> None:
        tool = _tool_with_mock_page()
        frame = MagicMock()
        locator = MagicMock()
        frame.get_by_test_id.return_value = locator
        with patch.object(tool, "_get_frame_for_bid", return_value=frame):
            tool.browser_type("a12", "hello")
        frame.get_by_test_id.assert_called_once_with("a12")
        locator.fill.assert_called_once_with("hello")

    def test_browser_click_uses_js_fallback_when_state_does_not_change(self) -> None:
        tool = _tool_with_mock_page()
        frame = MagicMock()
        locator = MagicMock()
        frame.get_by_test_id.return_value = locator
        with patch.object(tool, "_get_frame_for_bid", return_value=frame), patch.object(
            tool, "_get_checkbox_state", side_effect=[True, True]
        ), patch.object(tool, "_toggle_checkbox_js") as mock_toggle:
            tool.browser_click("a77")
        locator.click.assert_called_once_with(timeout=3000)
        mock_toggle.assert_called_once_with("a77", False)

    @patch("agentlab2.tools.browsergym.time.sleep")
    def test_browser_wait_clamps_to_max_wait(self, mock_sleep: MagicMock) -> None:
        tool = _tool_with_mock_page(BrowsergymConfig(max_wait=3))
        tool.browser_wait(12)
        mock_sleep.assert_called_once_with(3)


class TestBrowsergymToolExecuteActionAndPageObs:
    def test_execute_action_returns_action_and_page_observation(self) -> None:
        tool = _tool_with_mock_page(BrowsergymConfig(use_html=True, use_axtree=False, use_screenshot=False, prune_html=False))
        with patch.object(tool, "_extract_bgym_obs", return_value={"dom_object": {}}), patch(
            "agentlab2.tools.browsergym.flatten_dom_to_str", return_value="<html/>"
        ):
            action = Action(name="noop", arguments={})
            obs = tool.execute_action(action)
        assert isinstance(obs, Observation)
        assert len(obs.contents) >= 2

    def test_page_obs_updates_last_obs(self) -> None:
        tool = _tool_with_mock_page(BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=False))
        with patch.object(tool, "_extract_bgym_obs", return_value={"dom_object": {}}):
            _ = tool.page_obs()
        assert tool._last_obs == {"dom_object": {}}


class TestBrowsergymToolLifecycle:
    @patch("agentlab2.tools.browsergym._get_global_playwright")
    @patch("agentlab2.tools.browsergym.BrowsergymTool._extract_bgym_obs")
    def test_reset_creates_runtime_and_initial_observation(
        self, mock_extract_obs: MagicMock, mock_get_global_playwright: MagicMock
    ) -> None:
        mock_extract_obs.return_value = {"dom_object": {}}

        mock_page = MagicMock()
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_pw = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_get_global_playwright.return_value = mock_pw

        tool = BrowsergymTool(BrowsergymConfig())
        tool.reset()

        assert tool._page is mock_page
        assert tool._last_obs == {"dom_object": {}}

    def test_close_cleans_up_runtime_state(self) -> None:
        tool = _tool_with_mock_page()
        tool._browser = MagicMock()
        tool._last_obs = {"some": "data"}
        tool._last_info = {"info": "data"}
        tool.close()
        assert tool._page is None
        assert tool._context is None
        assert tool._browser is None
        assert tool._last_obs is None
        assert tool._last_info is None
