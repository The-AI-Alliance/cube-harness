"""Tests for cube_harness.tools.browsergym module."""

import tempfile
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.action.utils import get_elem_by_bid
from browsergym.core.constants import BROWSERGYM_ID_ATTRIBUTE
from cube.core import Action, Observation
from cube_browser_playwright.playwright_session import PlaywrightSession, PlaywrightSessionConfig
from PIL import Image
from playwright.sync_api import sync_playwright

from cube_harness.tools.browsergym import (
    BrowsergymConfig,
    BrowsergymTool,
    _action_to_bgym_string,
    _build_action_schemas,
)


@pytest.fixture
def mock_playwright_session() -> Generator[PlaywrightSession, None, None]:
    session = PlaywrightSession(
        playwright=MagicMock(),
        page=MagicMock(),
        context=MagicMock(),
        cdp_url="http://localhost:9222",
        user_data_dir=tempfile.mkdtemp(prefix="cube_harness_"),
    )
    yield session
    session.stop()


class TestBrowsergymConfig:
    """Tests for BrowsergymConfig."""

    def test_default_config_values(self) -> None:
        config = BrowsergymConfig()

        assert config.browser.headless is True
        assert config.use_html is True
        assert config.use_axtree is True
        assert config.use_screenshot is True
        assert config.prune_html is True
        assert config.action_subsets == ["bid", "nav", "tab"]

    def test_custom_config_values(self) -> None:
        config = BrowsergymConfig(
            browser=PlaywrightSessionConfig(headless=False, viewport={"width": 1920, "height": 1080}),
            use_html=False,
            use_axtree=False,
            use_screenshot=False,
            action_subsets=["workarena"],
        )

        assert config.browser.headless is False
        assert config.use_html is False
        assert config.use_axtree is False
        assert config.use_screenshot is False
        assert config.action_subsets == ["workarena"]
        assert config.browser.viewport.width == 1920
        assert config.browser.viewport.height == 1080

    def test_make_creates_tool_instance(self) -> None:
        config = BrowsergymConfig()
        tool = config.make()

        assert isinstance(tool, BrowsergymTool)
        assert tool.config is config

    def test_make_passes_config_to_tool(self) -> None:
        config = BrowsergymConfig(browser=PlaywrightSessionConfig(headless=False))
        tool = config.make()

        assert tool.config.browser.headless is False


class TestBrowsergymToolInitialization:
    """Tests for BrowsergymTool initialization and lifecycle."""

    def test_tool_init_sets_config(self) -> None:
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        assert tool.config is config
        assert tool._session is None
        assert tool._last_obs is None
        assert tool._last_info is None

    def test_page_property_raises_when_not_initialized(self) -> None:
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        with pytest.raises(RuntimeError, match="Browser is not initialized"):
            _ = tool.page

    def test_page_obs_raises_when_not_initialized(self) -> None:
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        with pytest.raises(RuntimeError, match="Browser is not initialized"):
            tool.page_obs()

    def test_evaluate_js_raises_when_not_initialized(self) -> None:
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        with pytest.raises(RuntimeError, match="Browser is not initialized"):
            tool.evaluate_js("return 1+1")


class TestBrowsergymToolObservationConversion:
    """Tests for _bgym_obs_to_cube_obs conversion."""

    def test_empty_observation(self) -> None:
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=False)
        tool = BrowsergymTool(config)

        bgym_obs: dict = {}
        obs = tool._bgym_obs_to_cube_obs(bgym_obs)

        assert isinstance(obs, Observation)
        assert len(obs.contents) == 0

    @patch("cube_harness.tools.browsergym.flatten_dom_to_str")
    def test_html_observation(self, mock_flatten_dom: MagicMock) -> None:
        mock_flatten_dom.return_value = "<html><body>Test</body></html>"

        config = BrowsergymConfig(use_html=True, use_axtree=False, use_screenshot=False, prune_html=False)
        tool = BrowsergymTool(config)

        dom_obj = {"documents": [], "strings": []}
        bgym_obs = {"dom_object": dom_obj}
        obs = tool._bgym_obs_to_cube_obs(bgym_obs)

        assert len(obs.contents) == 1
        assert obs.contents[0].name == "pruned_html"
        assert obs.contents[0].data == "<html><body>Test</body></html>"

    @patch("cube_harness.tools.browsergym.prune_html")
    @patch("cube_harness.tools.browsergym.flatten_dom_to_str")
    def test_html_observation_with_pruning(self, mock_flatten_dom: MagicMock, mock_prune_html: MagicMock) -> None:
        mock_flatten_dom.return_value = "<html><body>Full HTML</body></html>"
        mock_prune_html.return_value = "<body>Pruned</body>"

        config = BrowsergymConfig(use_html=True, use_axtree=False, use_screenshot=False, prune_html=True)
        tool = BrowsergymTool(config)

        bgym_obs = {"dom_object": {"documents": [], "strings": []}}
        obs = tool._bgym_obs_to_cube_obs(bgym_obs)

        mock_prune_html.assert_called_once_with("<html><body>Full HTML</body></html>")
        assert obs.contents[0].data == "<body>Pruned</body>"

    @patch("cube_harness.tools.browsergym.flatten_axtree_to_str")
    def test_axtree_observation(self, mock_flatten_axtree: MagicMock) -> None:
        mock_flatten_axtree.return_value = "[a1] button 'Submit'"

        config = BrowsergymConfig(use_html=False, use_axtree=True, use_screenshot=False)
        tool = BrowsergymTool(config)

        axtree_obj = {"nodes": []}
        bgym_obs = {"axtree_object": axtree_obj}
        obs = tool._bgym_obs_to_cube_obs(bgym_obs)

        assert len(obs.contents) == 1
        assert obs.contents[0].name == "axtree_txt"
        assert obs.contents[0].data == "[a1] button 'Submit'"

    def test_axtree_observation_empty_object(self) -> None:
        config = BrowsergymConfig(use_html=False, use_axtree=True, use_screenshot=False)
        tool = BrowsergymTool(config)

        bgym_obs = {"axtree_object": None}
        obs = tool._bgym_obs_to_cube_obs(bgym_obs)

        assert len(obs.contents) == 0

    def test_screenshot_observation_from_numpy(self) -> None:
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=True)
        tool = BrowsergymTool(config)

        screenshot_array = np.zeros((100, 100, 3), dtype=np.uint8)
        screenshot_array[:, :, 0] = 255

        bgym_obs = {"screenshot": screenshot_array}
        obs = tool._bgym_obs_to_cube_obs(bgym_obs)

        assert len(obs.contents) == 1
        assert obs.contents[0].name == "screenshot"
        assert isinstance(obs.contents[0].data, Image.Image)
        assert obs.contents[0].data.size == (100, 100)

    def test_screenshot_observation_from_pil_image(self) -> None:
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=True)
        tool = BrowsergymTool(config)

        screenshot_img = Image.new("RGB", (200, 150), color="blue")
        bgym_obs = {"screenshot": screenshot_img}
        obs = tool._bgym_obs_to_cube_obs(bgym_obs)

        assert len(obs.contents) == 1
        assert obs.contents[0].name == "screenshot"
        assert obs.contents[0].data is screenshot_img

    @patch("cube_harness.tools.browsergym.flatten_axtree_to_str")
    @patch("cube_harness.tools.browsergym.flatten_dom_to_str")
    def test_full_observation(self, mock_flatten_dom: MagicMock, mock_flatten_axtree: MagicMock) -> None:
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
        obs = tool._bgym_obs_to_cube_obs(bgym_obs)

        assert len(obs.contents) == 3
        content_names = {c.name for c in obs.contents}
        assert content_names == {"pruned_html", "axtree_txt", "screenshot"}

    def test_focused_element_observation(self) -> None:
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=False)
        tool = BrowsergymTool(config)

        bgym_obs = {"focused_element_bid": "a123"}
        obs = tool._bgym_obs_to_cube_obs(bgym_obs)

        assert len(obs.contents) == 1
        assert obs.contents[0].name == "focused_element"
        assert obs.contents[0].data == "a123"

    def test_focused_element_observation_empty(self) -> None:
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=False)
        tool = BrowsergymTool(config)

        bgym_obs = {"focused_element_bid": None}
        obs = tool._bgym_obs_to_cube_obs(bgym_obs)

        assert len(obs.contents) == 0

    def test_last_action_error_observation(self) -> None:
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=False)
        tool = BrowsergymTool(config)

        bgym_obs = {"last_action_error": "Element not found: bid='xyz'"}
        obs = tool._bgym_obs_to_cube_obs(bgym_obs)

        assert len(obs.contents) == 1
        assert obs.contents[0].name == "last_action_error"
        assert obs.contents[0].data == "Element not found: bid='xyz'"

    def test_last_action_error_observation_empty(self) -> None:
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=False)
        tool = BrowsergymTool(config)

        bgym_obs = {"last_action_error": None}
        obs = tool._bgym_obs_to_cube_obs(bgym_obs)

        assert len(obs.contents) == 0

    def test_last_action_error_observation_empty_string(self) -> None:
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=False)
        tool = BrowsergymTool(config)

        bgym_obs = {"last_action_error": ""}
        obs = tool._bgym_obs_to_cube_obs(bgym_obs)

        assert len(obs.contents) == 0

    def test_user_message_observation(self) -> None:
        config = BrowsergymConfig(use_html=False, use_axtree=False, use_screenshot=False)
        tool = BrowsergymTool(config)
        tool._last_info = {"user_messages": ["Task completed successfully"]}

        bgym_obs: dict = {}
        obs = tool._bgym_obs_to_cube_obs(bgym_obs)

        assert len(obs.contents) == 1
        assert obs.contents[0].name == "user_message"
        assert obs.contents[0].data == "Task completed successfully"


class TestActionToBgymString:
    """Tests for _action_to_bgym_string serialisation."""

    def test_click_action(self) -> None:
        action = Action(name="click", arguments={"bid": "a51"})
        assert _action_to_bgym_string(action) == "click(bid='a51')"

    def test_fill_action(self) -> None:
        action = Action(name="fill", arguments={"bid": "b12", "value": "Hello World"})
        assert _action_to_bgym_string(action) == "fill(bid='b12', value='Hello World')"

    def test_fill_with_quotes(self) -> None:
        action = Action(name="fill", arguments={"bid": "c1", "value": 'Say "Hello"'})
        result = _action_to_bgym_string(action)
        assert "fill(" in result
        assert "bid='c1'" in result

    def test_keyboard_press(self) -> None:
        action = Action(name="keyboard_press", arguments={"key": "Enter"})
        assert _action_to_bgym_string(action) == "keyboard_press(key='Enter')"

    def test_drag_and_drop(self) -> None:
        action = Action(name="drag_and_drop", arguments={"from_bid": "e1", "to_bid": "f2"})
        assert _action_to_bgym_string(action) == "drag_and_drop(from_bid='e1', to_bid='f2')"

    def test_hover(self) -> None:
        action = Action(name="hover", arguments={"bid": "g3"})
        assert _action_to_bgym_string(action) == "hover(bid='g3')"

    def test_select_option(self) -> None:
        action = Action(name="select_option", arguments={"bid": "h4", "options": "option1"})
        assert _action_to_bgym_string(action) == "select_option(bid='h4', options='option1')"

    def test_mouse_click(self) -> None:
        action = Action(name="mouse_click", arguments={"x": 100, "y": 200})
        assert _action_to_bgym_string(action) == "mouse_click(x=100, y=200)"

    def test_noop(self) -> None:
        action = Action(name="noop", arguments={})
        assert _action_to_bgym_string(action) == "noop()"

    def test_noop_with_wait(self) -> None:
        action = Action(name="noop", arguments={"wait_ms": 5000})
        assert _action_to_bgym_string(action) == "noop(wait_ms=5000)"

    def test_goto(self) -> None:
        action = Action(name="goto", arguments={"url": "http://example.com"})
        assert _action_to_bgym_string(action) == "goto(url='http://example.com')"

    def test_go_back(self) -> None:
        action = Action(name="go_back", arguments={})
        assert _action_to_bgym_string(action) == "go_back()"

    def test_go_forward(self) -> None:
        action = Action(name="go_forward", arguments={})
        assert _action_to_bgym_string(action) == "go_forward()"

    def test_scroll(self) -> None:
        action = Action(name="scroll", arguments={"delta_x": 0, "delta_y": 200})
        assert _action_to_bgym_string(action) == "scroll(delta_x=0, delta_y=200)"

    def test_send_msg_to_user(self) -> None:
        action = Action(name="send_msg_to_user", arguments={"text": "The answer is 42"})
        assert _action_to_bgym_string(action) == "send_msg_to_user(text='The answer is 42')"


class TestBuildActionSchemas:
    """Tests for _build_action_schemas."""

    def test_default_subsets_include_expected_actions(self) -> None:
        action_set = HighLevelActionSet(subsets=["chat", "infeas", "bid", "nav", "tab"], multiaction=False)
        schemas = _build_action_schemas(action_set)
        names = {s.name for s in schemas}

        # Core bgym actions should be present
        assert "click" in names
        assert "fill" in names
        assert "hover" in names
        assert "scroll" in names
        assert "goto" in names
        assert "go_back" in names
        assert "noop" in names
        assert "send_msg_to_user" in names
        assert "report_infeasible" in names
        assert "new_tab" in names

    def test_workarena_subset(self) -> None:
        action_set = HighLevelActionSet(subsets=["workarena"], multiaction=False)
        schemas = _build_action_schemas(action_set)
        names = {s.name for s in schemas}

        assert "click" in names
        assert "fill" in names
        assert "noop" in names

    def test_schemas_have_parameters(self) -> None:
        action_set = HighLevelActionSet(subsets=["bid"], multiaction=False)
        schemas = _build_action_schemas(action_set)

        click_schema = next(s for s in schemas if s.name == "click")
        assert "properties" in click_schema.parameters
        assert "bid" in click_schema.parameters["properties"]


class TestBrowsergymToolStepResults:
    """Tests for action step results and state updates."""

    def _create_tool_with_mock_page(self, mock_playwright_session: PlaywrightSession) -> BrowsergymTool:
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)
        tool._session = mock_playwright_session
        tool._action_set = MagicMock()
        return tool

    def test_execute_bgym_step_returns_success(self, mock_playwright_session: PlaywrightSession) -> None:
        tool = self._create_tool_with_mock_page(mock_playwright_session)

        with (
            patch("cube_harness.tools.browsergym.execute_python_code"),
            patch.object(tool, "_extract_bgym_obs", return_value={}),
        ):
            result = tool._execute_bgym_step("noop()")

        assert result == "Success"

    def test_execute_bgym_step_returns_failure_on_exception(self, mock_playwright_session: PlaywrightSession) -> None:
        tool = self._create_tool_with_mock_page(mock_playwright_session)

        with (
            patch("cube_harness.tools.browsergym.execute_python_code", side_effect=Exception("Some error")),
            patch.object(tool, "_extract_bgym_obs", return_value={}),
        ):
            result = tool._execute_bgym_step("noop()")

        assert "Failed" in result

    def test_execute_bgym_step_updates_last_obs(self, mock_playwright_session: PlaywrightSession) -> None:
        tool = self._create_tool_with_mock_page(mock_playwright_session)
        new_obs = {"screenshot": np.zeros((10, 10, 3))}

        with (
            patch("cube_harness.tools.browsergym.execute_python_code"),
            patch.object(tool, "_extract_bgym_obs", return_value=new_obs),
        ):
            tool._execute_bgym_step("noop()")

        assert tool._last_obs is new_obs

    def test_execute_bgym_step_updates_last_info(self, mock_playwright_session: PlaywrightSession) -> None:
        tool = self._create_tool_with_mock_page(mock_playwright_session)

        with (
            patch("cube_harness.tools.browsergym.execute_python_code"),
            patch.object(tool, "_extract_bgym_obs", return_value={}),
        ):
            tool._execute_bgym_step("noop()")

        assert tool._last_info is not None
        assert tool._last_info["source"] == "action"


class TestBrowsergymToolExecuteAction:
    """Tests for execute_action integration."""

    def test_execute_action_returns_observation(self, mock_playwright_session: PlaywrightSession) -> None:
        config = BrowsergymConfig(use_html=True, use_axtree=False, use_screenshot=False, prune_html=False)
        tool = BrowsergymTool(config)
        tool._session = mock_playwright_session

        action = Action(name="click", arguments={"bid": "a1"})

        with (
            patch.object(tool, "_execute_bgym_step", return_value="Success"),
            patch.object(tool, "page_obs", return_value=Observation()),
        ):
            obs = tool.execute_action(action)

        assert isinstance(obs, Observation)
        assert len(obs.contents) >= 1


class TestBrowsergymToolActionSet:
    """Tests for action_set property."""

    def test_action_set_contains_bgym_native_actions(self) -> None:
        """Test that default action_set contains pure browser actions only."""
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        action_names = {a.name for a in tool.action_set}

        # Default subsets (bid, nav, tab) should include browser actions
        assert "click" in action_names
        assert "fill" in action_names
        assert "hover" in action_names
        assert "scroll" in action_names
        assert "goto" in action_names
        assert "noop" in action_names

        # Chat/infeas actions should NOT be in default config — they belong to ChatTool
        assert "send_msg_to_user" not in action_names
        assert "report_infeasible" not in action_names

        # Old names should NOT be present
        assert "browser_click" not in action_names
        assert "browser_type" not in action_names
        assert "browser_hover" not in action_names

    def test_action_set_configurable_subsets(self) -> None:
        """Test that action_subsets config controls which actions are exposed."""
        config_full = BrowsergymConfig(action_subsets=["chat", "infeas", "bid", "nav", "tab"])
        tool_full = BrowsergymTool(config_full)

        config_minimal = BrowsergymConfig(action_subsets=["bid"])
        tool_minimal = BrowsergymTool(config_minimal)

        full_names = {a.name for a in tool_full.action_set}
        minimal_names = {a.name for a in tool_minimal.action_set}

        # Full set should have tab/nav actions
        assert "new_tab" in full_names
        assert "goto" in full_names

        # Minimal should not have tab/nav/chat
        assert "new_tab" not in minimal_names
        assert "send_msg_to_user" not in minimal_names


class TestBrowsergymToolLifecycle:
    """Tests for tool lifecycle methods."""

    def test_reset_initializes_browser(self) -> None:
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

    def test_reset_closes_existing_runtime(self, mock_playwright_session: PlaywrightSession) -> None:
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)
        tool._session = mock_playwright_session

        with (
            patch.object(tool, "_close_runtime") as mock_close,
            patch.object(tool, "_create_runtime"),
            patch.object(tool, "_wait_dom_loaded"),
            patch.object(tool, "_extract_bgym_obs", return_value={}),
        ):
            tool.reset()

        mock_close.assert_called_once()

    def test_close_cleans_up_browser(self, mock_playwright_session: PlaywrightSession) -> None:
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        tool._session = mock_playwright_session
        tool._last_obs = {"some": "data"}
        tool._last_info = {"info": "data"}

        tool.close()

        assert tool._session is None
        assert tool._last_obs is None
        assert tool._last_info is None

    def test_close_handles_exception(self, mock_playwright_session: PlaywrightSession) -> None:
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        mock_context = MagicMock()
        mock_context.close.side_effect = Exception("Close failed")
        tool._session = PlaywrightSession(
            playwright=MagicMock(),
            page=MagicMock(),
            context=mock_context,
            cdp_url="http://localhost:9222",
            user_data_dir=tempfile.mkdtemp(prefix="cube_harness_"),
        )
        tool._last_obs = {"some": "data"}

        tool.close()

        assert tool._session is None
        assert tool._last_obs is None

    def test_close_noop_when_no_browser(self) -> None:
        config = BrowsergymConfig()
        tool = BrowsergymTool(config)

        tool.close()


# HTML page with a keydown-based autocomplete simulation.
# fill() sets the value directly without firing keydown/keypress — suggestions stay empty.
# press_sequentially() fires a full keyboard event sequence per character — suggestions appear.
# bgym BIDs for top-level elements are numeric strings ("1", "2", ...).
# Leading lowercase letters are parsed as iframe-descent prefixes by get_elem_by_bid.
_AUTOCOMPLETE_BID = "1"
_AUTOCOMPLETE_PAGE = """<!DOCTYPE html>
<html><body>
  <input {bid_attr}="{bid}" id="search">
  <ul id="suggestions"></ul>
  <script>
    window.keydownCount = 0;
    document.getElementById('search').addEventListener('keydown', function(e) {{
      if (e.key.length === 1) {{
        window.keydownCount++;
        const li = document.createElement('li');
        li.textContent = e.key;
        document.getElementById('suggestions').appendChild(li);
      }}
    }});
  </script>
</body></html>""".format(bid_attr=BROWSERGYM_ID_ATTRIBUTE, bid=_AUTOCOMPLETE_BID)


@pytest.fixture(scope="module")
def live_page():
    """Real headless Chromium page configured with bgym's test-id attribute."""
    with sync_playwright() as p:
        p.selectors.set_test_id_attribute(BROWSERGYM_ID_ATTRIBUTE)
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        yield page
        browser.close()


@pytest.mark.integration
class TestKeyboardTypeIntoVsFill:
    """Integration tests verifying keyboard_type_into fires keydown events that fill() skips."""

    def test_fill_does_not_fire_keydown_events(self, live_page) -> None:
        live_page.set_content(_AUTOCOMPLETE_PAGE)

        get_elem_by_bid(live_page, _AUTOCOMPLETE_BID).fill("hello")

        keydown_count = live_page.evaluate("window.keydownCount")
        assert keydown_count == 0, "fill() must not fire keydown events"
        suggestions = live_page.locator("#suggestions li").count()
        assert suggestions == 0

    def test_press_sequentially_fires_keydown_per_character(self, live_page) -> None:
        live_page.set_content(_AUTOCOMPLETE_PAGE)

        get_elem_by_bid(live_page, _AUTOCOMPLETE_BID).press_sequentially("hello", delay=0)

        keydown_count = live_page.evaluate("window.keydownCount")
        assert keydown_count == 5, "press_sequentially must fire one keydown per character"
        suggestions = live_page.locator("#suggestions li").count()
        assert suggestions == 5

    def test_keyboard_type_into_action_fires_keydown_events(self, live_page) -> None:
        """End-to-end: BrowsergymTool.keyboard_type_into reaches the element via get_elem_by_bid."""
        live_page.set_content(_AUTOCOMPLETE_PAGE)

        config = BrowsergymConfig()
        tool = BrowsergymTool(config)
        tool._session = MagicMock()
        tool._session.page = live_page

        with patch.object(tool, "_extract_bgym_obs", return_value={}):
            result = tool.keyboard_type_into(_AUTOCOMPLETE_BID, "abc")

        assert result == "Success"
        keydown_count = live_page.evaluate("window.keydownCount")
        assert keydown_count == 3
        suggestions = live_page.locator("#suggestions li").count()
        assert suggestions == 3
