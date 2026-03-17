"""Tests for cube_harness.tools.browser_session module."""

from unittest.mock import MagicMock, patch

import pytest

from cube_harness.tools.browser_session import (
    BrowserConfig,
    BrowserSession,
    PlaywrightSession,
    PlaywrightSessionConfig,
)


class TestBrowserSession:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BrowserSession()  # type: ignore[abstract]


class TestBrowserConfig:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BrowserConfig()  # type: ignore[abstract]


class TestPlaywrightSession:
    def test_cdp_url_is_stored(self) -> None:
        session = PlaywrightSession(page=MagicMock(), context=MagicMock(), cdp_url="http://localhost:1234")
        assert session.cdp_url == "http://localhost:1234"

    def test_get_playwright_session_returns_page_and_context(self) -> None:
        mock_page = MagicMock()
        mock_context = MagicMock()
        session = PlaywrightSession(page=mock_page, context=mock_context, cdp_url="http://localhost:1234")
        page, context = session.get_playwright_session()
        assert page is mock_page
        assert context is mock_context

    def test_stop_closes_context(self) -> None:
        mock_context = MagicMock()
        session = PlaywrightSession(page=MagicMock(), context=mock_context, cdp_url="http://localhost:1234")
        session.stop()
        mock_context.close.assert_called_once()

    def test_stop_logs_warning_on_context_close_error(self) -> None:
        mock_context = MagicMock()
        mock_context.close.side_effect = RuntimeError("context close error")
        session = PlaywrightSession(page=MagicMock(), context=mock_context, cdp_url="http://localhost:1234")
        session.stop()  # should not raise


class TestPlaywrightSessionConfig:
    def _make_mock_pw(self) -> MagicMock:
        mock_page = MagicMock()
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_pw = MagicMock()
        mock_pw.chromium.launch_persistent_context.return_value = mock_context
        return mock_pw

    def test_make_returns_playwright_session(self) -> None:
        mock_pw = self._make_mock_pw()
        with (
            patch("cube_harness.tools.browser_session._get_global_playwright", return_value=mock_pw),
            patch("cube_harness.tools.browser_session._read_cdp_url", return_value="http://localhost:1234"),
        ):
            session = PlaywrightSessionConfig().make()
        assert isinstance(session, PlaywrightSession)

    def test_make_injects_remote_debugging_port_0(self) -> None:
        mock_pw = self._make_mock_pw()
        with (
            patch("cube_harness.tools.browser_session._get_global_playwright", return_value=mock_pw),
            patch("cube_harness.tools.browser_session._read_cdp_url", return_value="http://localhost:1234"),
        ):
            PlaywrightSessionConfig().make()
        launch_args = mock_pw.chromium.launch_persistent_context.call_args[1]["args"]
        assert "--remote-debugging-port=0" in launch_args

    def test_make_cdp_url_comes_from_read_cdp_url(self) -> None:
        mock_pw = self._make_mock_pw()
        with (
            patch("cube_harness.tools.browser_session._get_global_playwright", return_value=mock_pw),
            patch(
                "cube_harness.tools.browser_session._read_cdp_url", return_value="http://localhost:9876"
            ) as mock_read,
        ):
            session = PlaywrightSessionConfig().make()
        mock_read.assert_called_once()
        assert session.cdp_url == "http://localhost:9876"

    def test_make_passes_headless_flag(self) -> None:
        mock_pw = self._make_mock_pw()
        with (
            patch("cube_harness.tools.browser_session._get_global_playwright", return_value=mock_pw),
            patch("cube_harness.tools.browser_session._read_cdp_url", return_value="http://localhost:1234"),
        ):
            PlaywrightSessionConfig(headless=False).make()
        assert mock_pw.chromium.launch_persistent_context.call_args[1]["headless"] is False
