"""Tests for agentlab2.tools.browser_session module."""

import socket
from unittest.mock import MagicMock, patch

from agentlab2.tools.browser_session import (
    BrowserConfig,
    BrowserSession,
    PlaywrightSession,
    PlaywrightSessionConfig,
    _find_free_port,
)


class TestFindFreePort:
    def test_returns_bindable_port(self) -> None:
        port = _find_free_port()
        with socket.socket() as s:
            s.bind(("", port))  # should not raise

    def test_returns_valid_port_range(self) -> None:
        port = _find_free_port()
        assert 1024 <= port <= 65535

    def test_returns_different_ports_on_successive_calls(self) -> None:
        # Not guaranteed, but almost always true in practice
        ports = {_find_free_port() for _ in range(5)}
        assert len(ports) > 1


class TestBrowserSession:
    def test_cdp_url_defaults_to_none(self) -> None:
        assert BrowserSession.cdp_url is None

    def test_stop_raises_not_implemented(self) -> None:
        import pytest

        with pytest.raises(NotImplementedError):
            BrowserSession().stop()


class TestBrowserConfig:
    def test_make_raises_not_implemented(self) -> None:
        import pytest

        with pytest.raises(NotImplementedError):
            BrowserConfig().make()


class TestPlaywrightSession:
    def test_cdp_url_is_stored(self) -> None:
        session = PlaywrightSession(
            page=MagicMock(), context=MagicMock(), browser=MagicMock(), cdp_url="http://localhost:1234"
        )
        assert session.cdp_url == "http://localhost:1234"

    def test_get_playwright_session_returns_page_and_context(self) -> None:
        mock_page = MagicMock()
        mock_context = MagicMock()
        session = PlaywrightSession(
            page=mock_page, context=mock_context, browser=MagicMock(), cdp_url="http://localhost:1234"
        )
        page, context = session.get_playwright_session()
        assert page is mock_page
        assert context is mock_context

    def test_get_playwright_session_raises_after_stop(self) -> None:
        import pytest

        session = PlaywrightSession(
            page=MagicMock(), context=MagicMock(), browser=MagicMock(), cdp_url="http://localhost:1234"
        )
        session.stop()
        with pytest.raises(RuntimeError, match="stopped or was never started"):
            session.get_playwright_session()

    def test_stop_closes_context_and_browser(self) -> None:
        mock_context = MagicMock()
        mock_browser = MagicMock()
        session = PlaywrightSession(
            page=MagicMock(), context=mock_context, browser=mock_browser, cdp_url="http://localhost:1234"
        )
        session.stop()
        mock_context.close.assert_called_once()
        mock_browser.close.assert_called_once()
        assert session._page is None
        assert session._context is None
        assert session._browser is None


class TestPlaywrightSessionConfig:
    def _make_mock_pw(self) -> MagicMock:
        mock_page = MagicMock()
        mock_context = MagicMock()
        mock_browser = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_browser.new_context.return_value = mock_context
        mock_pw = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        return mock_pw

    def test_make_returns_playwright_session(self) -> None:
        mock_pw = self._make_mock_pw()
        with patch("agentlab2.tools.browser_session._get_global_playwright", return_value=mock_pw):
            session = PlaywrightSessionConfig().make()
        assert isinstance(session, PlaywrightSession)

    def test_make_injects_remote_debugging_port(self) -> None:
        mock_pw = self._make_mock_pw()
        with patch("agentlab2.tools.browser_session._get_global_playwright", return_value=mock_pw):
            PlaywrightSessionConfig().make()
        launch_args = mock_pw.chromium.launch.call_args[1]["args"]
        assert any(arg.startswith("--remote-debugging-port=") for arg in launch_args)

    def test_make_cdp_url_matches_launched_port(self) -> None:
        mock_pw = self._make_mock_pw()
        with patch("agentlab2.tools.browser_session._get_global_playwright", return_value=mock_pw):
            session = PlaywrightSessionConfig().make()
        launch_args = mock_pw.chromium.launch.call_args[1]["args"]
        port_arg = next(a for a in launch_args if a.startswith("--remote-debugging-port="))
        port = port_arg.split("=")[1]
        assert session.cdp_url == f"http://localhost:{port}"

    def test_make_passes_headless_flag(self) -> None:
        mock_pw = self._make_mock_pw()
        with patch("agentlab2.tools.browser_session._get_global_playwright", return_value=mock_pw):
            PlaywrightSessionConfig(headless=False).make()
        assert mock_pw.chromium.launch.call_args[1]["headless"] is False
