import asyncio
import json

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from mullet.core.mcp.oauth import (
    MulletTokenStorage,
    OAuthCallbackHandler,
    _find_free_port,
    _can_open_browser,
    _redirect_to_browser,
    _wait_for_callback,
    run_oauth_flow,
    _sanitize_server_name,
    build_oauth_auth,
)


@pytest.fixture
def mock_home(tmp_path):
    with patch("pathlib.Path.home", return_value=tmp_path):
        yield tmp_path


@pytest.fixture
def storage(mock_home):
    return MulletTokenStorage("test-server")


def test_sanitize_normal():
    assert _sanitize_server_name("my-server") == "my-server"


def test_sanitize_special_chars():
    result = _sanitize_server_name("My Server!@#")
    assert " " not in result
    assert "!" not in result
    assert "@" not in result


def test_sanitize_long():
    assert len(_sanitize_server_name("a" * 100)) <= 60


def test_storage_get_empty(storage):
    assert storage.get_tokens() is None
    assert storage.get_client_info() is None


def test_storage_roundtrip(storage):
    tokens = {"access_token": "abc"}
    storage.set_tokens(tokens)
    assert storage.get_tokens() == tokens
    info = {"client_id": "123"}
    storage.set_client_info(info)
    assert storage.get_client_info() == info


def test_storage_clear(storage):
    storage.set_tokens({"a": "b"})
    storage.set_client_info({"c": "d"})
    storage.clear()
    assert storage.get_tokens() is None
    assert storage.get_client_info() is None


def test_can_open_browser_ssh(monkeypatch):
    monkeypatch.setenv("SSH_CLIENT", "192.168.1.1")
    monkeypatch.delenv("SSH_TTY", raising=False)
    assert _can_open_browser() is False


def test_build_oauth_auth(mock_home):
    s = MulletTokenStorage("srv")
    s.set_tokens({"token": "t1"})
    result = build_oauth_auth("srv", "https://example.com")
    assert result == {"token": "t1"}


class TestOAuthMissingLines:
    # -- MulletTokenStorage.clear() OSError (lines 73-74) --

    def test_clear_unlink_oserror_logged(self, storage, caplog):
        with patch("pathlib.Path.unlink", side_effect=OSError("perm denied")):
            storage.clear()
        assert "failed to unlink" in caplog.text

    def test_clear_unlink_oserror_no_raise(self, storage):
        with patch("pathlib.Path.unlink", side_effect=OSError("boom")):
            storage.clear()  # must not raise

    # -- _read_json invalid JSON (lines 82-84) --

    def test_read_json_invalid_returns_none(self, tmp_path, caplog):
        p = tmp_path / "bad.json"
        p.write_text("not json {", encoding="utf-8")
        result = MulletTokenStorage._read_json(p)
        assert result is None
        assert "failed to read JSON" in caplog.text

    def test_read_json_decode_error_returns_none(self, tmp_path, caplog):
        p = tmp_path / "bin.json"
        p.write_bytes(b"\xff\xfe")
        result = MulletTokenStorage._read_json(p)
        assert result is None
        assert "failed to read JSON" in caplog.text

    # -- _write_json chmod OSError (lines 91-92) --

    def test_write_json_chmod_oserror_ignored(self, tmp_path):
        p = tmp_path / "out.json"
        with patch("pathlib.Path.chmod", side_effect=OSError("perm")):
            MulletTokenStorage._write_json(p, {"key": "val"})
        assert json.loads(p.read_text(encoding="utf-8")) == {"key": "val"}

    def test_write_json_success(self, tmp_path):
        p = tmp_path / "good.json"
        MulletTokenStorage._write_json(p, {"a": 1})
        assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}

    # -- OAuthCallbackHandler.do_GET (lines 103-109) --

    def test_handler_do_get_parses_code_and_state(self):
        handler = OAuthCallbackHandler.__new__(OAuthCallbackHandler)
        handler.path = "/?code=abc123&state=xyz"
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        OAuthCallbackHandler._result = {"auth_code": None, "state": None}
        handler.do_GET()
        assert OAuthCallbackHandler._result["auth_code"] == "abc123"
        assert OAuthCallbackHandler._result["state"] == "xyz"
        handler.send_response.assert_called_once_with(200)
        handler.send_header.assert_called_once_with("Content-Type", "text/html")
        handler.end_headers.assert_called_once()

    def test_handler_do_get_sends_html(self):
        handler = OAuthCallbackHandler.__new__(OAuthCallbackHandler)
        handler.path = "/?code=c&state=s"
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.do_GET()
        handler.wfile.write.assert_called_once_with(
            b"<html><body><h3>Authorization complete. You can close this tab.</h3></body></html>"
        )

    def test_handler_do_get_missing_params(self):
        handler = OAuthCallbackHandler.__new__(OAuthCallbackHandler)
        handler.path = "/"
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        OAuthCallbackHandler._result = {"auth_code": "prev", "state": "prev"}
        handler.do_GET()
        assert OAuthCallbackHandler._result["auth_code"] is None
        assert OAuthCallbackHandler._result["state"] is None

    def test_handler_do_get_only_code(self):
        handler = OAuthCallbackHandler.__new__(OAuthCallbackHandler)
        handler.path = "/?code=onlycode"
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        OAuthCallbackHandler._result = {"auth_code": None, "state": None}
        handler.do_GET()
        assert OAuthCallbackHandler._result["auth_code"] == "onlycode"
        assert OAuthCallbackHandler._result["state"] is None

    def test_handler_do_get_only_state(self):
        handler = OAuthCallbackHandler.__new__(OAuthCallbackHandler)
        handler.path = "/?state=onlystate"
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        OAuthCallbackHandler._result = {"auth_code": None, "state": None}
        handler.do_GET()
        assert OAuthCallbackHandler._result["auth_code"] is None
        assert OAuthCallbackHandler._result["state"] == "onlystate"

    # -- OAuthCallbackHandler.log_message (line 113) --

    def test_handler_log_message_noop(self):
        handler = OAuthCallbackHandler.__new__(OAuthCallbackHandler)
        handler.log_message("fmt %s", "arg")  # must not raise

    # -- _find_free_port (lines 116-118) --

    def test_find_free_port_returns_int(self):
        port = _find_free_port()
        assert isinstance(port, int)

    def test_find_free_port_in_valid_range(self):
        port = _find_free_port()
        assert 1024 <= port <= 65535

    # -- _can_open_browser (lines 123-125) --

    def test_can_open_browser_ssh_client(self, monkeypatch):
        monkeypatch.setenv("SSH_CLIENT", "1")
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        with patch("mullet.core.mcp.oauth.os.name", "posix"), patch(
            "mullet.core.mcp.oauth.os.uname"
        ) as mock_uname:
            mock_uname.return_value.sysname = "Linux"
            assert _can_open_browser() is False

    def test_can_open_browser_ssh_tty(self, monkeypatch):
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.setenv("SSH_TTY", "tty")
        monkeypatch.setenv("DISPLAY", ":0")
        with patch("mullet.core.mcp.oauth.os.name", "posix"), patch(
            "mullet.core.mcp.oauth.os.uname"
        ) as mock_uname:
            mock_uname.return_value.sysname = "Linux"
            assert _can_open_browser() is False

    def test_can_open_browser_no_display_linux(self, monkeypatch):
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        with patch("mullet.core.mcp.oauth.os.name", "posix"), patch(
            "mullet.core.mcp.oauth.os.uname"
        ) as mock_uname:
            mock_uname.return_value.sysname = "Linux"
            assert _can_open_browser() is False

    def test_can_open_browser_no_display_darwin(self, monkeypatch):
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        with patch("mullet.core.mcp.oauth.os.name", "posix"), patch(
            "mullet.core.mcp.oauth.os.uname"
        ) as mock_uname:
            mock_uname.return_value.sysname = "Darwin"
            assert _can_open_browser() is True

    def test_can_open_browser_windows_no_display(self, monkeypatch):
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        with patch("mullet.core.mcp.oauth.os.name", "nt"):
            assert _can_open_browser() is True

    def test_can_open_browser_linux_with_display(self, monkeypatch):
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        with patch("mullet.core.mcp.oauth.os.name", "posix"), patch(
            "mullet.core.mcp.oauth.os.uname"
        ) as mock_uname:
            mock_uname.return_value.sysname = "Linux"
            assert _can_open_browser() is True

    # -- _redirect_to_browser async (lines 129-136) --

    def test_redirect_opens_browser(self, capsys):
        with patch("mullet.core.mcp.oauth._can_open_browser", return_value=True), patch(
            "webbrowser.open"
        ) as mock_open:
            asyncio.run(_redirect_to_browser("http://auth/url"))
            mock_open.assert_called_once_with("http://auth/url")
            captured = capsys.readouterr()
            assert "Opened browser" in captured.out

    def test_redirect_prints_when_cannot_open(self, capsys):
        with patch("mullet.core.mcp.oauth._can_open_browser", return_value=False):
            asyncio.run(_redirect_to_browser("http://auth/url"))
            captured = capsys.readouterr()
            assert "Open this URL" in captured.out
            assert "http://auth/url" in captured.out

    def test_redirect_prints_on_exception(self, capsys):
        with patch("mullet.core.mcp.oauth._can_open_browser", return_value=True), patch(
            "webbrowser.open", side_effect=Exception("boom")
        ):
            asyncio.run(_redirect_to_browser("http://auth/url"))
            captured = capsys.readouterr()
            assert "Open this URL" in captured.out
            assert "http://auth/url" in captured.out

    # -- _wait_for_callback async (lines 141-163) --

    def test_wait_for_callback_fast(self, monkeypatch):
        monkeypatch.setattr("mullet.core.mcp.oauth._oauth_port", None)
        with patch("mullet.core.mcp.oauth.HTTPServer") as MockServer, patch(
            "mullet.core.mcp.oauth.threading.Thread"
        ) as MockThread:
            MockThread.return_value.start = MagicMock()
            OAuthCallbackHandler._result = {"auth_code": "code123", "state": "st"}
            code, state = asyncio.run(_wait_for_callback())
            assert code == "code123"
            assert state == "st"

    def test_wait_for_callback_timeout_prompts(self, monkeypatch):
        monkeypatch.setattr("mullet.core.mcp.oauth._oauth_port", None)
        with patch("mullet.core.mcp.oauth.HTTPServer") as MockServer, patch(
            "mullet.core.mcp.oauth.threading.Thread"
        ) as MockThread, patch("asyncio.sleep", new_callable=AsyncMock):
            MockThread.return_value.start = MagicMock()
            OAuthCallbackHandler._result = {"auth_code": None, "state": None}
            monkeypatch.setattr("builtins.input", lambda _: "manual_code")
            code, state = asyncio.run(_wait_for_callback())
            assert code == "manual_code"
            assert state is None

    # -- run_oauth_flow async (lines 171-220) --

    def test_run_oauth_flow_success(self, monkeypatch):
        monkeypatch.setattr("mullet.core.mcp.oauth._oauth_port", None)
        with patch(
            "mullet.core.mcp.oauth._redirect_to_browser", new_callable=AsyncMock
        ), patch(
            "mullet.core.mcp.oauth._wait_for_callback",
            new_callable=AsyncMock,
            return_value=("code", "state_val"),
        ), patch(
            "mullet.core.mcp.oauth.secrets.token_urlsafe",
            side_effect=["verifier96", "state_val"],
        ), patch(
            "httpx.AsyncClient"
        ) as MockClient:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"access_token": "tok"}
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.post.return_value = mock_resp
            MockClient.return_value = mock_client
            result = asyncio.run(
                run_oauth_flow("test", "http://s", "cid", "http://a", "http://t")
            )
            assert result["access_token"] == "tok"

    def test_run_oauth_flow_no_auth_code(self, monkeypatch):
        monkeypatch.setattr("mullet.core.mcp.oauth._oauth_port", None)
        with patch(
            "mullet.core.mcp.oauth._redirect_to_browser", new_callable=AsyncMock
        ), patch(
            "mullet.core.mcp.oauth._wait_for_callback",
            new_callable=AsyncMock,
            return_value=("", "state_val"),
        ), patch(
            "mullet.core.mcp.oauth.secrets.token_urlsafe",
            side_effect=["verifier96", "state_val"],
        ), patch(
            "httpx.AsyncClient"
        ):
            with pytest.raises(RuntimeError, match="failed to get authorization code"):
                asyncio.run(
                    run_oauth_flow("test", "http://s", "cid", "http://a", "http://t")
                )

    def test_run_oauth_flow_state_mismatch(self, monkeypatch):
        monkeypatch.setattr("mullet.core.mcp.oauth._oauth_port", None)
        with patch(
            "mullet.core.mcp.oauth._redirect_to_browser", new_callable=AsyncMock
        ), patch(
            "mullet.core.mcp.oauth._wait_for_callback",
            new_callable=AsyncMock,
            return_value=("code", "wrong_state"),
        ), patch(
            "mullet.core.mcp.oauth.secrets.token_urlsafe",
            side_effect=["verifier96", "state_val"],
        ), patch(
            "httpx.AsyncClient"
        ):
            with pytest.raises(RuntimeError, match="state parameter mismatch"):
                asyncio.run(
                    run_oauth_flow("test", "http://s", "cid", "http://a", "http://t")
                )

    def test_run_oauth_flow_custom_scopes(self, monkeypatch):
        monkeypatch.setattr("mullet.core.mcp.oauth._oauth_port", None)
        with patch(
            "mullet.core.mcp.oauth._redirect_to_browser", new_callable=AsyncMock
        ), patch(
            "mullet.core.mcp.oauth._wait_for_callback",
            new_callable=AsyncMock,
            return_value=("code", "state_val"),
        ), patch(
            "mullet.core.mcp.oauth.secrets.token_urlsafe",
            side_effect=["verifier96", "state_val"],
        ), patch(
            "httpx.AsyncClient"
        ) as MockClient:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"access_token": "tok2"}
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.post.return_value = mock_resp
            MockClient.return_value = mock_client
            result = asyncio.run(
                run_oauth_flow(
                    "test", "http://s", "cid", "http://a", "http://t", scopes=["custom"]
                )
            )
            assert result["access_token"] == "tok2"

    def test_run_oauth_flow_http_error(self, monkeypatch):
        monkeypatch.setattr("mullet.core.mcp.oauth._oauth_port", None)
        with patch(
            "mullet.core.mcp.oauth._redirect_to_browser", new_callable=AsyncMock
        ), patch(
            "mullet.core.mcp.oauth._wait_for_callback",
            new_callable=AsyncMock,
            return_value=("code", "state_val"),
        ), patch(
            "mullet.core.mcp.oauth.secrets.token_urlsafe",
            side_effect=["verifier96", "state_val"],
        ), patch(
            "httpx.AsyncClient"
        ) as MockClient:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = RuntimeError("HTTP Error")
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.post.return_value = mock_resp
            MockClient.return_value = mock_client
            with pytest.raises(RuntimeError, match="HTTP Error"):
                asyncio.run(
                    run_oauth_flow("test", "http://s", "cid", "http://a", "http://t")
                )


def test_wait_for_callback_serve_function_body():
    """Lines 146-147: _serve() inner function runs server.timeout + handle_request."""
    import asyncio
    serve_calls = []

    class FakeServer:
        def handle_request(self):
            serve_calls.append("handled")
        def server_close(self):
            pass

    fake_server = FakeServer()

    with patch("mullet.core.mcp.oauth.HTTPServer", return_value=fake_server), \
         patch("mullet.core.mcp.oauth._oauth_port", None), \
         patch("mullet.core.mcp.oauth._find_free_port", return_value=19876):
        # Override threading.Thread to run _serve synchronously before we await

        class SyncThread:
            def __init__(self, target=None, daemon=None):
                self._target = target
            def start(self):
                if self._target:
                    self._target()  # run _serve inline — hits lines 146-147

        with patch("mullet.core.mcp.oauth.threading.Thread", SyncThread):
            # Pre-set result so _wait_for_callback exits immediately
            OAuthCallbackHandler._result = {"auth_code": "code_xyz", "state": "s"}
            code, state = asyncio.run(_wait_for_callback())
            assert code == "code_xyz"

    assert "handled" in serve_calls, "_serve() body (lines 146-147) never executed"
