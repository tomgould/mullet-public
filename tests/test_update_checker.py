"""Tests for mullet.cli.update_checker."""

from unittest.mock import MagicMock, patch


from mullet.cli.update_checker import (
    _version_gt,
    check_for_update,
    VERSION_URL,
    get_installed_version,
    get_latest_version,
    maybe_print_update_notice,
)


def test_get_installed_version():
    version = get_installed_version()
    assert isinstance(version, str)
    assert len(version) > 0


def test_version_gt_true():
    assert _version_gt("1.7.2", "1.7.1") is True


def test_version_gt_false():
    assert _version_gt("1.7.1", "1.7.2") is False


def test_version_gt_equal():
    assert _version_gt("1.7.1", "1.7.1") is False


def _mock_response(body: bytes) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_get_latest_version_success():
    body = b'"""Mullet."""\n\n__version__ = "1.8.0"\n'

    with patch("urllib.request.urlopen", return_value=_mock_response(body)) as mock_open:
        result = get_latest_version()

    assert result == "1.8.0"
    assert mock_open.call_args[0][0] == VERSION_URL


def test_version_url_is_the_public_github_repo_not_pypi():
    assert VERSION_URL.startswith("https://raw.githubusercontent.com/tomgould/mullet-public/")
    assert "pypi" not in VERSION_URL


def test_get_latest_version_no_version_line():
    with patch("urllib.request.urlopen", return_value=_mock_response(b"no version here")):
        assert get_latest_version() is None


def test_get_latest_version_network_error():
    from urllib.error import URLError

    with patch("urllib.request.urlopen", side_effect=URLError("timeout")):
        result = get_latest_version()

    assert result is None


def test_check_for_update_available():
    with patch("mullet.cli.update_checker.get_latest_version", return_value="99.0.0"):
        result = check_for_update()

    assert result["update_available"] is True
    assert result["latest"] == "99.0.0"
    assert "99.0.0" in result["message"]
    assert "git pull" in result["message"]
    assert "pip install -U mullet" not in result["message"]


def test_check_for_update_uptodate():
    with patch("mullet.cli.update_checker.get_latest_version", return_value="0.0.1"):
        result = check_for_update()

    assert result["update_available"] is False
    assert result["latest"] == "0.0.1"


def test_version_gt_import_error_fallback():
    """Patch packaging import to fail; fallback tuple comparison must still work."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "packaging.version":
            raise ImportError("no packaging")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=fake_import):
        assert _version_gt("2.0.0", "1.0.0") is True
        assert _version_gt("1.0.0", "2.0.0") is False
        assert _version_gt("1.0.0", "1.0.0") is False


def test_check_for_update_github_fails():
    with patch("mullet.cli.update_checker.get_latest_version", return_value=None):
        result = check_for_update()
    assert result["latest"] is None
    assert result["update_available"] is False
    assert "check failed" in result["message"]


def test_maybe_print_update_notice_prints_when_update_available(capsys):
    with patch(
        "mullet.cli.update_checker.check_for_update",
        return_value={
            "installed": "1.0.0",
            "latest": "2.0.0",
            "update_available": True,
            "message": "update available",
        },
    ):
        maybe_print_update_notice()
    captured = capsys.readouterr()
    assert "2.0.0 available" in captured.err


def test_maybe_print_update_notice_silent_on_error(capsys):
    """maybe_print_update_notice must never raise, even on unexpected errors."""
    with patch("mullet.cli.update_checker.check_for_update", side_effect=RuntimeError("boom")):
        maybe_print_update_notice()  # must not raise

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_install_help_points_at_github_not_pypi(capsys, monkeypatch):
    from mullet.cli import main as cli_main

    monkeypatch.setattr("sys.argv", ["mullet", "--install-help"])
    cli_main._main_inner()
    out = capsys.readouterr().out
    assert "git clone https://github.com/tomgould/mullet-public.git" in out
    assert "pip install mullet" not in out
