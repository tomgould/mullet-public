import os
import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch


from mullet.hud.guard import BudgetStatus
from mullet.hud.statusline import _active_provider_from_log, render_statusline


class TestActiveProviderFromLog:
    def test_missing_file(self, tmp_path):
        log = tmp_path / "missing.log"
        assert _active_provider_from_log(log) is None

    def test_empty_file(self, tmp_path):
        log = tmp_path / "empty.log"
        log.write_text("")
        assert _active_provider_from_log(log) is None

    def test_valid_line(self, tmp_path):
        log = tmp_path / "routing.log"
        log.write_text("2026-04-28 22:03:15  kimi/kimi-k2.6  TIER  reasons  |  task\n")
        assert _active_provider_from_log(log) == "kimi"

    def test_no_slash_in_field(self, tmp_path):
        log = tmp_path / "routing.log"
        log.write_text("2026-04-28 22:03:15  provider  TIER  reasons\n")
        assert _active_provider_from_log(log) is None

    def test_malformed_single_part(self, tmp_path):
        log = tmp_path / "routing.log"
        log.write_text("onlyonefield\n")
        assert _active_provider_from_log(log) is None

    def test_whitespace_only_last_line(self, tmp_path):
        """Line 26: last line is blank after stripping → None."""
        log = tmp_path / "routing.log"
        log.write_text("valid line\n   \n")
        # Last line is whitespace; returns None
        assert _active_provider_from_log(log) is None


class TestRenderStatusline:
    @patch("mullet.hud.statusline.check_budget")
    def test_go_no_provider(self, mock_check, tmp_path, monkeypatch):
        mock_check.return_value = MagicMock(status=BudgetStatus.GO)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert render_statusline() == "●GO"

    @patch("mullet.hud.statusline.check_budget")
    def test_throttle_no_provider(self, mock_check, tmp_path, monkeypatch):
        mock_check.return_value = MagicMock(status=BudgetStatus.THROTTLE)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert render_statusline() == "⚠THROTTLE"

    @patch("mullet.hud.statusline.check_budget")
    def test_pause_no_provider(self, mock_check, tmp_path, monkeypatch):
        mock_check.return_value = MagicMock(status=BudgetStatus.PAUSE)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert render_statusline() == "✗PAUSE"

    @patch("mullet.hud.statusline.check_budget")
    def test_unknown_status(self, mock_check, tmp_path, monkeypatch):
        mock_check.return_value = MagicMock(status="weird")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert render_statusline() == "●?"

    @patch("mullet.hud.statusline.check_budget")
    def test_go_with_provider(self, mock_check, tmp_path, monkeypatch):
        mock_check.return_value = MagicMock(status=BudgetStatus.GO)
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        log = log_dir / "routing.log"
        log.write_text("2026-04-28 22:03:15  kimi/kimi-k2.6  TIER  reasons  |  task\n")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert render_statusline() == "●GO kimi"

    @patch("mullet.hud.statusline.check_budget")
    def test_check_budget_raises(self, mock_check, tmp_path, monkeypatch):
        mock_check.side_effect = RuntimeError("boom")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert render_statusline() == "●?"

    @patch("mullet.hud.statusline.check_budget")
    @patch("mullet.hud.statusline._active_provider_from_log", side_effect=RuntimeError("perm error"))
    def test_log_read_exception_is_swallowed(self, mock_read, mock_check, tmp_path, monkeypatch):
        """Lines 65-66: exception reading routing.log is caught and ignored."""
        mock_check.return_value = MagicMock(status=BudgetStatus.GO)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / ".mullet").mkdir()
        (tmp_path / ".mullet" / "routing.log").write_text("some line\n")
        # _active_provider_from_log raises → except block at lines 65-66
        result = render_statusline()
        assert result == "●GO"


def test_main_function_prints_statusline(capsys):
    """Line 75: main() calls print(render_statusline())."""
    from mullet.hud.statusline import main
    with patch("mullet.hud.statusline.render_statusline", return_value="●GO kimi"):
        main()
    captured = capsys.readouterr()
    assert "●GO kimi" in captured.out


def test_statusline_main_block_runs():
    """Line 79: if __name__ == '__main__' block executes without error."""
    statusline_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../mullet/hud/statusline.py")
    )
    with patch("mullet.hud.statusline.render_statusline", return_value="●GO"):
        runpy.run_path(statusline_path, run_name="__main__")
