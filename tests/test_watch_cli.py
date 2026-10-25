"""Tests for mullet.cli.watch — run_watch and error paths."""
from unittest.mock import MagicMock, patch


from mullet.cli.watch import _scan_files, run_watch


class TestScanFilesError:
    def test_oserror_logs_warning(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "test.py").write_text("x = 1")
        with patch("mullet.cli.watch.os.path.getmtime", side_effect=OSError("perm denied")):
            result = _scan_files(str(tmp_path))
        assert result == {}
        assert "Failed to get file modification time" in caplog.text


class TestRunWatch:
    def test_detects_change_and_runs_task(self, capsys):
        args = MagicMock()
        with patch("mullet.core.agent.project_detect.detect_project") as mock_detect:
            mock_detect.return_value.language = "unknown"
            with patch("mullet.cli.watch._scan_files") as mock_scan:
                mock_scan.side_effect = [
                    {"a.py": 1.0},
                    {"a.py": 2.0},
                    {"a.py": 2.0},
                    {"a.py": 2.0},
                ]
                with patch("mullet.cli.watch.time.sleep") as mock_sleep:
                    def sleep_side_effect(_):
                        if mock_sleep.call_count >= 3:
                            raise KeyboardInterrupt()
                    mock_sleep.side_effect = sleep_side_effect
                    with patch("mullet.cli.watch.time.time") as mock_time:
                        mock_time.side_effect = [10, 20, 30, 40]
                        with patch("mullet.cli.main.run_task") as mock_run_task:
                            run_watch("test task", args, interval=1.0)
        mock_run_task.assert_called_once_with("test task", args)
        out = capsys.readouterr().out
        assert "Watch stopped" in out

    def test_run_task_system_exit_caught(self):
        args = MagicMock()
        with patch("mullet.core.agent.project_detect.detect_project") as mock_detect:
            mock_detect.return_value.language = "unknown"
            with patch("mullet.cli.watch._scan_files") as mock_scan:
                mock_scan.side_effect = [
                    {"a.py": 1.0},
                    {"a.py": 2.0},
                    {"a.py": 2.0},
                    {"a.py": 2.0},
                ]
                with patch("mullet.cli.watch.time.sleep") as mock_sleep:
                    def sleep_side_effect(_):
                        if mock_sleep.call_count >= 3:
                            raise KeyboardInterrupt()
                    mock_sleep.side_effect = sleep_side_effect
                    with patch("mullet.cli.watch.time.time") as mock_time:
                        mock_time.side_effect = [10, 20, 30, 40]
                        with patch("mullet.cli.main.run_task", side_effect=SystemExit(0)):
                            run_watch("test task", args, interval=1.0)

    def test_run_task_exception_prints_error(self, capsys):
        args = MagicMock()
        with patch("mullet.core.agent.project_detect.detect_project") as mock_detect:
            mock_detect.return_value.language = "unknown"
            with patch("mullet.cli.watch._scan_files") as mock_scan:
                mock_scan.side_effect = [
                    {"a.py": 1.0},
                    {"a.py": 2.0},
                    {"a.py": 2.0},
                    {"a.py": 2.0},
                ]
                with patch("mullet.cli.watch.time.sleep") as mock_sleep:
                    def sleep_side_effect(_):
                        if mock_sleep.call_count >= 3:
                            raise KeyboardInterrupt()
                    mock_sleep.side_effect = sleep_side_effect
                    with patch("mullet.cli.watch.time.time") as mock_time:
                        mock_time.side_effect = [10, 20, 30, 40]
                        with patch("mullet.cli.main.run_task", side_effect=RuntimeError("boom")):
                            run_watch("test task", args, interval=1.0)
        err = capsys.readouterr().err
        assert "Error: boom" in err

    def test_keyboard_interrupt_stops(self, capsys):
        args = MagicMock()
        with patch("mullet.core.agent.project_detect.detect_project") as mock_detect:
            mock_detect.return_value.language = "unknown"
            with patch("mullet.cli.watch._scan_files", return_value={}):
                with patch("mullet.cli.watch.time.sleep", side_effect=KeyboardInterrupt()):
                    run_watch("test task", args, interval=1.0)
        out = capsys.readouterr().out
        assert "Watch stopped" in out

    def test_prints_language_when_known(self, capsys):
        """Line 81: prints project language when detect_project returns non-unknown language."""
        args = MagicMock()
        with patch("mullet.core.agent.project_detect.detect_project") as mock_detect:
            mock_detect.return_value.language = "python"
            mock_detect.return_value.framework = ""
            with patch("mullet.cli.watch._scan_files", return_value={}):
                with patch("mullet.cli.watch.time.sleep", side_effect=KeyboardInterrupt()):
                    run_watch("test task", args, interval=1.0)
        out = capsys.readouterr().out
        assert "python" in out
