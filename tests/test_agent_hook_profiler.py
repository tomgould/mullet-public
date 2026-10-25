"""Unit tests for hook.py, output_format.py, profiler.py, and error_log.py."""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.agent.error_log import get_error_log_path, setup_error_log
from mullet.core.agent.hook import HookRegistry
from mullet.core.agent.output_format import AgentOutput
from mullet.core.agent.profiler import Profiler


# -----------------------------------------------------------------------------
# HookRegistry tests
# -----------------------------------------------------------------------------

class TestHookRegistry:
    def test_init_creates_empty_registries(self):
        registry = HookRegistry()
        assert registry._pre_hooks == {}
        assert registry._post_hooks == {}

    def test_register_and_execute_pre_hook(self):
        registry = HookRegistry()
        mock_cb = MagicMock()
        registry.register_pre_hook("tool_a", mock_cb)
        registry.execute_pre_hooks("tool_a", {"arg": 1})
        mock_cb.assert_called_once_with("tool_a", {"arg": 1})

    def test_register_and_execute_post_hook(self):
        registry = HookRegistry()
        mock_cb = MagicMock()
        registry.register_post_hook("tool_b", mock_cb)
        registry.execute_post_hooks("tool_b", {"arg": 2}, "result")
        mock_cb.assert_called_once_with("tool_b", {"arg": 2}, "result")

    def test_multiple_hooks_same_tool(self):
        registry = HookRegistry()
        cb1 = MagicMock()
        cb2 = MagicMock()
        registry.register_pre_hook("tool", cb1)
        registry.register_pre_hook("tool", cb2)
        registry.execute_pre_hooks("tool", {})
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_execute_hooks_no_registered_hooks(self):
        registry = HookRegistry()
        # Should not raise
        registry.execute_pre_hooks("missing", {})
        registry.execute_post_hooks("missing", {}, None)

    def test_hooks_different_tools_isolated(self):
        registry = HookRegistry()
        cb = MagicMock()
        registry.register_pre_hook("tool1", cb)
        registry.execute_pre_hooks("tool2", {})
        cb.assert_not_called()


# -----------------------------------------------------------------------------
# AgentOutput tests
# -----------------------------------------------------------------------------

class TestAgentOutput:
    def test_format_summary_success(self):
        output = AgentOutput(
            response_text="Done",
            files_changed=["a.py"],
            commands_run=["git add"],
            tokens_used=42,
            duration_s=1.23,
            success=True,
        )
        summary = output.format_summary()
        assert "SUCCESS" in summary
        assert "1.23 seconds" in summary
        assert "42 tokens" in summary
        assert "Response: Done" in summary
        assert "  - a.py" in summary
        assert "  - git add" in summary

    def test_format_summary_failure(self):
        output = AgentOutput(
            response_text="Oops",
            files_changed=[],
            commands_run=[],
            tokens_used=10,
            duration_s=0.5,
            success=False,
        )
        summary = output.format_summary()
        assert "FAILURE" in summary
        assert "Oops" in summary
        assert "Files Changed:" not in summary
        assert "Commands Run:" not in summary

    def test_format_summary_without_response_text(self):
        output = AgentOutput(
            response_text="",
            files_changed=["b.py"],
            commands_run=[],
            tokens_used=5,
            duration_s=0.1,
            success=True,
        )
        summary = output.format_summary()
        assert "Response:" not in summary
        assert "Files Changed:" in summary

    def test_format_summary_without_files_changed(self):
        output = AgentOutput(
            response_text="OK",
            files_changed=[],
            commands_run=["echo hi"],
            tokens_used=1,
            duration_s=0.01,
            success=True,
        )
        summary = output.format_summary()
        assert "Files Changed:" not in summary
        assert "Commands Run:" in summary
        assert "  - echo hi" in summary

    def test_format_summary_without_commands_run(self):
        output = AgentOutput(
            response_text="OK",
            files_changed=["c.py"],
            commands_run=[],
            tokens_used=1,
            duration_s=0.01,
            success=True,
        )
        summary = output.format_summary()
        assert "Commands Run:" not in summary
        assert "Files Changed:" in summary


# -----------------------------------------------------------------------------
# Profiler tests
# -----------------------------------------------------------------------------

class TestProfiler:
    @patch("mullet.core.agent.profiler.time.monotonic")
    def test_start_and_stop_timer(self, mock_mono):
        mock_mono.side_effect = [100.0, 105.0]
        profiler = Profiler()
        profiler.start_timer("task1")
        profiler.stop_timer("task1")
        assert profiler._durations == {"task1": 5.0}
        assert profiler._timers == {}

    @patch("mullet.core.agent.profiler.time.monotonic")
    def test_start_timer_already_started_raises(self, mock_mono):
        mock_mono.return_value = 10.0
        profiler = Profiler()
        profiler.start_timer("x")
        with pytest.raises(ValueError, match="Timer 'x' already started."):
            profiler.start_timer("x")

    def test_stop_timer_not_started_raises(self):
        profiler = Profiler()
        with pytest.raises(ValueError, match="Timer 'y' not started."):
            profiler.stop_timer("y")

    @patch("mullet.core.agent.profiler.time.monotonic")
    def test_report_with_running_timers(self, mock_mono):
        mock_mono.side_effect = [0.0, 3.0]
        profiler = Profiler()
        profiler.start_timer("running")
        report = profiler.report()
        assert report == {"running": 3.0}
        assert profiler._timers == {}
        assert profiler._durations == {"running": 3.0}

    @patch("mullet.core.agent.profiler.time.monotonic")
    def test_report_with_stopped_and_running_timers(self, mock_mono):
        # start t1, stop t1, start t2, report
        mock_mono.side_effect = [0.0, 2.0, 5.0, 10.0]
        profiler = Profiler()
        profiler.start_timer("stopped")
        profiler.stop_timer("stopped")
        profiler.start_timer("running")
        report = profiler.report()
        assert report["stopped"] == 2.0
        assert report["running"] == 5.0
        assert profiler._timers == {}

    @patch("mullet.core.agent.profiler.time.monotonic")
    def test_report_empty(self, mock_mono):
        mock_mono.return_value = 0.0
        profiler = Profiler()
        assert profiler.report() == {}

    @patch("mullet.core.agent.profiler.time.monotonic")
    def test_stop_timer_accumulates_duration(self, mock_mono):
        mock_mono.side_effect = [0.0, 2.0, 10.0, 13.0]
        profiler = Profiler()
        profiler.start_timer("t")
        profiler.stop_timer("t")
        profiler.start_timer("t")
        profiler.stop_timer("t")
        assert profiler._durations["t"] == 5.0

    def test_reset(self):
        profiler = Profiler()
        profiler._timers = {"a": 1.0}
        profiler._durations = {"a": 2.0}
        profiler.reset()
        assert profiler._timers == {}
        assert profiler._durations == {}


# -----------------------------------------------------------------------------
# error_log tests
# -----------------------------------------------------------------------------

class TestErrorLog:
    @patch("mullet.core.agent.error_log.os.makedirs")
    def test_get_error_log_path(self, mock_makedirs, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", "/home/test/.local/state")
        path = get_error_log_path()
        assert path == "/home/test/.local/state/mullet/logs/errors.log"
        mock_makedirs.assert_called_once_with("/home/test/.local/state/mullet/logs", exist_ok=True)

    @patch("mullet.core.agent.error_log.get_error_log_path")
    def test_setup_error_log_adds_handler(self, mock_get_path, tmp_path):
        log_file = str(tmp_path / "errors.log")
        mock_get_path.return_value = log_file

        root_logger = logging.getLogger()
        # Clear any existing RotatingFileHandler for this path to ensure clean state
        original_handlers = list(root_logger.handlers)
        root_logger.handlers = [
            h for h in root_logger.handlers
            if not (isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == log_file)
        ]

        try:
            setup_error_log()
            handlers = [
                h for h in root_logger.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == log_file
            ]
            assert len(handlers) == 1
            handler = handlers[0]
            assert handler.level == logging.WARNING
            assert isinstance(handler.formatter, logging.Formatter)
        finally:
            root_logger.handlers = original_handlers

    @patch("mullet.core.agent.error_log.get_error_log_path")
    def test_setup_error_log_idempotent(self, mock_get_path, tmp_path):
        log_file = str(tmp_path / "errors.log")
        mock_get_path.return_value = log_file

        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        root_logger.handlers = [
            h for h in root_logger.handlers
            if not (isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == log_file)
        ]

        try:
            setup_error_log()
            setup_error_log()
            handlers = [
                h for h in root_logger.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == log_file
            ]
            assert len(handlers) == 1
        finally:
            root_logger.handlers = original_handlers

    @patch("mullet.core.agent.error_log.get_error_log_path")
    def test_setup_error_log_sets_level_when_notset(self, mock_get_path, tmp_path):
        log_file = str(tmp_path / "errors.log")
        mock_get_path.return_value = log_file

        root_logger = logging.getLogger()
        original_level = root_logger.level
        original_handlers = list(root_logger.handlers)
        root_logger.handlers = [
            h for h in root_logger.handlers
            if not (isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == log_file)
        ]
        root_logger.setLevel(logging.NOTSET)

        try:
            setup_error_log()
            assert root_logger.level == logging.WARNING
        finally:
            root_logger.setLevel(original_level)
            root_logger.handlers = original_handlers

    @patch("mullet.core.agent.error_log.get_error_log_path")
    def test_setup_error_log_preserves_existing_level(self, mock_get_path, tmp_path):
        log_file = str(tmp_path / "errors.log")
        mock_get_path.return_value = log_file

        root_logger = logging.getLogger()
        original_level = root_logger.level
        original_handlers = list(root_logger.handlers)
        root_logger.handlers = [
            h for h in root_logger.handlers
            if not (isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == log_file)
        ]
        root_logger.setLevel(logging.DEBUG)

        try:
            setup_error_log()
            assert root_logger.level == logging.DEBUG
        finally:
            root_logger.setLevel(original_level)
            root_logger.handlers = original_handlers

    def test_setup_error_log_real_path(self):
        # Minimal integration check: get_error_log_path works without mocking
        path = get_error_log_path()
        assert path.endswith("errors.log")
        assert os.path.isdir(os.path.dirname(path))
