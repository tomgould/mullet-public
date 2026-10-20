"""Tests for structured logging."""

import json
import logging
import pytest
from mullet.config.logging import (
    StructuredFormatter,
    HumanFormatter,
    setup_logging,
    set_request_context,
    new_request_id,
    _request_id,
    _task_id,
)


@pytest.fixture(autouse=True)
def clear_context():
    """Clear request context between tests."""
    _request_id.set("")
    _task_id.set("")
    yield
    _request_id.set("")
    _task_id.set("")


def test_structured_formatter_basic():
    fmt = StructuredFormatter()
    record = logging.LogRecord(
        name="mullet.test", level=logging.INFO,
        pathname="test.py", lineno=1,
        msg="hello world", args=(), exc_info=None,
    )
    output = fmt.format(record)
    data = json.loads(output)
    assert data["level"] == "INFO"
    assert data["msg"] == "hello world"
    assert data["logger"] == "mullet.test"
    assert "ts" in data


def test_structured_formatter_with_request_context():
    set_request_context(request_id="abc123", task_id="task-1")
    fmt = StructuredFormatter()
    record = logging.LogRecord(
        name="mullet.test", level=logging.INFO,
        pathname="test.py", lineno=1,
        msg="with context", args=(), exc_info=None,
    )
    data = json.loads(fmt.format(record))
    assert data["request_id"] == "abc123"
    assert data["task_id"] == "task-1"


def test_structured_formatter_extra_fields():
    fmt = StructuredFormatter()
    record = logging.LogRecord(
        name="mullet.test", level=logging.INFO,
        pathname="test.py", lineno=1,
        msg="task done", args=(), exc_info=None,
    )
    record.provider = "local"
    record.tokens = 1500
    record.cost = 0.003
    data = json.loads(fmt.format(record))
    assert data["provider"] == "local"
    assert data["tokens"] == 1500
    assert data["cost"] == 0.003


def test_structured_formatter_redacts_secrets():
    fmt = StructuredFormatter()
    record = logging.LogRecord(
        name="mullet.test", level=logging.INFO,
        pathname="test.py", lineno=1,
        msg="key is sk-abcdefghijklmnopqrstuvwxyz",
        args=(), exc_info=None,
    )
    data = json.loads(fmt.format(record))
    assert "abcdefghijklmnopqrstuvwxyz" not in data["msg"]


def test_structured_formatter_exception():
    fmt = StructuredFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        import sys
        record = logging.LogRecord(
            name="mullet.test", level=logging.ERROR,
            pathname="test.py", lineno=1,
            msg="failed", args=(), exc_info=sys.exc_info(),
        )
    data = json.loads(fmt.format(record))
    assert data["error_type"] == "ValueError"
    assert "error" in data


def test_human_formatter():
    fmt = HumanFormatter()
    record = logging.LogRecord(
        name="mullet.core.agent.runner", level=logging.INFO,
        pathname="test.py", lineno=1,
        msg="hello", args=(), exc_info=None,
    )
    output = fmt.format(record)
    assert "core.agent.runner" in output
    assert "hello" in output


def test_new_request_id():
    rid = new_request_id()
    assert len(rid) == 12
    assert _request_id.get("") == rid


def test_setup_logging_verbose():
    setup_logging(verbose=True)
    logger = logging.getLogger("mullet")
    assert logger.level == logging.DEBUG


def test_setup_logging_json(monkeypatch):
    monkeypatch.setenv("MULLET_LOG_FORMAT", "json")
    setup_logging()
    logger = logging.getLogger("mullet")
    assert any(
        isinstance(h.formatter, StructuredFormatter)
        for h in logger.handlers
    )


def test_setup_logging_env_level(monkeypatch):
    monkeypatch.setenv("MULLET_LOG_LEVEL", "DEBUG")
    setup_logging()
    logger = logging.getLogger("mullet")
    assert logger.level == logging.DEBUG


def test_structured_formatter_redact_import_error(monkeypatch):
    """Lines 79-81: ImportError from redact import is swallowed; msg still returned."""
    import sys
    monkeypatch.setitem(sys.modules, "mullet.core.agent.redact", None)
    fmt = StructuredFormatter()
    record = logging.LogRecord(
        name="mullet.test", level=logging.INFO,
        pathname="test.py", lineno=1,
        msg="secret token abc123", args=(), exc_info=None,
    )
    output = fmt.format(record)
    import json as _json
    data = _json.loads(output)
    assert data["msg"] == "secret token abc123"  # redaction skipped, original preserved


def test_setup_logging_file(tmp_path):
    log_file = str(tmp_path / "mullet.log")
    setup_logging(log_file=log_file)
    logger = logging.getLogger("mullet")
    # Should have 2 handlers: console + file
    assert len(logger.handlers) >= 2
    # Cleanup
    setup_logging()
