"""Structured logging configuration for Mullet.

JSON logging for production, human-readable for development.
Supports request-scoped context via contextvars.

Environment variables:
    MULLET_LOG_FORMAT=json   — Force JSON output
    MULLET_LOG_LEVEL=DEBUG   — Set log level
    MULLET_LOG_FILE=path     — Also log to file (always JSON)
"""

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Request-scoped context for structured logging
_request_id: ContextVar[str] = ContextVar("request_id", default="")
_task_id: ContextVar[str] = ContextVar("task_id", default="")


def set_request_context(request_id: str = "", task_id: str = ""):
    """Set request-scoped context visible in all log records."""
    if request_id:
        _request_id.set(request_id)
    if task_id:
        _task_id.set(task_id)


def new_request_id() -> str:
    """Generate and set a new request ID. Returns the ID."""
    rid = uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


class StructuredFormatter(logging.Formatter):
    """JSON log formatter for production/machine-parseable output.

    Includes request context, extra fields (provider, tokens, cost),
    and automatic secret redaction.
    """

    def format(self, record):
        """Serialise `record` to a single-line JSON dict (with secrets redacted)."""
        log_entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Add request context
        rid = _request_id.get("")
        if rid:
            log_entry["request_id"] = rid
        tid = _task_id.get("")
        if tid:
            log_entry["task_id"] = tid

        # Add extra structured fields
        for field in ("provider", "tier", "tokens", "cost", "duration_ms", "tool_name"):
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)

        if record.exc_info and record.exc_info[0]:
            log_entry["error"] = self.formatException(record.exc_info)
            log_entry["error_type"] = record.exc_info[0].__name__

        # Redact secrets from log messages
        try:
            from mullet.core.agent.redact import redact_sensitive_text
            log_entry["msg"] = redact_sensitive_text(log_entry["msg"])
        except ImportError as _exc:
            logger.warning("logging: Redaction library not found: %s", _exc)
            pass

        return json.dumps(log_entry, default=str)


class HumanFormatter(logging.Formatter):
    """Colored human-readable formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        """Render `record` as a colour-coded `LEVEL   logger: message` string."""
        color = self.COLORS.get(record.levelname, "")
        name = record.name.replace("mullet.", "")
        return f"{color}{record.levelname:7s}{self.RESET} {name}: {record.getMessage()}"


def setup_logging(
    verbose: bool = False,
    json_logs: bool | None = None,
    log_file: str | None = None,
):
    """Configure logging for the application.

    Args:
        verbose: Show DEBUG level output.
        json_logs: Use JSON format. Auto-detects from MULLET_LOG_FORMAT env.
        log_file: Optional file for structured logs. Auto-detects from MULLET_LOG_FILE env.
    """
    if json_logs is None:
        json_logs = os.environ.get("MULLET_LOG_FORMAT", "").lower() == "json"
    if log_file is None:
        log_file = os.environ.get("MULLET_LOG_FILE", "")

    level_str = os.environ.get("MULLET_LOG_LEVEL", "")
    if level_str:
        level = getattr(logging, level_str.upper(), logging.WARNING)
    else:
        level = logging.DEBUG if verbose else logging.WARNING

    root = logging.getLogger("mullet")
    root.handlers.clear()

    # Console handler
    handler = logging.StreamHandler(sys.stderr)
    if json_logs:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(HumanFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    # File handler (always JSON for structured analysis)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(StructuredFormatter())
        root.addHandler(fh)

    # Quiet noisy third-party loggers
    for name in ("openai", "httpcore", "httpx", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)
