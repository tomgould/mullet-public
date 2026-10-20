"""Tests for SIGINT handler restoration on REPL exit.

Ensures `run_repl` does not leak its custom SIGINT handler into the host
process after the REPL loop exits (normal quit, Ctrl+D, or exception).
Without the fix the REPL installs `_interrupt_handler` via
`signal.signal(SIGINT, ...)` and never restores the previous handler,
leaving long-lived processes with the wrong signal state.
"""

import signal
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.providers.base import Provider
from mullet.core.tools.registry import ToolRegistry


def _make_mock_provider():
    provider = MagicMock(spec=Provider)
    provider.name = "mock"
    provider.model = "mock-model"
    return provider


def _patch_repl_deps():
    mock_session = MagicMock(metadata={})
    return patch("mullet.core.agent.session.Session", return_value=mock_session)


def _sentinel_handler(signum, frame):  # pragma: no cover - never invoked
    pass


def test_repl_restores_previous_sigint_on_normal_exit():
    """After `run_repl` returns via 'exit', SIGINT should point at the
    previously-installed handler, NOT at `_interrupt_handler`.

    Expected failure (pre-fix): `signal.getsignal(SIGINT)` is the REPL's
    nested `_interrupt_handler` closure, not `_sentinel_handler`.
    """
    provider = _make_mock_provider()
    registry = ToolRegistry()

    previous = signal.signal(signal.SIGINT, _sentinel_handler)
    try:
        with patch("builtins.input", side_effect=["exit"]), \
             patch("sys.stdout", new_callable=StringIO), \
             _patch_repl_deps():
            from mullet.cli.repl import run_repl
            run_repl(provider, registry, ["core"])

        current = signal.getsignal(signal.SIGINT)
        assert current is _sentinel_handler, (
            f"SIGINT handler not restored after normal exit; "
            f"got {current!r}, expected {_sentinel_handler!r}"
        )
    finally:
        signal.signal(signal.SIGINT, previous)


def test_repl_restores_previous_sigint_on_exception():
    """If the REPL body raises, the previous SIGINT handler must still be
    restored (try/finally, not just try/except).

    Expected failure (pre-fix): exception propagates and the custom
    handler remains installed because there is no restoration path.
    """
    provider = _make_mock_provider()
    registry = ToolRegistry()

    previous = signal.signal(signal.SIGINT, _sentinel_handler)
    try:
        # Force the inner loop to blow up by making `input()` raise a
        # non-KeyboardInterrupt/EOFError exception (those are caught).
        with patch("builtins.input", side_effect=RuntimeError("boom")), \
             patch("sys.stdout", new_callable=StringIO), \
             _patch_repl_deps():
            from mullet.cli.repl import run_repl
            with pytest.raises(RuntimeError):
                run_repl(provider, registry, ["core"])

        current = signal.getsignal(signal.SIGINT)
        assert current is _sentinel_handler, (
            f"SIGINT handler not restored after exception; "
            f"got {current!r}, expected {_sentinel_handler!r}"
        )
    finally:
        signal.signal(signal.SIGINT, previous)


def test_repl_interrupt_handler_invoked_inside_context():
    """While the REPL is running, SIGINT must dispatch to the REPL's
    custom `_interrupt_handler` (not the sentinel we installed before).

    We capture the handler that `signal.signal` sees during REPL setup
    and assert it is a fresh callable different from the sentinel.

    Expected failure (pre-fix): also passes pre-fix (the install itself
    is unchanged); serves as a guard that the fix does not regress
    handler installation.
    """
    provider = _make_mock_provider()
    registry = ToolRegistry()

    captured = {}
    real_signal = signal.signal

    def _capture(signum, handler):
        if signum == signal.SIGINT and "repl_handler" not in captured:
            captured["repl_handler"] = handler
        return real_signal(signum, handler)

    previous = signal.signal(signal.SIGINT, _sentinel_handler)
    try:
        with patch("builtins.input", side_effect=["exit"]), \
             patch("sys.stdout", new_callable=StringIO), \
             patch("mullet.cli.repl.signal.signal", side_effect=_capture), \
             _patch_repl_deps():
            from mullet.cli.repl import run_repl
            run_repl(provider, registry, ["core"])

        assert "repl_handler" in captured, "REPL never installed a SIGINT handler"
        assert captured["repl_handler"] is not _sentinel_handler, (
            "REPL installed the sentinel instead of its own handler"
        )
        assert callable(captured["repl_handler"])
    finally:
        signal.signal(signal.SIGINT, previous)
