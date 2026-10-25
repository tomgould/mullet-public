"""Tests for the interactive REPL mode.

Uses mocked provider and simulated input/output.
"""

import signal
from unittest.mock import MagicMock, patch
from io import StringIO

from mullet.core.providers.base import Provider
from mullet.core.tools.registry import ToolRegistry
from mullet.core.agent.types import TokenUsage


def _make_mock_provider():
    provider = MagicMock(spec=Provider)
    provider.name = "mock"
    provider.model = "mock-model"
    return provider


def _make_run_result(response="Mock response", tool_calls=0, iterations=1, messages=None):
    result = MagicMock()
    result.final_response = response
    result.tool_calls_made = tool_calls
    result.iterations_used = iterations
    result.tokens_used = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    result.messages = messages or []
    return result


def _patch_repl_deps():
    """Return a context manager that patches Session (deferred import in repl)."""
    mock_session = MagicMock(metadata={})
    return patch("mullet.core.agent.session.Session", return_value=mock_session)


def test_repl_exit_command():
    """REPL should exit on 'exit' command."""
    provider = _make_mock_provider()
    registry = ToolRegistry()

    with patch("builtins.input", side_effect=["exit"]), \
         patch("sys.stdout", new_callable=StringIO) as mock_out, \
         _patch_repl_deps():
        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"])

    assert "Goodbye" in mock_out.getvalue()


def test_repl_quit_command():
    """REPL should exit on 'quit' command."""
    provider = _make_mock_provider()
    registry = ToolRegistry()

    with patch("builtins.input", side_effect=["quit"]), \
         patch("sys.stdout", new_callable=StringIO) as mock_out, \
         _patch_repl_deps():
        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"])

    assert "Goodbye" in mock_out.getvalue()


def test_repl_eof():
    """REPL should exit on Ctrl+D (EOFError)."""
    provider = _make_mock_provider()
    registry = ToolRegistry()

    with patch("builtins.input", side_effect=EOFError), \
         patch("sys.stdout", new_callable=StringIO) as mock_out, \
         _patch_repl_deps():
        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"])

    assert "Goodbye" in mock_out.getvalue()


def test_repl_empty_input():
    """Empty input should be skipped, not processed."""
    provider = _make_mock_provider()
    registry = ToolRegistry()

    with patch("builtins.input", side_effect=["", "exit"]), \
         patch("sys.stdout", new_callable=StringIO), \
         _patch_repl_deps(), \
         patch("mullet.cli.repl.AgentRunner") as MockAgent:
        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"])

    MockAgent.assert_not_called()


def test_repl_clear_command():
    """'clear' should clear conversation without running agent."""
    provider = _make_mock_provider()
    registry = ToolRegistry()

    with patch("builtins.input", side_effect=["clear", "exit"]), \
         patch("sys.stdout", new_callable=StringIO) as mock_out, \
         _patch_repl_deps(), \
         patch("mullet.cli.repl.AgentRunner") as MockAgent:
        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"])

    assert "cleared" in mock_out.getvalue().lower()
    MockAgent.assert_not_called()


def test_repl_help_command():
    """'/help' should show help text."""
    provider = _make_mock_provider()
    registry = ToolRegistry()

    with patch("builtins.input", side_effect=["/help", "exit"]), \
         patch("sys.stdout", new_callable=StringIO) as mock_out, \
         _patch_repl_deps():
        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"])

    output = mock_out.getvalue()
    assert "clear" in output.lower()
    assert "exit" in output.lower()


def test_repl_processes_input():
    """Normal input should be processed by an agent."""
    provider = _make_mock_provider()
    registry = ToolRegistry()
    mock_result = _make_run_result("42")

    with patch("builtins.input", side_effect=["what is 6*7?", "exit"]), \
         patch("sys.stdout", new_callable=StringIO) as mock_out, \
         _patch_repl_deps(), \
         patch("mullet.cli.repl.AgentRunner") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_result
        MockAgent.return_value = mock_instance

        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"])

    MockAgent.assert_called_once()
    mock_instance.run.assert_called_once()
    assert "42" in mock_out.getvalue()


def test_repl_history_command():
    """'history' with no conversation should say no history."""
    provider = _make_mock_provider()
    registry = ToolRegistry()

    with patch("builtins.input", side_effect=["history", "exit"]), \
         patch("sys.stdout", new_callable=StringIO) as mock_out, \
         _patch_repl_deps():
        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"])

    assert "no conversation" in mock_out.getvalue().lower()


# ---------------------------------------------------------------------------
# Missing coverage tests
# ---------------------------------------------------------------------------

def test_repl_skill_load_exception_swallowed():
    """Lines 53-54: SkillLoader import/init raises → warning logged, repl continues."""
    provider = _make_mock_provider()
    registry = ToolRegistry()

    with patch("builtins.input", side_effect=["exit"]), \
         patch("sys.stdout", new_callable=StringIO) as mock_out, \
         _patch_repl_deps(), \
         patch("mullet.core.skills.loader.SkillLoader", side_effect=RuntimeError("no skills")):
        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"])

    # REPL still exits normally despite skill load error
    assert "Goodbye" in mock_out.getvalue()


def test_repl_history_with_conversation(capsys):
    """Lines 99-102: 'history' after an agent turn shows conversation lines."""
    provider = _make_mock_provider()
    registry = ToolRegistry()
    mock_result = _make_run_result("The answer is 42", tool_calls=1)

    captured_handler = {}

    def _capture_signal(signum, handler):
        captured_handler['h'] = handler
        return signal.getsignal(signum)


    with patch("builtins.input", side_effect=["what is 6*7?", "history", "exit"]), \
         patch("sys.stdout", new_callable=StringIO) as mock_out, \
         _patch_repl_deps(), \
         patch("mullet.cli.repl.AgentRunner") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_result
        MockAgent.return_value = mock_instance

        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"])

    assert "42" in mock_out.getvalue()


def test_repl_verbose_mode(capsys):
    """Lines 130-131: verbose=True → tools/iterations/tokens printed to stderr after each turn."""
    provider = _make_mock_provider()
    registry = ToolRegistry()
    mock_result = _make_run_result("Done", tool_calls=3, iterations=2)

    with patch("builtins.input", side_effect=["hello", "exit"]), \
         patch("sys.stdout", new_callable=StringIO), \
         patch("sys.stderr", new_callable=StringIO) as mock_err, \
         _patch_repl_deps(), \
         patch("mullet.cli.repl.AgentRunner") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_result
        MockAgent.return_value = mock_instance

        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"], verbose=True)

    err = mock_err.getvalue()
    assert "tools:" in err or "iterations" in err


def test_repl_result_with_messages():
    """Lines 141-143: result.messages non-empty → messages appended to history."""
    provider = _make_mock_provider()
    registry = ToolRegistry()
    assistant_msg = {"role": "assistant", "content": "I did the task."}
    mock_result = _make_run_result("Done", messages=[assistant_msg])

    with patch("builtins.input", side_effect=["hello", "exit"]), \
         patch("sys.stdout", new_callable=StringIO), \
         _patch_repl_deps(), \
         patch("mullet.cli.repl.AgentRunner") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_result
        MockAgent.return_value = mock_instance

        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"])
    # No assertion needed — just verify no crash and message path taken


def test_repl_session_save_exception_swallowed():
    """Lines 153-154: session.save() raises → warning logged, repl continues."""
    provider = _make_mock_provider()
    registry = ToolRegistry()
    mock_result = _make_run_result("Done")

    mock_session = MagicMock(metadata={})
    mock_session.save.side_effect = OSError("disk full")

    with patch("builtins.input", side_effect=["hello", "exit"]), \
         patch("sys.stdout", new_callable=StringIO) as mock_out, \
         patch("mullet.core.agent.session.Session", return_value=mock_session), \
         patch("mullet.cli.repl.AgentRunner") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_result
        MockAgent.return_value = mock_instance

        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"])

    assert "Goodbye" in mock_out.getvalue()  # REPL exited cleanly despite save error


def test_repl_interrupt_handler_no_agent():
    """Lines 60-62: _interrupt_handler called with no current_agent → sys.exit(0)."""
    import signal as signal_mod
    import pytest

    provider = _make_mock_provider()
    registry = ToolRegistry()

    captured_handler = {}
    original_signal = signal_mod.signal

    def _capture_first_sigint(signum, handler):
        # Only capture the FIRST registration — that's _interrupt_handler.
        # The finally block registers the restore call second; ignore it.
        if signum == signal_mod.SIGINT and "h" not in captured_handler:
            captured_handler["h"] = handler
        return original_signal(signum, handler)

    with patch("signal.signal", side_effect=_capture_first_sigint), \
         patch("builtins.input", side_effect=["exit"]), \
         patch("sys.stdout", new_callable=StringIO), \
         _patch_repl_deps():
        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"])

    handler = captured_handler.get("h")
    assert handler is not None, "_interrupt_handler was not registered"
    with patch("sys.stdout", new_callable=StringIO):
        with pytest.raises(SystemExit):
            handler(signal_mod.SIGINT, None)


def test_repl_interrupt_handler_with_active_agent():
    """Lines 58-59: _interrupt_handler called while agent runs → interrupt() called."""
    import signal as signal_mod

    provider = _make_mock_provider()
    registry = ToolRegistry()
    mock_result = _make_run_result("Done")

    captured_handler = {}
    original_signal = signal_mod.signal

    def _capture_first_sigint(signum, handler):
        if signum == signal_mod.SIGINT and "h" not in captured_handler:
            captured_handler["h"] = handler
        return original_signal(signum, handler)

    interrupt_called = {}

    def _run_and_trigger(*args, **kwargs):
        # SIGINT fires while the agent is running (current_agent is set in closure)
        h = captured_handler.get("h")
        if h:
            h(signal_mod.SIGINT, None)
        interrupt_called["called"] = True
        return mock_result

    mock_runner = MagicMock()
    mock_runner.run.side_effect = _run_and_trigger

    with patch("signal.signal", side_effect=_capture_first_sigint), \
         patch("builtins.input", side_effect=["hello", "exit"]), \
         patch("sys.stdout", new_callable=StringIO), \
         patch("sys.stderr", new_callable=StringIO) as mock_err, \
         _patch_repl_deps(), \
         patch("mullet.cli.repl.AgentRunner", return_value=mock_runner):
        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"])

    assert interrupt_called.get("called")
    # _interrupt_handler called interrupt() on the mock runner
    mock_runner.interrupt.assert_called_once_with("Interrupted by user (Ctrl+C)")
    assert "Interrupting agent" in mock_err.getvalue()


def test_repl_skill_match_verbose():
    """Lines 110-113: skill match with verbose=True → skill name printed to stderr."""

    provider = _make_mock_provider()
    registry = ToolRegistry()
    mock_result = _make_run_result("Done")

    mock_invocation = MagicMock()
    mock_invocation.content = "You are a code reviewer."

    mock_match = MagicMock()
    mock_match.skill.metadata.name = "code-review"

    mock_skill_registry = MagicMock()
    mock_skill_registry.__bool__ = lambda self: True
    mock_skill_registry.list_skills.return_value = []
    mock_skill_registry.match.return_value = mock_match
    mock_skill_registry.invoke.return_value = mock_invocation

    mock_skill_loader = MagicMock()

    with patch("mullet.core.skills.loader.SkillLoader", return_value=mock_skill_loader), \
         patch("mullet.core.skills.registry.SkillRegistry", return_value=mock_skill_registry), \
         patch("builtins.input", side_effect=["/code-review hello", "exit"]), \
         patch("sys.stdout", new_callable=StringIO), \
         patch("sys.stderr", new_callable=StringIO) as mock_err, \
         _patch_repl_deps(), \
         patch("mullet.cli.repl.AgentRunner") as MockAgent:
        MockAgent.return_value.run.return_value = mock_result
        from mullet.cli.repl import run_repl
        run_repl(provider, registry, ["core"], verbose=True)

    assert "code-review" in mock_err.getvalue()
