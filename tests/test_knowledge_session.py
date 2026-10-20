"""Tests for knowledge_session context manager and build_knowledge_environment.

These cover the C2 refactor: a single helper for the repeated
KnowledgeManager construct -> register -> use -> close pattern.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.knowledge.manager import (
    KnowledgeContext,
    build_knowledge_environment,
    knowledge_session,
)


def _make_fake_km():
    """Return a MagicMock standing in for a constructed KnowledgeManager."""
    km = MagicMock(name="FakeKM")
    km.build_context.return_value = KnowledgeContext(
        markdown="## Project Context\nfoo", chunks_used=1, memories_used=0
    )
    return km


def test_knowledge_session_yields_km():
    """Happy path — the context manager should yield the constructed KM.

    Failure mode: missing yield, or exception during construction not surfaced.
    """
    fake_km = _make_fake_km()
    with patch("mullet.core.knowledge.manager.KnowledgeManager", return_value=fake_km) as ctor:
        with knowledge_session("demo") as km:
            assert km is fake_km
        ctor.assert_called_once_with("demo", clients_dir=None)


def test_knowledge_session_registers_tools_when_registry_passed():
    """When register_tools is provided, knowledge_tools.register must be called.

    Failure mode: helper silently drops registration -> agents get no tools.
    """
    fake_km = _make_fake_km()
    fake_registry = MagicMock(name="ToolRegistry")
    with patch("mullet.core.knowledge.manager.KnowledgeManager", return_value=fake_km), \
         patch("mullet.core.knowledge.tools.register") as mock_register:
        with knowledge_session("demo", register_tools=fake_registry) as _km:
            pass
    mock_register.assert_called_once_with(fake_registry, fake_km)


def test_knowledge_session_skips_register_when_registry_none():
    """No registry -> never call knowledge_tools.register.

    Failure mode: helper registers anyway -> bogus tools attached.
    """
    fake_km = _make_fake_km()
    with patch("mullet.core.knowledge.manager.KnowledgeManager", return_value=fake_km), \
         patch("mullet.core.knowledge.tools.register") as mock_register:
        with knowledge_session("demo") as _km:
            pass
    mock_register.assert_not_called()


def test_knowledge_session_calls_close_on_exit():
    """KM.close() must run inside finally even when the body raises.

    Failure mode: leak — searcher / DB handles never released.
    """
    fake_km = _make_fake_km()
    with patch("mullet.core.knowledge.manager.KnowledgeManager", return_value=fake_km):
        with pytest.raises(RuntimeError, match="boom"):
            with knowledge_session("demo") as _km:
                raise RuntimeError("boom")
    fake_km.close.assert_called_once()


def test_knowledge_session_close_failure_logged_not_raised(caplog):
    """If KM.close() raises, the context manager must swallow + warn.

    Failure mode: exception during cleanup masks the real error or breaks callers.
    """
    fake_km = _make_fake_km()
    fake_km.close.side_effect = RuntimeError("close exploded")
    with patch("mullet.core.knowledge.manager.KnowledgeManager", return_value=fake_km):
        with caplog.at_level(logging.WARNING, logger="mullet.core.knowledge.manager"):
            # Should NOT raise
            with knowledge_session("demo") as _km:
                pass
    assert any(
        "KM.close failed" in record.message for record in caplog.records
    ), f"expected warning, got: {[r.message for r in caplog.records]}"


def test_build_knowledge_environment_returns_tuple():
    """Helper should return (km, KnowledgeContext, augmented_prompt).

    Failure mode: wrong return shape breaks callers that unpack 3-tuples.
    """
    fake_km = _make_fake_km()
    fake_registry = MagicMock(name="ToolRegistry")
    with patch("mullet.core.knowledge.manager.KnowledgeManager", return_value=fake_km), \
         patch("mullet.core.knowledge.tools.register"):
        km, ctx, augmented = build_knowledge_environment(
            client_name="demo",
            task="how do tools work",
            registry=fake_registry,
            system_prompt="You are helpful.",
        )
    assert km is fake_km
    assert isinstance(ctx, KnowledgeContext)
    assert ctx.markdown == "## Project Context\nfoo"
    # build_system_prompt should have prepended the base prompt
    assert augmented.startswith("You are helpful.")
    assert "Project: demo" in augmented
    assert "## Project Context" in augmented


def test_build_knowledge_environment_calls_register():
    """The helper must register knowledge tools on the passed registry.

    Failure mode: agent instance has no trove tool -> knowledge unreachable.
    """
    fake_km = _make_fake_km()
    fake_registry = MagicMock(name="ToolRegistry")
    with patch("mullet.core.knowledge.manager.KnowledgeManager", return_value=fake_km), \
         patch("mullet.core.knowledge.tools.register") as mock_register:
        build_knowledge_environment(
            client_name="demo",
            task="hello",
            registry=fake_registry,
            system_prompt="Base.",
        )
    mock_register.assert_called_once_with(fake_registry, fake_km)
    fake_km.build_context.assert_called_once_with("hello")
