"""Tests for the IntentGate Message hook."""

from __future__ import annotations

import pytest

from mullet.harness.engine.loop import HarnessEngine
from mullet.harness.hooks.intent_gate import (
    INTENT_CATEGORIES,
    classify_intent,
    make_intent_gate_hook,
)
from mullet.harness.hooks.types import MessagePayload

pytestmark = [pytest.mark.harness]


# ---------------------------------------------------------------------------
# 5 prompts per category — verifies keyword classifier wires the right label.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "Please research the trade-offs between SQLite and DuckDB",
        "Compare httpx and requests",
        "Explore options for caching",
        "What is event sourcing",
        "Summarise the key changes in PEP 668",
    ],
)
def test_classify_intent_research(prompt: str) -> None:
    assert classify_intent(prompt) == "research"


@pytest.mark.parametrize(
    "prompt",
    [
        "Implement a circuit breaker decorator",
        "Build a caching layer",
        "Write a function that does X",
        "Develop a CLI for the loader",
        "Make me a README scaffold",
    ],
)
def test_classify_intent_implementation(prompt: str) -> None:
    assert classify_intent(prompt) == "implementation"


@pytest.mark.parametrize(
    "prompt",
    [
        "Investigate why imports are slow",
        "Debug the failing test_engine_loop case",
        "Diagnose this stack trace",
        "Trace the call graph from cmd_shell",
        "Analyse the cold-start regression",
    ],
)
def test_classify_intent_investigation(prompt: str) -> None:
    assert classify_intent(prompt) == "investigation"


@pytest.mark.parametrize(
    "prompt",
    [
        "Review my PR for the harness layer",
        "Audit the permission checker logic",
        "Critique the proposed routing matrix",
        "Assess the test coverage",
        "Validate the YAML config schema",
    ],
)
def test_classify_intent_evaluation(prompt: str) -> None:
    assert classify_intent(prompt) == "evaluation"


@pytest.mark.parametrize(
    "prompt",
    [
        "Fix the broken Kimi provider tests",
        "Repair the CI pipeline",
        "Resolve the lint failures",
        "Patch the off-by-one in router",
        "Address the cooldown regression",
    ],
)
def test_classify_intent_fix(prompt: str) -> None:
    assert classify_intent(prompt) == "fix"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_falls_back_to_implementation_when_no_keywords() -> None:
    assert classify_intent("hello world") == "implementation"
    assert classify_intent("foo bar baz") == "implementation"


def test_priority_order_breaks_ties() -> None:
    # Tie between investigation (debug) and evaluation (check).
    # investigation should win per priority order.
    assert classify_intent("debug and check this") == "investigation"

    # Tie between evaluation (review) and research (explain).
    # evaluation outranks research.
    assert classify_intent("review and explain it") == "evaluation"

    # Tie between research (compare) and fix (resolve).
    # research outranks fix.
    assert classify_intent("compare options to resolve this") == "research"

    # Tie between fix (fix) and implementation (build) — fix wins.
    assert classify_intent("fix and build it") == "fix"


def test_classify_intent_lowercases() -> None:
    assert classify_intent("INVESTIGATE this") == "investigation"
    assert classify_intent("REVIEW the diff") == "evaluation"


def test_intent_categories_constant_has_five_entries() -> None:
    assert len(INTENT_CATEGORIES) == 5


# ---------------------------------------------------------------------------
# Hook behaviour
# ---------------------------------------------------------------------------


def test_hook_attaches_intent_to_payload() -> None:
    hook = make_intent_gate_hook()
    payload = MessagePayload(
        message={"role": "user", "content": "Investigate the failing test"},
        session_id="sess-test",
        direction="user",
    )
    out = hook(payload)
    assert out is payload
    assert payload.model_extra["intent"] == "investigation"
    # Pydantic exposes extras as attributes too.
    assert payload.intent == "investigation"  # type: ignore[attr-defined]


def test_hook_ignores_non_user_messages() -> None:
    hook = make_intent_gate_hook()
    for direction in ("assistant", "system", "tool"):
        payload = MessagePayload(
            message={"role": direction, "content": "Investigate this"},
            session_id="sess-test",
            direction=direction,  # type: ignore[arg-type]
        )
        result = hook(payload)
        assert result is None
        assert "intent" not in (payload.model_extra or {})


def test_hook_handles_string_message_content() -> None:
    hook = make_intent_gate_hook()
    payload = MessagePayload(
        message={"role": "user", "content": "fix the failing build"},
        session_id="sess-test",
        direction="user",
    )
    hook(payload)
    assert payload.model_extra["intent"] == "fix"


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


def test_engine_default_init_registers_intent_gate(tmp_path) -> None:
    from mullet.harness.sessions.store import SessionStore

    engine = HarnessEngine(store=SessionStore(base_dir=tmp_path / "sessions"))
    handlers = engine.hooks.list_handlers("Message")
    assert len(handlers) >= 1
    # At least one handler is the intent-gate closure.
    payload = MessagePayload(
        message={"role": "user", "content": "investigate this"},
        session_id="sess-default",
        direction="user",
    )
    for h in handlers:
        h(payload)
    assert payload.model_extra.get("intent") == "investigation"
