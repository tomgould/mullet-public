"""Tests for harness hook payload schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mullet.harness.hooks.types import (
    EventPayload,
    MessagePayload,
    ParamsPayload,
    PostToolUsePayload,
    PreToolUsePayload,
    TransformPayload,
)

pytestmark = pytest.mark.harness


# --- PreToolUsePayload ---------------------------------------------------


def test_pre_tool_use_required_fields() -> None:
    p = PreToolUsePayload(
        tool_name="Read",
        tool_args={"path": "/tmp/x"},
        session_id="s1",
    )
    assert p.tool_name == "Read"
    assert p.tool_args == {"path": "/tmp/x"}
    assert p.session_id == "s1"
    assert p.tool_call_id is None


def test_pre_tool_use_extra_allowed() -> None:
    p = PreToolUsePayload(
        tool_name="Read",
        tool_args={},
        session_id="s1",
        trace_id="abc",  # type: ignore[call-arg]
    )
    assert p.trace_id == "abc"  # type: ignore[attr-defined]


def test_pre_tool_use_rejects_non_dict_args() -> None:
    with pytest.raises(ValidationError):
        PreToolUsePayload(
            tool_name="Read",
            tool_args=["not", "a", "dict"],  # type: ignore[arg-type]
            session_id="s1",
        )


# --- PostToolUsePayload --------------------------------------------------


def test_post_tool_use_required_fields() -> None:
    p = PostToolUsePayload(
        tool_name="Bash",
        tool_args={"cmd": "ls"},
        tool_result="ok",
        tool_error=None,
        session_id="s1",
    )
    assert p.tool_result == "ok"
    assert p.tool_error is None
    assert p.tool_call_id is None


def test_post_tool_use_extra_allowed() -> None:
    p = PostToolUsePayload(
        tool_name="Bash",
        tool_args={},
        tool_result=None,
        tool_error="boom",
        session_id="s1",
        duration_ms=42,  # type: ignore[call-arg]
    )
    assert p.duration_ms == 42  # type: ignore[attr-defined]
    assert p.tool_error == "boom"


# --- MessagePayload ------------------------------------------------------


def test_message_payload_required_fields() -> None:
    p = MessagePayload(
        message={"role": "user", "content": "hi"},
        session_id="s1",
        direction="user",
    )
    assert p.message["role"] == "user"
    assert p.direction == "user"


def test_message_payload_extra_allowed() -> None:
    p = MessagePayload(
        message={},
        session_id="s1",
        direction="assistant",
        plugin_tag="rate-limiter",  # type: ignore[call-arg]
    )
    assert p.plugin_tag == "rate-limiter"  # type: ignore[attr-defined]


def test_message_payload_rejects_unknown_direction() -> None:
    with pytest.raises(ValidationError):
        MessagePayload(
            message={},
            session_id="s1",
            direction="bogus",  # type: ignore[arg-type]
        )


# --- EventPayload --------------------------------------------------------


def test_event_payload_defaults() -> None:
    p = EventPayload(event="session.created", session_id="s1")
    assert p.error is None
    assert p.payload == {}


def test_event_payload_extra_allowed() -> None:
    p = EventPayload(
        event="session.error",
        session_id="s1",
        error="rate-limited",
        payload={"retry_after": 5},
        attempt=2,  # type: ignore[call-arg]
    )
    assert p.error == "rate-limited"
    assert p.payload == {"retry_after": 5}
    assert p.attempt == 2  # type: ignore[attr-defined]


# --- TransformPayload ----------------------------------------------------


def test_transform_payload_required_fields() -> None:
    p = TransformPayload(
        messages=[{"role": "user", "content": "hi"}],
        session_id="s1",
        purpose="compact",
    )
    assert p.purpose == "compact"
    assert len(p.messages) == 1


def test_transform_payload_rejects_unknown_purpose() -> None:
    with pytest.raises(ValidationError):
        TransformPayload(
            messages=[],
            session_id="s1",
            purpose="rewrite",  # type: ignore[arg-type]
        )


def test_transform_payload_extra_allowed() -> None:
    p = TransformPayload(
        messages=[],
        session_id="s1",
        purpose="inject",
        reason="add system prompt",  # type: ignore[call-arg]
    )
    assert p.reason == "add system prompt"  # type: ignore[attr-defined]


# --- ParamsPayload -------------------------------------------------------


def test_params_payload_required_fields_with_defaults() -> None:
    p = ParamsPayload(
        messages=[{"role": "user", "content": "hi"}],
        provider="kimi",
        model=None,
        session_id="s1",
    )
    assert p.provider == "kimi"
    assert p.model is None
    assert p.max_tokens is None
    assert p.temperature is None
    assert p.thinking is None


def test_params_payload_full() -> None:
    p = ParamsPayload(
        messages=[],
        provider="anthropic",
        model="claude-opus-4",
        session_id="s1",
        max_tokens=2048,
        temperature=0.2,
        thinking={"type": "enabled", "budget_tokens": 1000},
    )
    assert p.max_tokens == 2048
    assert p.temperature == 0.2
    assert p.thinking == {"type": "enabled", "budget_tokens": 1000}


def test_params_payload_rejects_non_int_max_tokens() -> None:
    with pytest.raises(ValidationError):
        ParamsPayload(
            messages=[],
            provider="kimi",
            model=None,
            session_id="s1",
            max_tokens="lots",  # type: ignore[arg-type]
        )


def test_params_payload_extra_allowed() -> None:
    p = ParamsPayload(
        messages=[],
        provider="kimi",
        model=None,
        session_id="s1",
        budget_tier="STANDARD",  # type: ignore[call-arg]
    )
    assert p.budget_tier == "STANDARD"  # type: ignore[attr-defined]
