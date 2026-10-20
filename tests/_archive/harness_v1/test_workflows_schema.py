"""Tests for mullet.harness.workflows.schema Pydantic models."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from mullet.harness.workflows.schema import (
    WorkflowEvent,
    WorkflowMessage,
    WorkflowRun,
)

pytestmark = pytest.mark.harness


def test_run_constructs_with_required_fields() -> None:
    now = time.time()
    run = WorkflowRun(
        run_id="abc123",
        workflow_type="agent.run",
        created_at=now,
        updated_at=now,
    )
    assert run.run_id == "abc123"
    assert run.workflow_type == "agent.run"
    assert run.status == "pending"
    assert run.metadata == {}
    assert run.output is None
    assert run.parent_run_id is None
    assert run.created_at == now
    assert run.updated_at == now


def test_run_accepts_optional_fields() -> None:
    run = WorkflowRun(
        run_id="r1",
        workflow_type="ralph.loop",
        status="running",
        created_at=1.0,
        updated_at=2.0,
        metadata={"trace": "xyz"},
        output={"result": "ok"},
        parent_run_id="parent",
    )
    assert run.status == "running"
    assert run.metadata == {"trace": "xyz"}
    assert run.output == {"result": "ok"}
    assert run.parent_run_id == "parent"


def test_event_constructs() -> None:
    event = WorkflowEvent(
        event_id="e1",
        run_id="r1",
        sequence=1,
        event_type="started",
        timestamp=1234.5,
        payload={"foo": "bar"},
    )
    assert event.event_id == "e1"
    assert event.run_id == "r1"
    assert event.sequence == 1
    assert event.event_type == "started"
    assert event.timestamp == 1234.5
    assert event.payload == {"foo": "bar"}


def test_event_default_payload() -> None:
    event = WorkflowEvent(
        event_id="e1",
        run_id="r1",
        sequence=1,
        event_type="started",
        timestamp=1.0,
    )
    assert event.payload == {}


def test_message_constructs() -> None:
    msg = WorkflowMessage(
        message_id="m1",
        run_id="r1",
        sequence=1,
        role="user",
        content="hello",
        timestamp=10.0,
    )
    assert msg.message_id == "m1"
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.tool_call_id is None
    assert msg.metadata == {}


def test_message_with_tool_call_id() -> None:
    msg = WorkflowMessage(
        message_id="m2",
        run_id="r1",
        sequence=2,
        role="tool",
        content="tool output",
        timestamp=11.0,
        tool_call_id="tool-abc",
        metadata={"k": "v"},
    )
    assert msg.tool_call_id == "tool-abc"
    assert msg.metadata == {"k": "v"}


def test_workflow_status_literal_validates() -> None:
    # Valid statuses round-trip without error.
    for status in ("pending", "running", "completed", "failed", "cancelled"):
        run = WorkflowRun(
            run_id="r",
            workflow_type="t",
            status=status,  # type: ignore[arg-type]
            created_at=1.0,
            updated_at=1.0,
        )
        assert run.status == status

    # Invalid status raises ValidationError.
    with pytest.raises(ValidationError):
        WorkflowRun(
            run_id="r",
            workflow_type="t",
            status="bogus",  # type: ignore[arg-type]
            created_at=1.0,
            updated_at=1.0,
        )


def test_message_role_literal_validates() -> None:
    with pytest.raises(ValidationError):
        WorkflowMessage(
            message_id="m",
            run_id="r",
            sequence=1,
            role="boss",  # type: ignore[arg-type]
            content="x",
            timestamp=1.0,
        )
