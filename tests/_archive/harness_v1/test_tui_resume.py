"""Tests for :mod:`mullet.harness.tui.screens.resume_picker`."""

from __future__ import annotations

import datetime as _dt
import time

import pytest

from mullet.harness.sessions.store import SessionMeta
from mullet.harness.tui.screens.resume_picker import (
    ResumePicker,
    format_created_recently,
)


def _meta(
    sid: str = "sess-20260101-120000-aaaaaa",
    *,
    created_at: float | None = None,
    message_count: int = 3,
    cwd: str = "/home/x/proj-foo",
    project_hash_value: str = "abcdef012345",
) -> SessionMeta:
    now = created_at if created_at is not None else time.time() - 60
    return SessionMeta(
        id=sid,
        created_at=now,
        updated_at=now,
        project_hash=project_hash_value,
        cwd=cwd,
        message_count=message_count,
        status="active",
    )


@pytest.mark.harness
@pytest.mark.tui
class TestFormatCreatedRecently:
    def test_just_now(self) -> None:
        now = _dt.datetime(2026, 5, 7, 12, 0, 0).timestamp()
        assert format_created_recently(now - 5, now=now) == "just now"

    def test_minutes_ago(self) -> None:
        now = _dt.datetime(2026, 5, 7, 12, 0, 0).timestamp()
        assert format_created_recently(now - 5 * 60, now=now) == "5 minutes ago"

    def test_hours_ago(self) -> None:
        now = _dt.datetime(2026, 5, 7, 12, 0, 0).timestamp()
        assert format_created_recently(now - 2 * 3600, now=now) == "2 hours ago"

    def test_yesterday(self) -> None:
        now = _dt.datetime(2026, 5, 7, 12, 0, 0).timestamp()
        # ~26 hours ago lands in the "yesterday" bucket.
        assert format_created_recently(now - 26 * 3600, now=now) == "yesterday"

    def test_days_ago(self) -> None:
        now = _dt.datetime(2026, 5, 7, 12, 0, 0).timestamp()
        assert format_created_recently(now - 3 * 86_400, now=now) == "3 days ago"

    def test_old_falls_back_to_date(self) -> None:
        now = _dt.datetime(2026, 5, 7, 12, 0, 0).timestamp()
        ten_days_ago = now - 10 * 86_400
        result = format_created_recently(ten_days_ago, now=now)
        # ISO date for ten days before 2026-05-07.
        assert result == "2026-04-27"


@pytest.mark.harness
@pytest.mark.tui
class TestResumePicker:
    def test_resume_picker_renders_session_list(self) -> None:
        sessions = [
            _meta("sess-20260101-120000-aaaaaa", message_count=3, cwd="/p/foo"),
            _meta("sess-20260101-110000-bbbbbb", message_count=7, cwd="/p/bar"),
            _meta("sess-20260101-100000-cccccc", message_count=1, cwd="/p/baz"),
        ]
        picker = ResumePicker(sessions)
        opts = picker._render_options()
        assert len(opts) == 3
        # Each option is an OptionList Option carrying the session id.
        ids = [o.id for o in opts]
        assert ids == [s.id for s in sessions]
        # Labels include the short hash, the message count, and the cwd basename.
        labels = [str(o.prompt) for o in opts]
        assert any("aaaaaa" in lbl and "3msg" in lbl and "foo" in lbl for lbl in labels)
        assert any("bbbbbb" in lbl and "7msg" in lbl and "bar" in lbl for lbl in labels)

    def test_resume_picker_handles_empty_list(self) -> None:
        picker = ResumePicker([])
        # Empty list path: no options to render.
        assert picker._render_options() == []
        # The picker still constructs without error.
        assert picker.sessions == []

    def test_resume_picker_compose_with_sessions_includes_option_list(self) -> None:
        from textual.widgets import OptionList

        picker = ResumePicker([_meta("sess-a"), _meta("sess-b")])
        # We can't easily walk the descendants without mounting in an App,
        # but `_render_options` is the data path used by `compose`.
        opts = picker._render_options()
        assert len(opts) == 2
        # Constructing an OptionList over those options must succeed.
        ol = OptionList(*opts)
        assert ol is not None
