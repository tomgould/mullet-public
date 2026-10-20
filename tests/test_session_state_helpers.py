"""Tests for mullet.core.session_state — file-backed override management."""

import json
from datetime import datetime, timedelta, timezone

from mullet.core.session_state import (
    set_override,
    get_override,
    clear_override,
    is_expired,
)


# ---------------------------------------------------------------------------
# set_override / get_override
# ---------------------------------------------------------------------------

class TestSetAndGetOverride:
    def test_set_then_get_returns_override(self, tmp_path):
        set_override("opus", state_dir=tmp_path)
        result = get_override(state_dir=tmp_path)
        assert result is not None
        assert result["provider"] == "opus"

    def test_set_with_model(self, tmp_path):
        set_override("sonnet", model="claude-sonnet-4-6", state_dir=tmp_path)
        result = get_override(state_dir=tmp_path)
        assert result["model"] == "claude-sonnet-4-6"

    def test_set_without_model_is_none(self, tmp_path):
        set_override("haiku", state_dir=tmp_path)
        result = get_override(state_dir=tmp_path)
        assert result["model"] is None

    def test_creates_session_json(self, tmp_path):
        set_override("kimi", state_dir=tmp_path)
        assert (tmp_path / "session.json").exists()

    def test_override_has_expires_at(self, tmp_path):
        set_override("opus", state_dir=tmp_path)
        result = get_override(state_dir=tmp_path)
        assert "expires_at" in result

    def test_override_has_activated_at(self, tmp_path):
        set_override("opus", state_dir=tmp_path)
        result = get_override(state_dir=tmp_path)
        assert "activated_at" in result

    def test_second_set_replaces_first(self, tmp_path):
        set_override("opus", state_dir=tmp_path)
        set_override("haiku", state_dir=tmp_path)
        result = get_override(state_dir=tmp_path)
        assert result["provider"] == "haiku"

    def test_empty_state_dir_returns_none(self, tmp_path):
        result = get_override(state_dir=tmp_path)
        assert result is None

    def test_custom_expires_at(self, tmp_path):
        future = (datetime.now(timezone.utc) + timedelta(hours=10)).isoformat()
        set_override("sonnet", expires_at_iso=future, state_dir=tmp_path)
        result = get_override(state_dir=tmp_path)
        assert result is not None

    def test_expired_override_returns_none(self, tmp_path):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        set_override("sonnet", expires_at_iso=past, state_dir=tmp_path)
        result = get_override(state_dir=tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# clear_override
# ---------------------------------------------------------------------------

class TestClearOverride:
    def test_clear_makes_get_return_none(self, tmp_path):
        set_override("opus", state_dir=tmp_path)
        clear_override(state_dir=tmp_path)
        assert get_override(state_dir=tmp_path) is None

    def test_clear_on_empty_no_crash(self, tmp_path):
        clear_override(state_dir=tmp_path)  # Should not raise

    def test_clear_preserves_other_keys(self, tmp_path):
        # Write a session file with an extra key
        state_file = tmp_path / "session.json"
        state_file.write_text(json.dumps({"override": {"provider": "opus"}, "extra": "data"}))
        clear_override(state_dir=tmp_path)
        remaining = json.loads(state_file.read_text())
        assert "extra" in remaining
        assert "override" not in remaining


# ---------------------------------------------------------------------------
# is_expired
# ---------------------------------------------------------------------------

class TestIsExpired:
    def test_no_override_not_expired(self, tmp_path):
        assert is_expired(state_dir=tmp_path) is False

    def test_future_override_not_expired(self, tmp_path):
        set_override("opus", state_dir=tmp_path)
        assert is_expired(state_dir=tmp_path) is False

    def test_past_override_is_expired(self, tmp_path):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        set_override("opus", expires_at_iso=past, state_dir=tmp_path)
        assert is_expired(state_dir=tmp_path) is True

    def test_malformed_expires_at_not_expired(self, tmp_path):
        # ValueError on parse → returns False (safe fallback, doesn't assume expired)
        state_file = tmp_path / "session.json"
        state_file.write_text(json.dumps({"override": {"provider": "opus", "expires_at": "not-a-date"}}))
        assert is_expired(state_dir=tmp_path) is False

    def test_missing_expires_at_not_expired(self, tmp_path):
        state_file = tmp_path / "session.json"
        state_file.write_text(json.dumps({"override": {"provider": "opus"}}))
        assert is_expired(state_dir=tmp_path) is False
