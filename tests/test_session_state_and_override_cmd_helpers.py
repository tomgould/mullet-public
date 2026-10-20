"""Tests for session_state pure helpers and override_cmd constants."""

import json
from datetime import datetime, timedelta, timezone

from mullet.core.session_state import (
    set_override,
    get_override,
    clear_override,
    is_expired,
    _load_raw,
    _DEFAULT_TTL_HOURS,
    _STATE_FILE,
)
from mullet.cli.override_cmd import _PROVIDER_MODEL_DEFAULTS


# ---------------------------------------------------------------------------
# _load_raw
# ---------------------------------------------------------------------------

class TestLoadRaw:
    def test_missing_file_returns_empty(self, tmp_path):
        result = _load_raw(tmp_path)
        assert result == {}

    def test_valid_json_returned(self, tmp_path):
        (tmp_path / _STATE_FILE).write_text('{"key": "val"}')
        assert _load_raw(tmp_path) == {"key": "val"}

    def test_invalid_json_returns_empty(self, tmp_path):
        (tmp_path / _STATE_FILE).write_text("not json {")
        assert _load_raw(tmp_path) == {}


# ---------------------------------------------------------------------------
# set_override / get_override
# ---------------------------------------------------------------------------

class TestSetGetOverride:
    def test_get_override_empty_returns_none(self, tmp_path):
        assert get_override(tmp_path) is None

    def test_set_override_stores_provider(self, tmp_path):
        set_override("opus", state_dir=tmp_path)
        ov = get_override(tmp_path)
        assert ov["provider"] == "opus"

    def test_set_override_stores_model(self, tmp_path):
        set_override("opus", model="claude-opus-4-7", state_dir=tmp_path)
        ov = get_override(tmp_path)
        assert ov["model"] == "claude-opus-4-7"

    def test_set_override_model_none_stored(self, tmp_path):
        set_override("haiku", state_dir=tmp_path)
        ov = get_override(tmp_path)
        assert ov["model"] is None

    def test_set_override_has_expires_at(self, tmp_path):
        set_override("sonnet", state_dir=tmp_path)
        ov = get_override(tmp_path)
        assert "expires_at" in ov

    def test_set_override_has_activated_at(self, tmp_path):
        set_override("sonnet", state_dir=tmp_path)
        ov = get_override(tmp_path)
        assert "activated_at" in ov

    def test_set_override_replaces_existing(self, tmp_path):
        set_override("opus", state_dir=tmp_path)
        set_override("haiku", state_dir=tmp_path)
        ov = get_override(tmp_path)
        assert ov["provider"] == "haiku"

    def test_expired_override_returns_none(self, tmp_path):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        set_override("opus", expires_at_iso=past, state_dir=tmp_path)
        assert get_override(tmp_path) is None

    def test_custom_expires_at_honored(self, tmp_path):
        future = (datetime.now(timezone.utc) + timedelta(hours=10)).isoformat()
        set_override("opus", expires_at_iso=future, state_dir=tmp_path)
        ov = get_override(tmp_path)
        assert ov is not None

    def test_state_file_created(self, tmp_path):
        set_override("opus", state_dir=tmp_path)
        assert (tmp_path / _STATE_FILE).exists()


# ---------------------------------------------------------------------------
# clear_override
# ---------------------------------------------------------------------------

class TestClearOverride:
    def test_clear_on_empty_no_error(self, tmp_path):
        clear_override(tmp_path)  # should not raise
        assert get_override(tmp_path) is None

    def test_clear_removes_override(self, tmp_path):
        set_override("opus", state_dir=tmp_path)
        clear_override(tmp_path)
        assert get_override(tmp_path) is None

    def test_clear_preserves_other_keys(self, tmp_path):
        set_override("opus", state_dir=tmp_path)
        # Manually add another key
        path = tmp_path / _STATE_FILE
        data = json.loads(path.read_text())
        data["other_key"] = "preserved"
        path.write_text(json.dumps(data))
        clear_override(tmp_path)
        raw = _load_raw(tmp_path)
        assert raw.get("other_key") == "preserved"
        assert "override" not in raw


# ---------------------------------------------------------------------------
# is_expired
# ---------------------------------------------------------------------------

class TestIsExpired:
    def test_no_state_returns_false(self, tmp_path):
        assert is_expired(tmp_path) is False

    def test_active_override_returns_false(self, tmp_path):
        set_override("opus", state_dir=tmp_path)
        assert is_expired(tmp_path) is False

    def test_expired_override_returns_true(self, tmp_path):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        set_override("opus", expires_at_iso=past, state_dir=tmp_path)
        assert is_expired(tmp_path) is True

    def test_invalid_expires_at_returns_false(self, tmp_path):
        path = tmp_path / _STATE_FILE
        path.write_text('{"override": {"provider": "opus", "expires_at": "not-a-date"}}')
        assert is_expired(tmp_path) is False

    def test_no_expires_at_returns_false(self, tmp_path):
        path = tmp_path / _STATE_FILE
        path.write_text('{"override": {"provider": "opus"}}')
        assert is_expired(tmp_path) is False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestSessionStateConstants:
    def test_default_ttl_hours_positive(self):
        assert _DEFAULT_TTL_HOURS > 0

    def test_state_file_is_json(self):
        assert _STATE_FILE.endswith(".json")


# ---------------------------------------------------------------------------
# _PROVIDER_MODEL_DEFAULTS (override_cmd)
# ---------------------------------------------------------------------------

class TestProviderModelDefaults:
    def test_opus_in_defaults(self):
        assert "opus" in _PROVIDER_MODEL_DEFAULTS

    def test_sonnet_in_defaults(self):
        assert "sonnet" in _PROVIDER_MODEL_DEFAULTS

    def test_haiku_in_defaults(self):
        assert "haiku" in _PROVIDER_MODEL_DEFAULTS

    def test_all_values_are_strings(self):
        assert all(isinstance(v, str) for v in _PROVIDER_MODEL_DEFAULTS.values())

    def test_nonempty(self):
        assert len(_PROVIDER_MODEL_DEFAULTS) > 0

    def test_opus_model_contains_claude(self):
        assert "claude" in _PROVIDER_MODEL_DEFAULTS["opus"].lower()
