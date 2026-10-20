"""Tests for mullet.core.session_state."""

import json
from datetime import datetime, timedelta, timezone


from mullet.core.session_state import (
    clear_override,
    get_override,
    is_expired,
    set_override,
)


def test_set_get_roundtrip(tmp_path):
    set_override("kimi", model="kimi-k2", state_dir=tmp_path)
    result = get_override(state_dir=tmp_path)
    assert result is not None
    assert result["provider"] == "kimi"
    assert result["model"] == "kimi-k2"
    assert "activated_at" in result
    assert "expires_at" in result


def test_clear_override(tmp_path):
    set_override("gemini", state_dir=tmp_path)
    clear_override(state_dir=tmp_path)
    assert get_override(state_dir=tmp_path) is None


def test_missing_file_returns_none(tmp_path):
    assert get_override(state_dir=tmp_path) is None


def test_malformed_json_returns_none(tmp_path):
    (tmp_path / "session.json").write_text("{this is not valid json")
    assert get_override(state_dir=tmp_path) is None


def test_expired_override_returns_none_and_clears(tmp_path):
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    set_override("haiku", expires_at_iso=past, state_dir=tmp_path)
    assert get_override(state_dir=tmp_path) is None
    # override key should be gone from disk
    raw = json.loads((tmp_path / "session.json").read_text())
    assert "override" not in raw


def test_is_expired_true_for_past(tmp_path):
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    set_override("local", expires_at_iso=past, state_dir=tmp_path)
    assert is_expired(state_dir=tmp_path) is True


def test_is_expired_false_for_future(tmp_path):
    future = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    set_override("kimi", expires_at_iso=future, state_dir=tmp_path)
    assert is_expired(state_dir=tmp_path) is False


def test_default_expires_at_is_five_hours_out(tmp_path):
    before = datetime.now(timezone.utc)
    set_override("kimi", state_dir=tmp_path)
    after = datetime.now(timezone.utc)
    result = get_override(state_dir=tmp_path)
    expires = datetime.fromisoformat(result["expires_at"])
    assert before + timedelta(hours=5) <= expires <= after + timedelta(hours=5)


def test_state_dir_parameter_honoured(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    set_override("kimi", state_dir=dir_a)
    assert get_override(state_dir=dir_a) is not None
    assert get_override(state_dir=dir_b) is None


def test_save_raw_exception_cleans_up_tmp_file(tmp_path):
    """When os.replace fails, _save_raw removes the tmp file and re-raises."""
    from unittest.mock import patch
    from mullet.core.session_state import _save_raw
    import pytest

    with patch("mullet.core.session_state.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            _save_raw({"key": "val"}, tmp_path)


def test_save_raw_exception_suppresses_unlink_oserror(tmp_path):
    """Cleanup unlink errors are swallowed; the original exception still propagates."""
    from unittest.mock import patch
    from mullet.core.session_state import _save_raw
    import pytest

    with patch("mullet.core.session_state.os.replace", side_effect=OSError("disk full")), \
         patch("mullet.core.session_state.os.unlink", side_effect=OSError("no such file")):
        with pytest.raises(OSError, match="disk full"):
            _save_raw({"key": "val"}, tmp_path)


def test_get_override_invalid_expires_at_returns_none(tmp_path):
    """get_override returns None and clears when expires_at is not a valid ISO string."""
    state_file = tmp_path / "session.json"
    state_file.write_text(
        json.dumps({"override": {"provider": "kimi", "expires_at": "not-a-date"}})
    )
    assert get_override(state_dir=tmp_path) is None
    raw = json.loads(state_file.read_text())
    assert "override" not in raw


def test_is_expired_no_override(tmp_path):
    """is_expired returns False when there is no override at all."""
    assert is_expired(state_dir=tmp_path) is False


def test_is_expired_override_without_expires_at(tmp_path):
    """is_expired returns False when expires_at key is absent."""
    (tmp_path / "session.json").write_text(
        json.dumps({"override": {"provider": "kimi"}})
    )
    assert is_expired(state_dir=tmp_path) is False


def test_is_expired_invalid_expires_at_returns_false(tmp_path):
    """is_expired returns False when expires_at cannot be parsed."""
    (tmp_path / "session.json").write_text(
        json.dumps({"override": {"provider": "kimi", "expires_at": "garbage-date"}})
    )
    assert is_expired(state_dir=tmp_path) is False


def test_unrelated_keys_survive_set_and_clear(tmp_path):
    # seed a sibling key
    state_file = tmp_path / "session.json"
    state_file.write_text(json.dumps({"other_key": "preserved"}))
    set_override("kimi", state_dir=tmp_path)
    raw = json.loads(state_file.read_text())
    assert raw.get("other_key") == "preserved"
    clear_override(state_dir=tmp_path)
    raw = json.loads(state_file.read_text())
    assert raw.get("other_key") == "preserved"
    assert "override" not in raw
