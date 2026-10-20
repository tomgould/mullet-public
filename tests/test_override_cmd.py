"""Tests for `mullet override` CLI subcommand (override_cmd.py).

Uses the real session_state module backed by a tmp directory (monkeypatch.chdir).
No mocks needed — session_state writes real files relative to cwd.
"""
from __future__ import annotations

import json
import pytest

import mullet.cli.override_cmd as override_cmd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(argv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return override_cmd.run(argv)


# ---------------------------------------------------------------------------
# status — no override
# ---------------------------------------------------------------------------

def test_status_no_override(tmp_path, monkeypatch, capsys):
    rc = _run(["status"], tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0
    assert "not active" in out


def test_status_default_action_no_override(tmp_path, monkeypatch, capsys):
    """Calling run([]) with no action should default to 'status'."""
    rc = _run([], tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0
    assert "not active" in out


# ---------------------------------------------------------------------------
# on — defaults
# ---------------------------------------------------------------------------

def test_on_defaults_activates_opus_4_8(tmp_path, monkeypatch, capsys):
    rc = _run(["on"], tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0
    assert "opus-4-8" in out
    assert "claude-opus-4-8" in out

    # Verify set_override actually wrote the session file
    from mullet.core.session_state import get_override
    monkeypatch.chdir(tmp_path)
    state = get_override()
    assert state is not None
    assert state["provider"] == "opus-4-8"
    assert state["model"] == "claude-opus-4-8"


# ---------------------------------------------------------------------------
# on --provider opus
# ---------------------------------------------------------------------------

def test_on_provider_opus_uses_opus_model(tmp_path, monkeypatch, capsys):
    rc = _run(["on", "--provider", "opus"], tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0
    assert "opus" in out

    from mullet.core.session_state import get_override
    monkeypatch.chdir(tmp_path)
    state = get_override()
    assert state["provider"] == "opus"
    assert state["model"] == "claude-opus-4-8"


# ---------------------------------------------------------------------------
# on --provider opus-4-7 --model <explicit>
# ---------------------------------------------------------------------------

def test_on_explicit_model_wins(tmp_path, monkeypatch, capsys):
    explicit = "claude-opus-4-7-20260101"
    rc = _run(["on", "--provider", "opus-4-7", "--model", explicit], tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0
    assert explicit in out

    from mullet.core.session_state import get_override
    monkeypatch.chdir(tmp_path)
    state = get_override()
    assert state["model"] == explicit


# ---------------------------------------------------------------------------
# off — clears override
# ---------------------------------------------------------------------------

def test_off_clears_override(tmp_path, monkeypatch, capsys):
    # activate first
    _run(["on"], tmp_path, monkeypatch)
    capsys.readouterr()  # discard activation output

    rc = _run(["off"], tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0
    assert "cleared" in out

    # status should now say not active
    _run(["status"], tmp_path, monkeypatch)
    out2 = capsys.readouterr().out
    assert "not active" in out2


# ---------------------------------------------------------------------------
# on then status — shows provider / model / expiry
# ---------------------------------------------------------------------------

def test_status_after_on_shows_details(tmp_path, monkeypatch, capsys):
    _run(["on"], tmp_path, monkeypatch)
    capsys.readouterr()

    _run(["status"], tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert "active" in out
    assert "opus-4-8" in out
    assert "claude-opus-4-8" in out
    # expiry hint: either "min remaining" or "expires"
    assert ("min remaining" in out or "expires" in out)


# ---------------------------------------------------------------------------
# Invalid action → SystemExit from argparse
# ---------------------------------------------------------------------------

def test_invalid_action_raises(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run(["bogus"], tmp_path, monkeypatch)


# ---------------------------------------------------------------------------
# status with invalid expires_at → except (ValueError, TypeError) branch
# ---------------------------------------------------------------------------

def test_status_missing_expires_at_uses_raw_fallback(tmp_path, monkeypatch, capsys):
    """Lines 70-71: no expires_at in state → fromisoformat("?") raises ValueError → expiry_hint = "expires ?"."""
    monkeypatch.chdir(tmp_path)
    state_dir = tmp_path / ".mullet"
    state_dir.mkdir(parents=True, exist_ok=True)
    # No expires_at key: get_override() returns the dict unchanged (no expiry check),
    # then override_cmd uses the "?" default which fails fromisoformat.
    (state_dir / "session.json").write_text(json.dumps({
        "override": {
            "provider": "opus-4-8",
            "model": "claude-opus-4-8",
            # deliberately omit expires_at so get_override skips the fromisoformat check
        }
    }))
    rc = override_cmd.run(["status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "expires ?" in out
