"""Tests for smart routing learning."""

import pytest
from mullet.core.router.learning import (
    extract_keywords,
    record_outcome,
    suggest_tier,
    _load_history,
    _trim_history,
    RoutingOutcome,
)


@pytest.fixture(autouse=True)
def learning_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("mullet.core.router.learning.LEARNING_DIR", tmp_path / "routing")
    return tmp_path / "routing"


def test_extract_keywords():
    kw = extract_keywords("fix the authentication bug in the login module")
    assert "authentication" in kw
    assert "bug" in kw
    assert "login" in kw
    assert "module" in kw
    # Stop words filtered
    assert "the" not in kw
    assert "in" not in kw


def test_extract_keywords_empty():
    assert extract_keywords("") == []
    assert extract_keywords("a the is") == []


def test_record_outcome(learning_dir):
    record_outcome("fix the auth bug", tier="LOW", provider="local",
                    success=True, tool_calls=2, tokens=500)

    history = _load_history()
    assert len(history) == 1
    assert history[0]["tier"] == "LOW"
    assert history[0]["success"] is True


def test_record_multiple_outcomes(learning_dir):
    for i in range(5):
        record_outcome(f"task {i} with auth", tier="MEDIUM", provider="local",
                        success=True, tokens=1000)

    history = _load_history()
    assert len(history) == 5


def test_suggest_tier_no_history(learning_dir):
    # No history — should return None
    result = suggest_tier("fix the auth bug")
    assert result is None


def test_suggest_tier_insufficient_evidence(learning_dir):
    # Only 1 similar task — not enough evidence (need >= 3)
    record_outcome("fix auth bug", tier="LOW", provider="local", success=True)
    result = suggest_tier("fix the auth bug")
    assert result is None


def test_suggest_tier_with_evidence(learning_dir):
    # Record multiple similar tasks that succeeded at LOW
    for i in range(5):
        record_outcome(f"fix auth bug variant {i}", tier="LOW", provider="local",
                        success=True, tokens=500)

    result = suggest_tier("fix the authentication bug")
    # Should suggest LOW since similar tasks succeeded there
    assert result == "LOW"


def test_suggest_tier_prefers_lower(learning_dir):
    # Some tasks succeeded at LOW, some at HIGH
    for i in range(5):
        record_outcome(f"fix auth issue {i}", tier="LOW", provider="local",
                        success=True, tokens=500)
    for i in range(3):
        record_outcome(f"fix auth problem {i}", tier="HIGH", provider="anthropic",
                        success=True, tokens=2000)

    result = suggest_tier("fix the auth error")
    # Should prefer LOW (cheaper + successful)
    assert result == "LOW"


def test_suggest_tier_avoids_failing_tier(learning_dir):
    # Tasks at LOW failed, at MEDIUM succeeded
    for i in range(5):
        record_outcome(f"complex auth task {i}", tier="LOW", provider="local",
                        success=False, tokens=500)
    for i in range(5):
        record_outcome(f"complex auth work {i}", tier="MEDIUM", provider="cheap",
                        success=True, tokens=1000)

    result = suggest_tier("complex auth implementation")
    # Should suggest MEDIUM since LOW failed
    assert result in ("MEDIUM", None)


def test_record_outcome_empty_task(learning_dir):
    # Empty task should be silently ignored
    record_outcome("", tier="LOW", provider="local", success=True)
    assert _load_history() == []


def test_load_history_nonexistent_path():
    assert _load_history() == []


def test_load_history_empty_file(learning_dir):
    path = learning_dir / "outcomes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    assert _load_history() == []


def test_load_history_corrupt_line_skipped(learning_dir):
    path = learning_dir / "outcomes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"valid": true}\nnot json at all\n{"also_valid": true}\n')
    history = _load_history()
    assert len(history) == 2
    assert history[0] == {"valid": True}
    assert history[1] == {"also_valid": True}


def test_trim_history_no_trim_needed(learning_dir, monkeypatch):
    monkeypatch.setattr("mullet.core.router.learning.MAX_HISTORY", 5)
    path = learning_dir / "outcomes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("line1\nline2\n")
    _trim_history(path)
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2


def test_trim_history_trims_oldest(learning_dir, monkeypatch):
    monkeypatch.setattr("mullet.core.router.learning.MAX_HISTORY", 3)
    path = learning_dir / "outcomes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("old1\nold2\nkeep1\nkeep2\nkeep3\n")
    _trim_history(path)
    lines = path.read_text().strip().split("\n")
    assert lines == ["keep1", "keep2", "keep3"]


def test_trim_history_empty_file(learning_dir, monkeypatch):
    monkeypatch.setattr("mullet.core.router.learning.MAX_HISTORY", 3)
    path = learning_dir / "outcomes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    _trim_history(path)
    assert path.read_text() == ""


def test_routing_outcome_defaults():
    outcome = RoutingOutcome(task_keywords=["fix", "bug"], tier_used="LOW", provider_used="local", success=True)
    assert outcome.tool_calls == 0
    assert outcome.tokens_used == 0
    assert outcome.timestamp > 0
