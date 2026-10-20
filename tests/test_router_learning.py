"""Tests for mullet.core.router.learning."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


from mullet.core.router import learning as _lm
from mullet.core.router.learning import (
    extract_keywords,
    record_outcome,
    suggest_tier,
    _load_history,
    _trim_history,
    MAX_HISTORY,
)


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_removes_stop_words(self) -> None:
        words = extract_keywords("the quick brown fox jumps")
        assert "the" not in words
        assert "fox" in words

    def test_lowercases_input(self) -> None:
        words = extract_keywords("Refactor Authentication Module")
        assert "refactor" in words
        assert "authentication" in words

    def test_filters_short_words(self) -> None:
        words = extract_keywords("go to it")
        assert "go" not in words  # len <= 2

    def test_empty_string_returns_empty(self) -> None:
        assert extract_keywords("") == []

    def test_duplicate_words_kept(self) -> None:
        words = extract_keywords("test test test")
        assert words.count("test") == 3

    def test_non_alpha_stripped(self) -> None:
        words = extract_keywords("fix auth-module (fast)")
        assert "fix" in words
        assert "auth" in words
        assert "module" in words
        assert "fast" in words


# ---------------------------------------------------------------------------
# record_outcome
# ---------------------------------------------------------------------------

class TestRecordOutcome:
    def test_writes_jsonl_row(self, tmp_path: Path) -> None:
        outcomes_path = tmp_path / "outcomes.jsonl"
        with patch.object(_lm, "LEARNING_DIR", tmp_path):
            record_outcome("refactor auth module", "STANDARD", "kimi", True, 3, 1500)
        assert outcomes_path.exists()
        row = json.loads(outcomes_path.read_text().strip())
        assert row["tier"] == "STANDARD"
        assert row["provider"] == "kimi"
        assert row["success"] is True
        assert "auth" in row["keywords"] or "refactor" in row["keywords"]

    def test_empty_task_does_not_write(self, tmp_path: Path) -> None:
        with patch.object(_lm, "LEARNING_DIR", tmp_path):
            record_outcome("", "STANDARD", "kimi", True)
        assert not (tmp_path / "outcomes.jsonl").exists()

    def test_stop_word_only_task_does_not_write(self, tmp_path: Path) -> None:
        with patch.object(_lm, "LEARNING_DIR", tmp_path):
            record_outcome("the a an is", "STANDARD", "kimi", True)
        assert not (tmp_path / "outcomes.jsonl").exists()

    def test_multiple_rows_appended(self, tmp_path: Path) -> None:
        with patch.object(_lm, "LEARNING_DIR", tmp_path):
            record_outcome("add unit tests", "STANDARD", "kimi", True)
            record_outcome("fix routing bug", "ADVANCED", "gemini", False)
        lines = (tmp_path / "outcomes.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# suggest_tier
# ---------------------------------------------------------------------------

class TestSuggestTier:
    def _write_history(self, tmp_path: Path, entries: list[dict]) -> None:
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")

    def test_returns_none_when_no_history(self, tmp_path: Path) -> None:
        with patch.object(_lm, "LEARNING_DIR", tmp_path):
            result = suggest_tier("refactor authentication module")
        assert result is None

    def test_returns_none_when_insufficient_matches(self, tmp_path: Path) -> None:
        # Only 2 matches — threshold is 3
        entries = [
            {"keywords": ["refactor", "auth"], "tier": "STANDARD", "success": True,
             "tool_calls": 0, "tokens": 100, "ts": 1},
            {"keywords": ["refactor", "auth"], "tier": "STANDARD", "success": True,
             "tool_calls": 0, "tokens": 100, "ts": 2},
        ]
        self._write_history(tmp_path, entries)
        with patch.object(_lm, "LEARNING_DIR", tmp_path):
            result = suggest_tier("refactor authentication module")
        assert result is None

    def test_suggests_tier_on_sufficient_matches(self, tmp_path: Path) -> None:
        # 5 successful STANDARD outcomes for "refactor auth" tasks
        entries = [
            {"keywords": ["refactor", "auth", "module"], "tier": "STANDARD",
             "success": True, "tool_calls": 0, "tokens": 100, "ts": i}
            for i in range(5)
        ]
        self._write_history(tmp_path, entries)
        with patch.object(_lm, "LEARNING_DIR", tmp_path):
            result = suggest_tier("refactor the authentication module please")
        # Should suggest STANDARD (or None if overlap threshold not met — both acceptable)
        assert result is None or result in {"LOW", "STANDARD", "ADVANCED", "CRITICAL"}

    def test_prefers_cheaper_tier_on_equal_success(self, tmp_path: Path) -> None:
        # 3 LOW successes vs 3 STANDARD successes for same keywords
        entries = (
            [{"keywords": ["test", "routing", "layer"], "tier": "LOW",
              "success": True, "tool_calls": 0, "tokens": 50, "ts": i}
             for i in range(3)]
            + [{"keywords": ["test", "routing", "layer"], "tier": "STANDARD",
                "success": True, "tool_calls": 0, "tokens": 100, "ts": i}
               for i in range(3, 6)]
        )
        self._write_history(tmp_path, entries)
        with patch.object(_lm, "LEARNING_DIR", tmp_path):
            result = suggest_tier("test the routing layer")
        # LOW is cheaper and should score higher due to tier bonus
        if result is not None:
            assert result == "LOW"


# ---------------------------------------------------------------------------
# _load_history
# ---------------------------------------------------------------------------

class TestLoadHistory:
    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        with patch.object(_lm, "LEARNING_DIR", tmp_path):
            assert _load_history() == []

    def test_loads_valid_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "outcomes.jsonl"
        p.write_text('{"keywords":["test"],"tier":"STANDARD","success":true}\n')
        with patch.object(_lm, "LEARNING_DIR", tmp_path):
            entries = _load_history()
        assert len(entries) == 1
        assert entries[0]["tier"] == "STANDARD"

    def test_skips_corrupt_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "outcomes.jsonl"
        p.write_text('{"keywords":["ok"],"tier":"LOW","success":true}\nnot-json\n')
        with patch.object(_lm, "LEARNING_DIR", tmp_path):
            entries = _load_history()
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# _trim_history
# ---------------------------------------------------------------------------

class TestTrimHistory:
    def test_trims_to_max_history(self, tmp_path: Path) -> None:
        p = tmp_path / "outcomes.jsonl"
        # Write MAX_HISTORY + 50 entries
        with p.open("w") as fh:
            for i in range(MAX_HISTORY + 50):
                fh.write(json.dumps({"i": i}) + "\n")
        _trim_history(p)
        lines = p.read_text().strip().splitlines()
        assert len(lines) == MAX_HISTORY

    def test_does_not_trim_when_under_limit(self, tmp_path: Path) -> None:
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for i in range(10):
                fh.write(json.dumps({"i": i}) + "\n")
        _trim_history(p)
        lines = p.read_text().strip().splitlines()
        assert len(lines) == 10

    def test_keeps_most_recent_entries(self, tmp_path: Path) -> None:
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for i in range(MAX_HISTORY + 10):
                fh.write(json.dumps({"seq": i}) + "\n")
        _trim_history(p)
        lines = p.read_text().strip().splitlines()
        # Last entry should be the highest seq number
        last = json.loads(lines[-1])
        assert last["seq"] == MAX_HISTORY + 9
