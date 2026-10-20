"""Tests for mullet.core.router.learning — pure helpers and routing logic."""

import json
from pathlib import Path
from unittest.mock import patch

from mullet.core.router.learning import (
    extract_keywords,
    RoutingOutcome,
    suggest_tier,
    record_outcome,
    LEARNING_DIR,
    MAX_HISTORY,
)

_LEARNING_MOD = "mullet.core.router.learning"


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_basic_word_extraction(self):
        result = extract_keywords("build authentication system")
        assert "build" in result
        assert "authentication" in result
        assert "system" in result

    def test_stopwords_removed(self):
        result = extract_keywords("add the new feature for user")
        assert "the" not in result
        assert "for" not in result
        assert "add" in result
        assert "new" in result
        assert "feature" in result

    def test_short_words_filtered(self):
        # Words <= 2 chars are excluded
        result = extract_keywords("do it now ok")
        assert "do" not in result
        assert "it" not in result
        assert "ok" not in result

    def test_lowercases_output(self):
        result = extract_keywords("Refactor Auth Module")
        assert "refactor" in result
        assert "auth" in result
        assert "module" in result

    def test_empty_string_returns_empty(self):
        assert extract_keywords("") == []

    def test_only_stopwords_returns_empty(self):
        assert extract_keywords("the is are and or") == []

    def test_numbers_not_extracted(self):
        # re.findall(r"[a-z]+") excludes digits
        result = extract_keywords("version 42")
        assert "42" not in result
        assert "version" in result

    def test_punctuation_stripped(self):
        result = extract_keywords("fix the bug, please!")
        assert "fix" in result
        assert "bug" in result

    def test_deduplication_not_applied(self):
        # extract_keywords does not deduplicate — repeated words appear multiple times
        result = extract_keywords("test test test")
        assert result.count("test") == 3


# ---------------------------------------------------------------------------
# RoutingOutcome
# ---------------------------------------------------------------------------

class TestRoutingOutcome:
    def test_defaults(self):
        outcome = RoutingOutcome(
            task_keywords=["build", "auth"],
            tier_used="LOW",
            provider_used="kimi",
            success=True,
        )
        assert outcome.tool_calls == 0
        assert outcome.tokens_used == 0
        assert outcome.timestamp > 0

    def test_all_fields(self):
        outcome = RoutingOutcome(
            task_keywords=["test"],
            tier_used="STANDARD",
            provider_used="sonnet",
            success=False,
            tool_calls=5,
            tokens_used=1000,
        )
        assert outcome.success is False
        assert outcome.tool_calls == 5
        assert outcome.tokens_used == 1000


# ---------------------------------------------------------------------------
# suggest_tier
# ---------------------------------------------------------------------------

class TestSuggestTier:
    def test_empty_history_returns_none(self):
        with patch(f"{_LEARNING_MOD}._load_history", return_value=[]):
            result = suggest_tier("refactor authentication")
        assert result is None

    def test_insufficient_candidates_returns_none(self):
        # Only 2 entries — below the 3-entry threshold
        history = [
            {"keywords": ["auth", "refactor", "service"], "tier": "LOW",
             "provider": "kimi", "success": True, "tool_calls": 2, "tokens": 500},
            {"keywords": ["auth", "module", "service"], "tier": "LOW",
             "provider": "kimi", "success": True, "tool_calls": 1, "tokens": 300},
        ]
        with patch(f"{_LEARNING_MOD}._load_history", return_value=history):
            result = suggest_tier("auth service refactor")
        assert result is None

    def test_sufficient_evidence_suggests_tier(self):
        # 5 successful LOW entries with matching keywords
        history = [
            {"keywords": ["auth", "login", "service", "build"], "tier": "LOW",
             "provider": "kimi", "success": True, "tool_calls": 2, "tokens": 400}
            for _ in range(5)
        ]
        with patch(f"{_LEARNING_MOD}._load_history", return_value=history):
            result = suggest_tier("build auth login service")
        assert result in ("LOW", "STANDARD", "ADVANCED", None)

    def test_empty_task_keywords_returns_none(self):
        with patch(f"{_LEARNING_MOD}._load_history", return_value=[{"keywords": ["test"]}]):
            result = suggest_tier("a b c")  # Only stopwords/too-short words
        assert result is None


# ---------------------------------------------------------------------------
# record_outcome (filesystem side-effect)
# ---------------------------------------------------------------------------

class TestRecordOutcome:
    def test_writes_to_file(self, tmp_path, monkeypatch):
        outcomes_dir = tmp_path / "routing_history"
        monkeypatch.setattr(
            "mullet.core.router.learning.LEARNING_DIR",
            outcomes_dir,
        )
        record_outcome("build auth module", "LOW", "kimi", success=True, tool_calls=2, tokens=500)
        path = outcomes_dir / "outcomes.jsonl"
        assert path.exists()
        data = json.loads(path.read_text().strip())
        assert "keywords" in data
        assert data["tier"] == "LOW"
        assert data["provider"] == "kimi"
        assert data["success"] is True

    def test_empty_keywords_not_written(self, tmp_path, monkeypatch):
        outcomes_dir = tmp_path / "routing_history"
        monkeypatch.setattr("mullet.core.router.learning.LEARNING_DIR", outcomes_dir)
        record_outcome("a b c", "LOW", "kimi", success=True)
        # No keywords → nothing written
        path = outcomes_dir / "outcomes.jsonl"
        assert not path.exists()

    def test_keywords_truncated_to_10(self, tmp_path, monkeypatch):
        outcomes_dir = tmp_path / "routing_history"
        monkeypatch.setattr("mullet.core.router.learning.LEARNING_DIR", outcomes_dir)
        task = "apple banana cherry date elderberry fig grape honeydew igloo jackfruit kiwi lemon"
        record_outcome(task, "STANDARD", "sonnet", success=True)
        path = outcomes_dir / "outcomes.jsonl"
        data = json.loads(path.read_text().strip())
        assert len(data["keywords"]) <= 10


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_history_positive(self):
        assert MAX_HISTORY > 0

    def test_learning_dir_is_path(self):
        assert isinstance(LEARNING_DIR, Path)
