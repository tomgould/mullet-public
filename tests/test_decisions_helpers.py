"""Tests for mullet.core.decisions — pure extraction helpers."""

from mullet.core.decisions import (
    Decision,
    _is_false_positive,
    _classify_line,
    _make_id,
    _PATTERNS,
    _KIND_PRIORITY,
    MIN_WORD_COUNT,
    search,
    persist,
    load,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestDecisionsConstants:
    def test_patterns_is_dict(self):
        assert isinstance(_PATTERNS, dict)

    def test_patterns_has_three_kinds(self):
        assert "decision" in _PATTERNS
        assert "rejection" in _PATTERNS
        assert "rationale" in _PATTERNS

    def test_kind_priority_order(self):
        assert _KIND_PRIORITY[0] == "decision"

    def test_min_word_count_positive(self):
        assert MIN_WORD_COUNT > 0


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------

class TestDecisionDataclass:
    def test_stores_fields(self):
        d = Decision(
            id="abc123",
            source_path="docs/README.md",
            source_type="docs",
            line_number=5,
            timestamp="2025-01-01",
            excerpt="We decided to use Kimi for delegation.",
            kind="decision",
        )
        assert d.id == "abc123"
        assert d.kind == "decision"

    def test_to_dict_round_trip(self):
        d = Decision(
            id="x", source_path="f", source_type="docs",
            line_number=1, timestamp="2025-01-01",
            excerpt="decided", kind="decision", keywords=["decided"],
        )
        d2 = Decision.from_dict(d.to_dict())
        assert d2.id == "x"
        assert d2.keywords == ["decided"]


# ---------------------------------------------------------------------------
# _make_id
# ---------------------------------------------------------------------------

class TestMakeId:
    def test_returns_string(self):
        assert isinstance(_make_id("path", "excerpt"), str)

    def test_same_inputs_same_id(self):
        assert _make_id("path", "text") == _make_id("path", "text")

    def test_different_inputs_different_id(self):
        assert _make_id("path1", "text") != _make_id("path2", "text")

    def test_length_is_16(self):
        assert len(_make_id("a", "b")) == 16


# ---------------------------------------------------------------------------
# _is_false_positive
# ---------------------------------------------------------------------------

class TestIsFalsePositive:
    def test_pytest_output_is_false_positive(self):
        line = "3584 passed, 2 skipped, 0 failed in 12.3s"
        assert _is_false_positive(line, "passed") is True

    def test_normal_prose_not_false_positive(self):
        line = "We decided to use Kimi because it is cost-effective for delegation tasks."
        assert _is_false_positive(line, "decided") is False

    def test_table_separator_is_false_positive(self):
        line = "|------|------|"
        assert _is_false_positive(line, "decided") is True

    def test_inline_code_only_is_false_positive(self):
        # keyword only inside backtick code span
        line = "See `decided` for reference."
        assert _is_false_positive(line, "decided") is True

    def test_high_non_alnum_ratio_is_false_positive(self):
        line = "|-----|----|-----|-----|-----|"
        assert _is_false_positive(line, "decided") is True

    def test_too_few_words_is_false_positive(self):
        line = "we decided yes"  # < MIN_WORD_COUNT alpha words, < 80 chars
        assert _is_false_positive(line, "decided") is True


# ---------------------------------------------------------------------------
# _classify_line
# ---------------------------------------------------------------------------

class TestClassifyLine:
    def test_decision_line_classified(self):
        line = "We decided to use FastAPI because it has async support and great performance."
        result = _classify_line(line)
        assert result is not None
        assert result[0] == "decision"

    def test_rejection_line_classified(self):
        line = "We ruled out Django as it was too heavy for our lightweight microservice approach."
        result = _classify_line(line)
        assert result is not None
        assert result[0] == "rejection"

    def test_rationale_line_classified(self):
        line = "This approach is preferred because it preserves backward compatibility with older systems."
        result = _classify_line(line)
        assert result is not None
        assert result[0] == "rationale"

    def test_empty_ish_line_returns_none(self):
        assert _classify_line("   ") is None

    def test_too_short_returns_none(self):
        assert _classify_line("decided") is None

    def test_too_long_returns_none(self):
        assert _classify_line("x" * 401) is None

    def test_markdown_only_returns_none(self):
        assert _classify_line("### ## ---") is None

    def test_returns_keywords(self):
        line = "We decided to use FastAPI because it has excellent async support and performance."
        result = _classify_line(line)
        assert result is not None
        assert len(result[1]) > 0  # has keywords

    def test_decision_wins_over_rationale(self):
        # "decided to use ... because" — should be classified as decision
        line = "We decided to use this library because it is well-maintained and widely used."
        result = _classify_line(line)
        assert result is not None
        assert result[0] == "decision"  # decision takes priority


# ---------------------------------------------------------------------------
# persist and load (filesystem integration)
# ---------------------------------------------------------------------------

class TestPersistLoad:
    def test_persist_and_load_round_trip(self, tmp_path):
        d = Decision(
            id="abc", source_path="docs/x.md", source_type="docs",
            line_number=1, timestamp="2025-01-01T00:00:00",
            excerpt="We decided to use Kimi for all delegated tasks.",
            kind="decision", keywords=["decided"],
        )
        count = persist([d], base=tmp_path)
        assert count == 1
        loaded = load(base=tmp_path)
        assert len(loaded) == 1
        assert loaded[0].id == "abc"

    def test_load_empty_returns_empty(self, tmp_path):
        result = load(base=tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def _make_decision(self, id: str, excerpt: str, kind: str = "decision", ts: str | None = "2025-01-01") -> Decision:
        return Decision(
            id=id, source_path="f", source_type="docs",
            line_number=1, timestamp=ts, excerpt=excerpt, kind=kind,
        )

    def test_finds_matching_excerpt(self, tmp_path):
        d = self._make_decision("d1", "We decided to use Kimi for delegation.")
        persist([d], base=tmp_path)
        results = search("Kimi", base=tmp_path)
        assert any(r.id == "d1" for r in results)

    def test_no_match_returns_empty(self, tmp_path):
        d = self._make_decision("d1", "We decided to use Kimi for delegation.")
        persist([d], base=tmp_path)
        results = search("nonexistent_xyz", base=tmp_path)
        assert results == []

    def test_kind_filter(self, tmp_path):
        d1 = self._make_decision("d1", "We decided to use Kimi because it works well.", kind="decision")
        d2 = self._make_decision("d2", "We rejected Django because too heavy.", kind="rejection")
        persist([d1, d2], base=tmp_path)
        results = search("", kind="rejection", base=tmp_path)
        assert all(r.kind == "rejection" for r in results)

    def test_case_insensitive(self, tmp_path):
        d = self._make_decision("d1", "We decided to use KIMI for all delegated tasks.")
        persist([d], base=tmp_path)
        results = search("kimi", base=tmp_path)
        assert any(r.id == "d1" for r in results)

    def test_limit_respected(self, tmp_path):
        decisions = [
            self._make_decision(f"d{i}", f"We decided to use option {i} because it is better.")
            for i in range(10)
        ]
        persist(decisions, base=tmp_path)
        results = search("decided", limit=3, base=tmp_path)
        assert len(results) <= 3
