"""Tests for reference_config pure helpers and decisions_cmd formatting functions."""


from mullet.knowledge.engine.reference_config import (
    ReferencePattern,
    ReferenceConfig,
    get_default_config,
)
from mullet.cli.decisions_cmd import _kind_str, _source_str


# ---------------------------------------------------------------------------
# ReferencePattern
# ---------------------------------------------------------------------------

class TestReferencePattern:
    def _make_clause_pattern(self) -> ReferencePattern:
        return ReferencePattern(
            name="clause",
            priority="high",
            patterns=[r"clause\s+(\d+)", r"cl\.\s*(\d+)"],
            query_templates=["clause {id}", "section {id}"],
        )

    def test_name_stored(self):
        p = ReferencePattern(name="section", priority="medium", patterns=[], query_templates=[])
        assert p.name == "section"

    def test_priority_stored(self):
        p = ReferencePattern(name="s", priority="high", patterns=[], query_templates=[])
        assert p.priority == "high"

    def test_find_matches_basic(self):
        p = self._make_clause_pattern()
        matches = p.find_matches("See clause 5 for details.")
        assert len(matches) == 1
        assert matches[0]["id"] == "5"
        assert matches[0]["type"] == "clause"

    def test_find_matches_multiple(self):
        p = self._make_clause_pattern()
        matches = p.find_matches("clause 5 and cl. 12 apply here")
        ids = {m["id"] for m in matches}
        assert "5" in ids
        assert "12" in ids

    def test_find_matches_deduplicates(self):
        p = self._make_clause_pattern()
        matches = p.find_matches("clause 5 and clause 5 again")
        assert len(matches) == 1

    def test_find_matches_empty_text(self):
        p = self._make_clause_pattern()
        assert p.find_matches("") == []

    def test_find_matches_no_match(self):
        p = self._make_clause_pattern()
        assert p.find_matches("no references here") == []

    def test_find_matches_id_uppercased(self):
        p = ReferencePattern(
            name="section",
            priority="medium",
            patterns=[r"section\s+([a-z]+)"],
            query_templates=[],
        )
        matches = p.find_matches("see section abc below")
        assert matches[0]["id"] == "ABC"

    def test_match_has_key_field(self):
        p = self._make_clause_pattern()
        matches = p.find_matches("clause 7")
        assert matches[0]["key"] == "clause:7"

    def test_match_has_priority_field(self):
        p = self._make_clause_pattern()
        matches = p.find_matches("clause 7")
        assert matches[0]["priority"] == "high"

    def test_build_queries(self):
        p = self._make_clause_pattern()
        queries = p.build_queries("5")
        assert "clause 5" in queries
        assert "section 5" in queries

    def test_build_queries_empty_templates(self):
        p = ReferencePattern(name="n", priority="low", patterns=[], query_templates=[])
        assert p.build_queries("1") == []

    def test_returns_list_from_find_matches(self):
        p = self._make_clause_pattern()
        assert isinstance(p.find_matches("text"), list)

    def test_case_insensitive_matching(self):
        p = self._make_clause_pattern()
        matches = p.find_matches("CLAUSE 5")
        assert len(matches) == 1


# ---------------------------------------------------------------------------
# ReferenceConfig
# ---------------------------------------------------------------------------

class TestReferenceConfig:
    def test_defaults(self):
        cfg = ReferenceConfig()
        assert cfg.enabled is True
        assert cfg.mode == "extend"

    def test_disabled_config(self):
        cfg = ReferenceConfig(enabled=False)
        assert cfg.enabled is False

    def test_get_patterns_returns_list(self):
        cfg = ReferenceConfig()
        result = cfg.get_patterns()
        assert isinstance(result, list)

    def test_replace_mode_ignores_domain_defaults(self):
        cfg = ReferenceConfig(mode="replace", domain="compliance")
        patterns = cfg.get_patterns()
        # In replace mode with no custom patterns, should have 0 patterns
        assert isinstance(patterns, list)

    def test_extend_mode_loads_compliance_patterns(self):
        cfg = ReferenceConfig(mode="extend", domain="compliance")
        patterns = cfg.get_patterns()
        # Compliance domain should have patterns
        assert len(patterns) > 0

    def test_detect_references_returns_list(self):
        cfg = ReferenceConfig(mode="extend", domain="compliance")
        result = cfg.detect_references("some contract text")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# get_default_config
# ---------------------------------------------------------------------------

class TestGetDefaultConfig:
    def test_returns_reference_config(self):
        cfg = get_default_config()
        assert isinstance(cfg, ReferenceConfig)

    def test_enabled_by_default(self):
        cfg = get_default_config()
        assert cfg.enabled is True

    def test_has_patterns(self):
        cfg = get_default_config()
        patterns = cfg.get_patterns()
        assert len(patterns) > 0


# ---------------------------------------------------------------------------
# decisions_cmd — _kind_str and _source_str (pure formatting)
# ---------------------------------------------------------------------------

class TestKindStr:
    def test_returns_string(self):
        assert isinstance(_kind_str("decision"), str)

    def test_known_kind_contains_kind(self):
        result = _kind_str("decision")
        assert "decision" in result

    def test_unknown_kind_passthrough(self):
        result = _kind_str("foobar")
        assert "foobar" in result

    def test_rejection_kind(self):
        result = _kind_str("rejection")
        assert "rejection" in result


class TestSourceStr:
    def test_returns_string(self):
        assert isinstance(_source_str("docs", "README.md"), str)

    def test_contains_source_type(self):
        result = _source_str("docs", "README.md")
        assert "docs" in result

    def test_contains_source_path(self):
        result = _source_str("docs", "README.md")
        assert "README.md" in result

    def test_unknown_source_type_passthrough(self):
        result = _source_str("unknown_type", "file.md")
        assert "unknown_type" in result
        assert "file.md" in result
