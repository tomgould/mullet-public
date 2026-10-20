"""Tests for mullet.knowledge.engine.reference_config — pure helpers."""

from mullet.knowledge.engine.reference_config import (
    ReferencePattern,
    ReferenceConfig,
    load_reference_config,
    get_default_config,
    DOMAIN_PATTERNS,
)


# ---------------------------------------------------------------------------
# ReferencePattern
# ---------------------------------------------------------------------------

class TestReferencePatternFindMatches:
    def _make_pattern(self, **kwargs) -> ReferencePattern:
        defaults = dict(
            name="section",
            priority="medium",
            patterns=[r'\bSection\s+(\d+(?:\.\d+)*)\b'],
            query_templates=["Section {id}"],
        )
        defaults.update(kwargs)
        return ReferencePattern(**defaults)

    def test_single_match(self):
        p = self._make_pattern()
        matches = p.find_matches("See Section 3 for details.")
        assert len(matches) == 1
        assert matches[0]["id"] == "3"
        assert matches[0]["type"] == "section"

    def test_multiple_matches(self):
        p = self._make_pattern()
        matches = p.find_matches("See Section 3 and Section 4.1.")
        ids = {m["id"] for m in matches}
        assert "3" in ids
        assert "4.1" in ids

    def test_case_insensitive(self):
        p = self._make_pattern(patterns=[r'\b(?:Section|section)\s+(\d+)\b'])
        matches = p.find_matches("section 5 applies here")
        assert len(matches) == 1
        assert matches[0]["id"] == "5"

    def test_deduplication(self):
        p = self._make_pattern()
        # "Section 3" appears twice
        matches = p.find_matches("See Section 3, then refer to Section 3 again.")
        ids = [m["id"] for m in matches]
        assert ids.count("3") == 1

    def test_no_match_returns_empty(self):
        p = self._make_pattern()
        assert p.find_matches("No references here.") == []

    def test_match_dict_structure(self):
        p = self._make_pattern()
        matches = p.find_matches("See Section 2.")
        m = matches[0]
        assert "type" in m
        assert "id" in m
        assert "match" in m
        assert "key" in m
        assert "priority" in m

    def test_match_priority_preserved(self):
        p = self._make_pattern(priority="high")
        matches = p.find_matches("See Section 1.")
        assert matches[0]["priority"] == "high"

    def test_key_format(self):
        p = ReferencePattern(
            name="appendix",
            patterns=[r'\bAppendix\s+([A-Z])\b'],
        )
        matches = p.find_matches("See Appendix A.")
        assert len(matches) == 1
        assert ":" in matches[0]["key"]
        assert matches[0]["key"] == "appendix:A"

    def test_empty_text_returns_empty(self):
        p = self._make_pattern()
        assert p.find_matches("") == []


class TestReferencePatternBuildQueries:
    def test_single_template(self):
        p = ReferencePattern(
            name="section",
            query_templates=["Section {id}"],
        )
        queries = p.build_queries("3")
        assert queries == ["Section 3"]

    def test_multiple_templates(self):
        p = ReferencePattern(
            name="appendix",
            query_templates=["Appendix {id}", "Annex {id}"],
        )
        queries = p.build_queries("B")
        assert "Appendix B" in queries
        assert "Annex B" in queries

    def test_empty_templates_returns_empty(self):
        p = ReferencePattern(name="x", query_templates=[])
        assert p.build_queries("1") == []


# ---------------------------------------------------------------------------
# ReferenceConfig
# ---------------------------------------------------------------------------

class TestReferenceConfigDefaults:
    def test_default_enabled_true(self):
        c = ReferenceConfig()
        assert c.enabled is True

    def test_default_mode_extend(self):
        c = ReferenceConfig()
        assert c.mode == "extend"

    def test_default_domain_compliance(self):
        c = ReferenceConfig()
        assert c.domain == "compliance"

    def test_extend_mode_loads_domain_patterns(self):
        c = ReferenceConfig(domain="compliance", mode="extend")
        patterns = c.get_patterns()
        pattern_names = {p.name for p in patterns}
        assert "section" in pattern_names or "appendix" in pattern_names

    def test_replace_mode_no_domain_patterns(self):
        c = ReferenceConfig(mode="replace", patterns={})
        assert c.get_patterns() == []


class TestReferenceConfigGetSemanticPhrases:
    def test_returns_list(self):
        c = ReferenceConfig()
        phrases = c.get_semantic_phrases()
        assert isinstance(phrases, list)

    def test_custom_phrases_included(self):
        c = ReferenceConfig(
            semantic_patterns={"compound_phrases": ["my custom phrase"]}
        )
        assert "my custom phrase" in c.get_semantic_phrases()

    def test_deduplication(self):
        c = ReferenceConfig()
        phrases = c.get_semantic_phrases()
        assert len(phrases) == len(set(phrases))


class TestReferenceConfigDetectReferences:
    def test_section_detected(self):
        c = ReferenceConfig(domain="compliance")
        refs = c.detect_references("Please refer to Section 4.1 for details.")
        assert any(r["type"] == "section" and r["id"] == "4.1" for r in refs)

    def test_no_references_returns_empty(self):
        c = ReferenceConfig(domain="compliance")
        assert c.detect_references("No references in this text.") == []

    def test_multiple_types_detected(self):
        c = ReferenceConfig(domain="compliance")
        text = "See Appendix A and Section 3."
        refs = c.detect_references(text)
        types = {r["type"] for r in refs}
        assert "appendix" in types
        assert "section" in types

    def test_high_priority_refs_sorted_first(self):
        c = ReferenceConfig(domain="compliance")
        text = "See Section 2 and Appendix B."  # appendix=high, section=medium
        refs = c.detect_references(text)
        if len(refs) >= 2:
            priority_order = {"high": 0, "medium": 1, "low": 2}
            for i in range(len(refs) - 1):
                assert (priority_order.get(refs[i]["priority"], 1) <=
                        priority_order.get(refs[i + 1]["priority"], 1))

    def test_disabled_config_still_detects(self):
        # enabled flag doesn't affect detect_references — caller decides
        c = ReferenceConfig(enabled=False)
        refs = c.detect_references("See Appendix A.")
        # May or may not have results — just ensure no crash
        assert isinstance(refs, list)


class TestReferenceConfigBuildQueriesForReference:
    def test_known_ref_type_builds_queries(self):
        c = ReferenceConfig(domain="compliance")
        # Find the section pattern and test building queries
        refs = c.detect_references("Section 5 applies.")
        if refs:
            queries = c.build_queries_for_reference(refs[0])
            assert isinstance(queries, list)
            assert len(queries) > 0

    def test_unknown_ref_type_fallback(self):
        c = ReferenceConfig()
        ref = {"type": "unknown_type", "id": "99", "priority": "medium"}
        queries = c.build_queries_for_reference(ref)
        assert len(queries) == 1
        assert "99" in queries[0]


# ---------------------------------------------------------------------------
# load_reference_config
# ---------------------------------------------------------------------------

class TestLoadReferenceConfig:
    def test_empty_dict_returns_default(self):
        c = load_reference_config({})
        assert c.enabled is True
        assert c.mode == "extend"

    def test_none_returns_default(self):
        c = load_reference_config(None)
        assert c.domain == "compliance"

    def test_enabled_false_loaded(self):
        c = load_reference_config({"enabled": False})
        assert c.enabled is False

    def test_mode_override(self):
        c = load_reference_config({"mode": "replace"})
        assert c.mode == "replace"

    def test_domain_override(self):
        c = load_reference_config({"domain": "legal"})
        assert c.domain == "legal"

    def test_returns_reference_config(self):
        assert isinstance(load_reference_config({}), ReferenceConfig)


# ---------------------------------------------------------------------------
# get_default_config
# ---------------------------------------------------------------------------

class TestGetDefaultConfig:
    def test_returns_reference_config(self):
        assert isinstance(get_default_config(), ReferenceConfig)

    def test_default_domain_compliance(self):
        c = get_default_config()
        assert c.domain == "compliance"

    def test_legal_domain(self):
        c = get_default_config("legal")
        assert c.domain == "legal"

    def test_legal_has_patterns(self):
        c = get_default_config("legal")
        names = {p.name for p in c.get_patterns()}
        assert "exhibit" in names or "clause" in names


# ---------------------------------------------------------------------------
# DOMAIN_PATTERNS constant
# ---------------------------------------------------------------------------

class TestDomainPatterns:
    def test_is_dict(self):
        assert isinstance(DOMAIN_PATTERNS, dict)

    def test_compliance_domain_exists(self):
        assert "compliance" in DOMAIN_PATTERNS

    def test_legal_domain_exists(self):
        assert "legal" in DOMAIN_PATTERNS

    def test_each_domain_has_patterns(self):
        for domain, patterns in DOMAIN_PATTERNS.items():
            assert isinstance(patterns, dict), f"{domain} should have a dict of patterns"
            assert len(patterns) > 0, f"{domain} has no patterns"
