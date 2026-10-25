"""Tests for mullet/knowledge/engine/reference_config.py."""
from mullet.knowledge.engine.reference_config import (
    ReferencePattern,
    ReferenceConfig,
    load_reference_config,
    get_default_config,
)


class TestReferencePatternFindMatches:
    def test_finds_appendix_reference(self):
        rp = ReferencePattern(
            name="appendix",
            priority="high",
            patterns=[r'\b(?:see\s+)?(?:Appendix|Annex)\s+([A-Z0-9]+)\b'],
            query_templates=["Appendix {id}"],
        )
        matches = rp.find_matches("Please see Appendix B for details.")
        assert len(matches) == 1
        assert matches[0]["id"] == "B"
        assert matches[0]["type"] == "appendix"

    def test_deduplicates_same_reference(self):
        rp = ReferencePattern(
            name="section",
            priority="medium",
            patterns=[r'\bSection\s+(\d+)\b'],
            query_templates=["Section {id}"],
        )
        matches = rp.find_matches("See Section 3 and Section 3 again.")
        assert len(matches) == 1

    def test_no_match_returns_empty(self):
        rp = ReferencePattern(name="section", patterns=[r'\bSection\s+(\d+)\b'])
        matches = rp.find_matches("No references here.")
        assert matches == []

    def test_multiple_different_matches(self):
        rp = ReferencePattern(
            name="section",
            patterns=[r'\bSection\s+(\d+)\b'],
            query_templates=["Section {id}"],
        )
        matches = rp.find_matches("See Section 1 and Section 2.")
        assert len(matches) == 2
        ids = {m["id"] for m in matches}
        assert ids == {"1", "2"}

    def test_case_insensitive(self):
        rp = ReferencePattern(
            name="appendix", patterns=[r'\bAppendix\s+([A-Z0-9]+)\b']
        )
        matches = rp.find_matches("see appendix C")
        assert len(matches) == 1
        assert matches[0]["id"] == "C"

    def test_match_dict_has_required_keys(self):
        rp = ReferencePattern(
            name="appendix",
            patterns=[r'\bAppendix\s+([A-Z]+)\b'],
            priority="high",
        )
        matches = rp.find_matches("Appendix A")
        assert "type" in matches[0]
        assert "id" in matches[0]
        assert "match" in matches[0]
        assert "key" in matches[0]
        assert "priority" in matches[0]


class TestReferencePatternBuildQueries:
    def test_builds_queries_from_templates(self):
        rp = ReferencePattern(
            name="appendix",
            patterns=[],
            query_templates=["Appendix {id}", "Annex {id}"],
        )
        queries = rp.build_queries("B")
        assert queries == ["Appendix B", "Annex B"]

    def test_empty_templates_returns_empty(self):
        rp = ReferencePattern(name="appendix", patterns=[])
        assert rp.build_queries("X") == []


class TestReferenceConfigDefault:
    def test_default_domain_is_compliance(self):
        rc = ReferenceConfig()
        assert rc.domain == "compliance"

    def test_compliance_patterns_loaded(self):
        rc = ReferenceConfig(domain="compliance")
        pattern_names = {p.name for p in rc.get_patterns()}
        assert "appendix" in pattern_names
        assert "section" in pattern_names

    def test_legal_patterns_loaded(self):
        rc = ReferenceConfig(domain="legal")
        pattern_names = {p.name for p in rc.get_patterns()}
        assert "exhibit" in pattern_names
        assert "clause" in pattern_names

    def test_technical_patterns_loaded(self):
        rc = ReferenceConfig(domain="technical")
        pattern_names = {p.name for p in rc.get_patterns()}
        assert "rfc" in pattern_names
        assert "issue" in pattern_names

    def test_academic_patterns_loaded(self):
        rc = ReferenceConfig(domain="academic")
        pattern_names = {p.name for p in rc.get_patterns()}
        assert "theorem" in pattern_names
        assert "equation" in pattern_names

    def test_replace_mode_ignores_domain(self):
        rc = ReferenceConfig(mode="replace", domain="compliance")
        # With no custom patterns, compiled patterns should be empty
        assert len(rc.get_patterns()) == 0

    def test_custom_pattern_added(self):
        custom = {
            "my_ref": {
                "priority": "high",
                "patterns": [r'\bFoo\s+(\d+)\b'],
                "query_templates": ["Foo {id}"],
            }
        }
        rc = ReferenceConfig(mode="replace", patterns=custom)
        names = {p.name for p in rc.get_patterns()}
        assert "my_ref" in names

    def test_custom_pattern_overrides_existing(self):
        custom = {
            "appendix": {
                "priority": "low",
                "patterns": [r'\bApp\s+(\d+)\b'],
                "query_templates": [],
            }
        }
        rc = ReferenceConfig(domain="compliance", mode="extend", patterns=custom)
        appendix_patterns = [p for p in rc.get_patterns() if p.name == "appendix"]
        assert len(appendix_patterns) == 1  # only one appendix pattern (replaced)
        assert appendix_patterns[0].priority == "low"

    def test_custom_pattern_uses_regex_key_fallback(self):
        custom = {"my_ref": {"regex": [r'\bBar\s+(\d+)\b']}}
        rc = ReferenceConfig(mode="replace", patterns=custom)
        p = next(x for x in rc.get_patterns() if x.name == "my_ref")
        assert len(p.patterns) == 1


class TestReferenceConfigGetSemanticPhrases:
    def test_includes_default_phrases(self):
        rc = ReferenceConfig()
        phrases = rc.get_semantic_phrases()
        assert "due diligence" in phrases

    def test_includes_custom_phrases(self):
        rc = ReferenceConfig(
            semantic_patterns={"compound_phrases": ["my custom phrase"]}
        )
        phrases = rc.get_semantic_phrases()
        assert "my custom phrase" in phrases

    def test_deduplicates_phrases(self):
        rc = ReferenceConfig(
            semantic_patterns={"compound_phrases": ["due diligence"]}
        )
        phrases = rc.get_semantic_phrases()
        assert phrases.count("due diligence") == 1


class TestReferenceConfigDetectReferences:
    def test_detects_appendix_in_compliance(self):
        rc = ReferenceConfig(domain="compliance")
        refs = rc.detect_references(
            "Please see Appendix A for the risk matrix."
        )
        types = {r["type"] for r in refs}
        assert "appendix" in types

    def test_detects_section_reference(self):
        rc = ReferenceConfig(domain="compliance")
        refs = rc.detect_references("See Section 3.1 for details.")
        types = {r["type"] for r in refs}
        assert "section" in types

    def test_high_priority_sorted_first(self):
        rc = ReferenceConfig(domain="compliance")
        # Appendix is high priority, section is medium
        refs = rc.detect_references("See Section 1 and Appendix B.")
        if refs:
            high_priority_first = refs[0]["priority"] in ("high",)
            # Not guaranteed to be strictly high first if text order matters,
            # but priority sort should apply
            priority_values = [r["priority"] for r in refs]
            assert (
                sorted(
                    priority_values,
                    key=lambda p: {"high": 0, "medium": 1, "low": 2}.get(p, 1),
                )
                == priority_values
            )

    def test_empty_text_returns_empty(self):
        rc = ReferenceConfig(domain="compliance")
        assert rc.detect_references("") == []

    def test_technical_rfc_detected(self):
        rc = ReferenceConfig(domain="technical")
        refs = rc.detect_references("As per RFC 7231, the method is GET.")
        types = {r["type"] for r in refs}
        assert "rfc" in types


class TestReferenceConfigBuildQueriesForReference:
    def test_builds_queries_for_known_type(self):
        rc = ReferenceConfig(domain="compliance")
        ref = {
            "type": "appendix",
            "id": "B",
            "priority": "high",
            "match": "Appendix B",
            "key": "appendix:B",
        }
        queries = rc.build_queries_for_reference(ref)
        assert any("B" in q for q in queries)

    def test_fallback_query_for_unknown_type(self):
        rc = ReferenceConfig(domain="compliance")
        ref = {
            "type": "unknown_type",
            "id": "42",
            "priority": "medium",
            "match": "Unknown 42",
            "key": "unknown_type:42",
        }
        queries = rc.build_queries_for_reference(ref)
        assert len(queries) == 1
        assert "42" in queries[0]


class TestLoadReferenceConfig:
    def test_empty_dict_returns_default(self):
        rc = load_reference_config({})
        assert isinstance(rc, ReferenceConfig)
        assert rc.domain == "compliance"

    def test_none_dict_returns_default(self):
        rc = load_reference_config(None)
        assert isinstance(rc, ReferenceConfig)

    def test_loads_domain(self):
        rc = load_reference_config({"domain": "legal"})
        assert rc.domain == "legal"

    def test_loads_mode(self):
        rc = load_reference_config({"mode": "replace"})
        assert rc.mode == "replace"

    def test_loads_enabled(self):
        rc = load_reference_config({"enabled": False})
        assert rc.enabled is False

    def test_loads_custom_patterns(self):
        rc = load_reference_config(
            {
                "patterns": {
                    "foo": {
                        "patterns": [r'\bFoo\s+(\d+)\b'],
                        "query_templates": [],
                    }
                }
            }
        )
        names = {p.name for p in rc.get_patterns()}
        assert "foo" in names


class TestGetDefaultConfig:
    def test_compliance_domain(self):
        rc = get_default_config("compliance")
        names = {p.name for p in rc.get_patterns()}
        assert "appendix" in names

    def test_legal_domain(self):
        rc = get_default_config("legal")
        names = {p.name for p in rc.get_patterns()}
        assert "exhibit" in names

    def test_default_domain_is_compliance(self):
        rc = get_default_config()
        assert rc.domain == "compliance"
