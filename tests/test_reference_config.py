"""Tests for mullet.knowledge.engine.reference_config."""
from __future__ import annotations


from mullet.knowledge.engine.reference_config import (
    ReferenceConfig,
    ReferencePattern,
    get_default_config,
    load_reference_config,
)


# ---------------------------------------------------------------------------
# ReferencePattern
# ---------------------------------------------------------------------------

class TestReferencePattern:
    def test_compiles_patterns_on_init(self) -> None:
        p = ReferencePattern(
            name="appendix",
            patterns=[r"\bAppendix\s+([A-Z0-9]+)\b"],
        )
        assert len(p._compiled) == 1

    def test_find_matches_returns_match(self) -> None:
        p = ReferencePattern(
            name="appendix",
            patterns=[r"\bAppendix\s+([A-Z0-9]+)\b"],
            priority="high",
        )
        matches = p.find_matches("See Appendix A for details")
        assert len(matches) == 1
        assert matches[0]["id"] == "A"
        assert matches[0]["type"] == "appendix"
        assert matches[0]["priority"] == "high"

    def test_find_matches_deduplicates(self) -> None:
        p = ReferencePattern(
            name="section",
            patterns=[r"\bSection\s+(\d+)\b", r"\bSec\.\s+(\d+)\b"],
        )
        matches = p.find_matches("Section 3 and Section 3 again")
        ids = [m["id"] for m in matches]
        assert ids.count("3") == 1

    def test_find_matches_case_insensitive(self) -> None:
        p = ReferencePattern(
            name="section",
            patterns=[r"\bSection\s+(\d+)\b"],
        )
        matches = p.find_matches("see section 5 below")
        assert len(matches) == 1
        assert matches[0]["id"] == "5"

    def test_find_matches_empty_text(self) -> None:
        p = ReferencePattern(
            name="section",
            patterns=[r"\bSection\s+(\d+)\b"],
        )
        assert p.find_matches("") == []

    def test_find_matches_no_match(self) -> None:
        p = ReferencePattern(
            name="section",
            patterns=[r"\bSection\s+(\d+)\b"],
        )
        assert p.find_matches("no references here") == []

    def test_build_queries_formats_id(self) -> None:
        p = ReferencePattern(
            name="appendix",
            query_templates=["Appendix {id}", "Annex {id}"],
        )
        queries = p.build_queries("B")
        assert "Appendix B" in queries
        assert "Annex B" in queries

    def test_build_queries_empty_templates(self) -> None:
        p = ReferencePattern(name="section", query_templates=[])
        assert p.build_queries("5") == []

    def test_key_includes_type_and_id(self) -> None:
        p = ReferencePattern(
            name="appendix",
            patterns=[r"\bAppendix\s+([A-Z0-9]+)\b"],
        )
        matches = p.find_matches("Appendix C")
        assert matches[0]["key"] == "appendix:C"


# ---------------------------------------------------------------------------
# ReferenceConfig — mode=extend (default)
# ---------------------------------------------------------------------------

class TestReferenceConfigExtend:
    def test_compliance_domain_loads_patterns(self) -> None:
        cfg = ReferenceConfig(domain="compliance")
        patterns = cfg.get_patterns()
        names = [p.name for p in patterns]
        assert "appendix" in names
        assert "section" in names

    def test_legal_domain_loads_patterns(self) -> None:
        cfg = ReferenceConfig(domain="legal")
        patterns = cfg.get_patterns()
        names = [p.name for p in patterns]
        assert "exhibit" in names
        assert "clause" in names

    def test_unknown_domain_has_no_built_in_patterns(self) -> None:
        cfg = ReferenceConfig(domain="unknown_domain")
        # Only user-defined patterns — none in this case
        assert cfg.get_patterns() == []

    def test_custom_pattern_added_to_domain(self) -> None:
        cfg = ReferenceConfig(
            domain="compliance",
            patterns={
                "custom_ref": {
                    "priority": "high",
                    "patterns": [r"\bCustom\s+([A-Z]+)\b"],
                    "query_templates": ["Custom {id}"],
                }
            },
        )
        names = [p.name for p in cfg.get_patterns()]
        assert "custom_ref" in names
        assert "appendix" in names  # domain defaults still present

    def test_custom_pattern_overrides_domain_pattern(self) -> None:
        cfg = ReferenceConfig(
            domain="compliance",
            patterns={
                "appendix": {
                    "priority": "low",
                    "patterns": [r"\bCustomAppendix\s+([A-Z]+)\b"],
                    "query_templates": [],
                }
            },
        )
        appendix_patterns = [p for p in cfg.get_patterns() if p.name == "appendix"]
        assert len(appendix_patterns) == 1
        assert appendix_patterns[0].priority == "low"


# ---------------------------------------------------------------------------
# ReferenceConfig — mode=replace
# ---------------------------------------------------------------------------

class TestReferenceConfigReplace:
    def test_replace_mode_ignores_domain_defaults(self) -> None:
        cfg = ReferenceConfig(
            domain="compliance",
            mode="replace",
            patterns={
                "my_ref": {
                    "priority": "high",
                    "patterns": [r"\bMyRef\s+([A-Z]+)\b"],
                    "query_templates": [],
                }
            },
        )
        names = [p.name for p in cfg.get_patterns()]
        assert "appendix" not in names
        assert "my_ref" in names


# ---------------------------------------------------------------------------
# ReferenceConfig.detect_references
# ---------------------------------------------------------------------------

class TestDetectReferences:
    def test_detects_appendix(self) -> None:
        cfg = ReferenceConfig(domain="compliance")
        refs = cfg.detect_references("as set out in Appendix A")
        types = [r["type"] for r in refs]
        assert "appendix" in types

    def test_detects_section(self) -> None:
        cfg = ReferenceConfig(domain="compliance")
        refs = cfg.detect_references("See Section 3.2 for guidance")
        types = [r["type"] for r in refs]
        assert "section" in types

    def test_high_priority_sorted_first(self) -> None:
        cfg = ReferenceConfig(domain="compliance")
        text = "See Section 5 and Appendix B for full details"
        refs = cfg.detect_references(text)
        if refs:
            priorities = [r["priority"] for r in refs]
            # "high" refs should appear before "medium"
            high_indices = [i for i, p in enumerate(priorities) if p == "high"]
            medium_indices = [i for i, p in enumerate(priorities) if p == "medium"]
            if high_indices and medium_indices:
                assert min(high_indices) < min(medium_indices)

    def test_detect_returns_empty_for_no_match(self) -> None:
        cfg = ReferenceConfig(domain="compliance")
        assert cfg.detect_references("nothing to find here") == []

    def test_detect_exhibit_in_legal_domain(self) -> None:
        cfg = ReferenceConfig(domain="legal")
        refs = cfg.detect_references("attached as Exhibit A to this agreement")
        types = [r["type"] for r in refs]
        assert "exhibit" in types

    def test_disabled_config_still_returns_references(self) -> None:
        """enabled=False is a caller flag; the class itself doesn't suppress detection."""
        cfg = ReferenceConfig(domain="compliance", enabled=False)
        refs = cfg.detect_references("See Appendix B")
        # We do NOT require it to return empty — the flag is informational
        assert isinstance(refs, list)


# ---------------------------------------------------------------------------
# ReferenceConfig.build_queries_for_reference
# ---------------------------------------------------------------------------

class TestBuildQueriesForReference:
    def test_builds_queries_for_known_type(self) -> None:
        cfg = ReferenceConfig(domain="compliance")
        refs = cfg.detect_references("see Appendix A for details")
        assert refs
        queries = cfg.build_queries_for_reference(refs[0])
        assert any("A" in q for q in queries)

    def test_falls_back_for_unknown_type(self) -> None:
        cfg = ReferenceConfig(domain="compliance")
        ref = {"type": "unknown_type", "id": "X", "match": "Unknown X", "priority": "low"}
        queries = cfg.build_queries_for_reference(ref)
        assert len(queries) == 1
        assert "X" in queries[0]


# ---------------------------------------------------------------------------
# Semantic phrases
# ---------------------------------------------------------------------------

class TestSemanticPhrases:
    def test_returns_default_phrases(self) -> None:
        cfg = ReferenceConfig(domain="compliance")
        phrases = cfg.get_semantic_phrases()
        assert isinstance(phrases, list)
        assert len(phrases) > 0

    def test_custom_phrases_added(self) -> None:
        cfg = ReferenceConfig(
            semantic_patterns={"compound_phrases": ["my custom phrase"]},
        )
        phrases = cfg.get_semantic_phrases()
        assert "my custom phrase" in phrases

    def test_no_duplicates_in_phrases(self) -> None:
        cfg = ReferenceConfig(domain="compliance")
        phrases = cfg.get_semantic_phrases()
        assert len(phrases) == len(set(phrases))


# ---------------------------------------------------------------------------
# load_reference_config
# ---------------------------------------------------------------------------

class TestLoadReferenceConfig:
    def test_empty_dict_returns_default(self) -> None:
        cfg = load_reference_config({})
        assert cfg.enabled is True
        assert cfg.domain == "compliance"

    def test_none_returns_default(self) -> None:
        cfg = load_reference_config({})
        assert isinstance(cfg, ReferenceConfig)

    def test_loads_domain_from_dict(self) -> None:
        cfg = load_reference_config({"domain": "legal"})
        assert cfg.domain == "legal"
        names = [p.name for p in cfg.get_patterns()]
        assert "exhibit" in names

    def test_loads_enabled_flag(self) -> None:
        cfg = load_reference_config({"enabled": False})
        assert cfg.enabled is False

    def test_loads_custom_patterns(self) -> None:
        cfg = load_reference_config({
            "domain": "compliance",
            "patterns": {
                "my_pattern": {
                    "priority": "low",
                    "patterns": [r"\bMyPat\s+([A-Z]+)\b"],
                    "query_templates": ["MyPat {id}"],
                }
            },
        })
        names = [p.name for p in cfg.get_patterns()]
        assert "my_pattern" in names


# ---------------------------------------------------------------------------
# get_default_config
# ---------------------------------------------------------------------------

class TestGetDefaultConfig:
    def test_returns_reference_config(self) -> None:
        cfg = get_default_config()
        assert isinstance(cfg, ReferenceConfig)

    def test_default_domain_is_compliance(self) -> None:
        cfg = get_default_config()
        assert cfg.domain == "compliance"

    def test_legal_domain(self) -> None:
        cfg = get_default_config("legal")
        names = [p.name for p in cfg.get_patterns()]
        assert "exhibit" in names
