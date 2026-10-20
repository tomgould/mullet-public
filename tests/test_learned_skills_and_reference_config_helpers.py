"""Tests for learned_skills and reference_config pure helpers."""


from mullet.core.skills.learned import LearnedSkill, LearnedSkillManager
from mullet.knowledge.engine.reference_config import (
    ReferencePattern,
    ReferenceConfig,
    DOMAIN_PATTERNS,
    get_default_config,
    load_reference_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(name: str, triggers: list[str], use_count: int = 0) -> LearnedSkill:
    return LearnedSkill(name=name, description=f"{name} desc", triggers=triggers, use_count=use_count)


# ---------------------------------------------------------------------------
# LearnedSkill
# ---------------------------------------------------------------------------

class TestLearnedSkill:
    def test_fields_stored(self):
        ls = _make_skill("deploy", ["deploy to"], use_count=3)
        assert ls.use_count == 3
        assert ls.name == "deploy"
        assert ls.triggers == ["deploy to"]

    def test_use_count_defaults_zero(self):
        ls = _make_skill("deploy", ["deploy to"])
        assert ls.use_count == 0


# ---------------------------------------------------------------------------
# LearnedSkillManager.match — exact
# ---------------------------------------------------------------------------

class TestLearnedSkillManagerMatchExact:
    def _mgr_with_skill(self, name, triggers, **kw) -> LearnedSkillManager:
        mgr = LearnedSkillManager()
        mgr._skills = [_make_skill(name, triggers, **kw)]
        return mgr

    def test_exact_match_returns_result(self):
        mgr = self._mgr_with_skill("deploy", ["deploy to production"])
        result = mgr.match("please deploy to production now")
        assert result is not None
        assert result.name == "deploy"

    def test_no_match_empty(self):
        mgr = self._mgr_with_skill("deploy", ["deploy to"])
        result = mgr.match("something completely unrelated xyz")
        assert result is None

    def test_case_insensitive_match(self):
        mgr = self._mgr_with_skill("deploy", ["deploy to"])
        result = mgr.match("DEPLOY TO production")
        assert result is not None


# ---------------------------------------------------------------------------
# DOMAIN_PATTERNS
# ---------------------------------------------------------------------------

class TestDomainPatterns:
    def test_is_dict(self):
        assert isinstance(DOMAIN_PATTERNS, dict)

    def test_has_compliance(self):
        assert "compliance" in DOMAIN_PATTERNS

    def test_has_legal(self):
        assert "legal" in DOMAIN_PATTERNS

    def test_has_technical(self):
        assert "technical" in DOMAIN_PATTERNS

    def test_has_academic(self):
        assert "academic" in DOMAIN_PATTERNS

    def test_each_domain_is_dict(self):
        for domain, patterns in DOMAIN_PATTERNS.items():
            assert isinstance(patterns, dict), f"{domain} should be a dict"


# ---------------------------------------------------------------------------
# ReferencePattern
# ---------------------------------------------------------------------------

class TestReferencePattern:
    def _make(self, name="section", patterns=None, templates=None):
        return ReferencePattern(
            name=name,
            priority="medium",
            patterns=patterns or [r'\bSection\s+(\d+(?:\.\d+)*)\b'],
            query_templates=templates or ["Section {id}"],
        )

    def test_find_matches_returns_list(self):
        rp = self._make()
        result = rp.find_matches("See Section 3.1 for details.")
        assert isinstance(result, list)

    def test_find_matches_detects_section(self):
        rp = self._make()
        result = rp.find_matches("See Section 3.1 for details.")
        assert any(r["id"] == "3.1" for r in result)

    def test_find_matches_empty_for_no_match(self):
        rp = self._make()
        assert rp.find_matches("Nothing relevant here.") == []

    def test_find_matches_deduplicates(self):
        rp = self._make()
        # Same reference appears twice
        result = rp.find_matches("Section 3.1 and Section 3.1 again.")
        ids = [r["id"] for r in result]
        assert ids.count("3.1") == 1

    def test_find_matches_result_has_type(self):
        rp = self._make()
        result = rp.find_matches("Section 2.4 applies.")
        assert result[0]["type"] == "section"

    def test_find_matches_result_has_key(self):
        rp = self._make()
        result = rp.find_matches("Section 5.")
        assert "key" in result[0]

    def test_build_queries_formats_template(self):
        rp = self._make(templates=["Section {id}", "Sec. {id}"])
        queries = rp.build_queries("3.1")
        assert "Section 3.1" in queries
        assert "Sec. 3.1" in queries

    def test_build_queries_returns_list(self):
        rp = self._make()
        result = rp.build_queries("5")
        assert isinstance(result, list)

    def test_case_insensitive_match(self):
        rp = self._make()
        result = rp.find_matches("see section 4.2 below.")
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# ReferenceConfig
# ---------------------------------------------------------------------------

class TestReferenceConfigDefaults:
    def test_enabled_by_default(self):
        cfg = ReferenceConfig()
        assert cfg.enabled is True

    def test_domain_defaults_compliance(self):
        cfg = ReferenceConfig()
        assert cfg.domain == "compliance"

    def test_mode_defaults_extend(self):
        cfg = ReferenceConfig()
        assert cfg.mode == "extend"

    def test_get_patterns_returns_list(self):
        cfg = ReferenceConfig()
        assert isinstance(cfg.get_patterns(), list)

    def test_extend_mode_loads_domain_patterns(self):
        cfg = ReferenceConfig(domain="compliance", mode="extend")
        assert len(cfg.get_patterns()) > 0

    def test_replace_mode_no_default_patterns(self):
        cfg = ReferenceConfig(domain="compliance", mode="replace")
        assert len(cfg.get_patterns()) == 0


class TestReferenceConfigDetectReferences:
    def test_detects_appendix(self):
        cfg = get_default_config("compliance")
        refs = cfg.detect_references("See Appendix A for details.")
        assert any(r["type"] == "appendix" for r in refs)

    def test_detects_section(self):
        cfg = get_default_config("compliance")
        refs = cfg.detect_references("Refer to Section 5.2.")
        assert any(r["type"] == "section" for r in refs)

    def test_empty_text_returns_empty(self):
        cfg = get_default_config("compliance")
        assert cfg.detect_references("") == []

    def test_no_refs_in_plain_text(self):
        cfg = get_default_config("compliance")
        result = cfg.detect_references("The quick brown fox.")
        assert result == []

    def test_high_priority_sorted_first(self):
        cfg = get_default_config("compliance")
        refs = cfg.detect_references("See Appendix B and Section 2.1.")
        if len(refs) >= 2:
            priorities = [r["priority"] for r in refs]
            assert priorities[0] in ("high", "medium")

    def test_returns_list_of_dicts(self):
        cfg = get_default_config("compliance")
        refs = cfg.detect_references("Appendix C applies.")
        for r in refs:
            assert isinstance(r, dict)
            assert "type" in r
            assert "id" in r


class TestReferenceConfigBuildQueries:
    def test_build_queries_for_detected_ref(self):
        cfg = get_default_config("compliance")
        refs = cfg.detect_references("See Appendix A.")
        if refs:
            queries = cfg.build_queries_for_reference(refs[0])
            assert isinstance(queries, list)
            assert len(queries) > 0

    def test_unknown_ref_type_has_fallback(self):
        cfg = ReferenceConfig()
        ref = {"type": "unknown_type_xyz", "id": "99"}
        queries = cfg.build_queries_for_reference(ref)
        assert isinstance(queries, list)
        assert len(queries) > 0


class TestReferenceConfigCustomPatterns:
    def test_custom_pattern_added(self):
        cfg = ReferenceConfig(patterns={
            "clause": {
                "priority": "high",
                "patterns": [r'\bClause\s+(\d+)\b'],
                "query_templates": ["Clause {id}"],
            }
        })
        refs = cfg.detect_references("See Clause 7 for terms.")
        assert any(r["type"] == "clause" for r in refs)


# ---------------------------------------------------------------------------
# get_default_config
# ---------------------------------------------------------------------------

class TestGetDefaultConfig:
    def test_returns_reference_config(self):
        cfg = get_default_config()
        assert isinstance(cfg, ReferenceConfig)

    def test_domain_set_correctly(self):
        cfg = get_default_config("legal")
        assert cfg.domain == "legal"

    def test_has_patterns_for_domain(self):
        cfg = get_default_config("legal")
        assert len(cfg.get_patterns()) > 0


# ---------------------------------------------------------------------------
# load_reference_config
# ---------------------------------------------------------------------------

class TestLoadReferenceConfig:
    def test_empty_dict_returns_default(self):
        cfg = load_reference_config({})
        assert isinstance(cfg, ReferenceConfig)

    def test_none_returns_default(self):
        cfg = load_reference_config(None)
        assert isinstance(cfg, ReferenceConfig)

    def test_enabled_false_loaded(self):
        cfg = load_reference_config({"enabled": False})
        assert cfg.enabled is False

    def test_domain_loaded(self):
        cfg = load_reference_config({"domain": "academic"})
        assert cfg.domain == "academic"

    def test_mode_loaded(self):
        cfg = load_reference_config({"mode": "replace"})
        assert cfg.mode == "replace"

    def test_custom_patterns_loaded(self):
        cfg = load_reference_config({
            "patterns": {
                "clause": {
                    "patterns": [r'\bClause\s+(\d+)\b'],
                    "query_templates": ["Clause {id}"],
                }
            }
        })
        names = [p.name for p in cfg.get_patterns()]
        assert "clause" in names
