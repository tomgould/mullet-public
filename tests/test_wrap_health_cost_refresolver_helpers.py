"""Tests for wrap, health_dashboard, cost_tracker, and reference_resolver pure helpers."""

import pytest

from mullet.hud.render.wrap import wcwidth_truncate, split_wrap_parts, _visible_width
from mullet.core.providers.health_dashboard import ProviderHealthDashboard
from mullet.core.providers.cost_tracker import CostTracker
from mullet.core.knowledge.reference_resolver import (
    Reference,
    REFERENCE_PATTERNS,
    detect_references,
    resolve_reference,
)


# ---------------------------------------------------------------------------
# _visible_width
# ---------------------------------------------------------------------------

class TestVisibleWidth:
    def test_plain_ascii(self):
        assert _visible_width("hello") == 5

    def test_empty_string(self):
        assert _visible_width("") == 0

    def test_ansi_not_counted(self):
        assert _visible_width("\x1b[31mhello\x1b[0m") == 5

    def test_multiple_ansi(self):
        text = "\x1b[1m\x1b[32mOK\x1b[0m"
        assert _visible_width(text) == 2


# ---------------------------------------------------------------------------
# wcwidth_truncate
# ---------------------------------------------------------------------------

class TestWcwidthTruncate:
    def test_short_line_unchanged(self):
        assert wcwidth_truncate("hi", 20) == "hi"

    def test_zero_max_width_returns_line(self):
        result = wcwidth_truncate("hello", 0)
        assert result == "hello"

    def test_negative_max_width_returns_line(self):
        result = wcwidth_truncate("hello", -1)
        assert result == "hello"

    def test_truncated_ends_with_ellipsis(self):
        result = wcwidth_truncate("hello world testing", 8)
        assert result.endswith("…")

    def test_truncated_visible_width_at_most_max(self):
        result = wcwidth_truncate("hello world testing", 8)
        assert _visible_width(result) <= 8

    def test_exact_fit_not_truncated(self):
        result = wcwidth_truncate("hello", 5)
        assert "hello" in result
        assert "…" not in result

    def test_ansi_preserved_in_output(self):
        result = wcwidth_truncate("\x1b[31mhello world test\x1b[0m", 8)
        assert "\x1b[31m" in result

    def test_returns_string(self):
        assert isinstance(wcwidth_truncate("test string here", 10), str)

    def test_single_char_max_width(self):
        result = wcwidth_truncate("hello", 1)
        # With ellipsis width of 1, budget=0 → just ellipsis
        assert "…" in result


# ---------------------------------------------------------------------------
# split_wrap_parts
# ---------------------------------------------------------------------------

class TestSplitWrapParts:
    def test_short_line_single_element(self):
        result = split_wrap_parts("Short", 80)
        assert result == ["Short"]

    def test_zero_max_width_returns_original(self):
        result = split_wrap_parts("A | B | C", 0)
        assert result == ["A | B | C"]

    def test_fits_on_one_line_not_split(self):
        result = split_wrap_parts("A | B", 80)
        assert len(result) == 1

    def test_overflow_splits_into_two(self):
        result = split_wrap_parts("A | B | C | D", 6)
        assert len(result) >= 2

    def test_returns_list(self):
        result = split_wrap_parts("Part1 | Part2", 80)
        assert isinstance(result, list)

    def test_all_parts_present(self):
        result = split_wrap_parts("A | B | C", 80)
        joined = " | ".join(result)
        assert "A" in joined
        assert "B" in joined
        assert "C" in joined

    def test_empty_string_returns_list(self):
        result = split_wrap_parts("", 80)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# ProviderHealthDashboard
# ---------------------------------------------------------------------------

class TestProviderHealthDashboardEmpty:
    def test_empty_report_is_empty_dict(self):
        d = ProviderHealthDashboard()
        assert d.get_report() == {}

    def test_providers_data_initially_empty(self):
        d = ProviderHealthDashboard()
        assert d.providers_data == {}


class TestProviderHealthDashboardRecordCall:
    def test_record_call_adds_provider(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        assert "kimi" in d.get_report()

    def test_total_calls_increments(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        d.record_call("kimi", 0.3, True)
        assert d.get_report()["kimi"]["total_calls"] == 2

    def test_success_reduces_error_rate(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        assert d.get_report()["kimi"]["error_rate"] == 0.0

    def test_failure_increases_error_rate(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, False)
        assert d.get_report()["kimi"]["error_rate"] == 1.0

    def test_mixed_error_rate(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        d.record_call("kimi", 0.5, False)
        assert d.get_report()["kimi"]["error_rate"] == pytest.approx(0.5)

    def test_average_latency_computed(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.2, True)
        d.record_call("kimi", 0.4, True)
        assert d.get_report()["kimi"]["average_latency"] == pytest.approx(0.3)

    def test_min_latency_tracked(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        d.record_call("kimi", 0.1, True)
        assert d.get_report()["kimi"]["min_latency"] == pytest.approx(0.1)

    def test_max_latency_tracked(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.1, True)
        d.record_call("kimi", 0.9, True)
        assert d.get_report()["kimi"]["max_latency"] == pytest.approx(0.9)

    def test_multiple_providers_independent(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.3, True)
        d.record_call("gemini", 0.5, False)
        report = d.get_report()
        assert report["kimi"]["error_rate"] == 0.0
        assert report["gemini"]["error_rate"] == 1.0

    def test_report_has_all_keys(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.3, True)
        entry = d.get_report()["kimi"]
        for key in ("total_calls", "error_rate", "average_latency", "min_latency", "max_latency"):
            assert key in entry


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class TestCostTrackerEmpty:
    def test_total_cost_zero_initially(self):
        ct = CostTracker()
        assert ct.total_cost() == 0.0

    def test_breakdown_empty_initially(self):
        ct = CostTracker()
        assert ct.breakdown_by_model() == {}


class TestCostTrackerRecord:
    def test_record_adds_cost(self):
        ct = CostTracker()
        ct.record("claude-3-opus", 1000, 500)
        assert ct.total_cost() > 0

    def test_total_cost_accumulates(self):
        ct = CostTracker()
        ct.record("claude-3-opus", 1000, 0)
        ct.record("claude-3-opus", 1000, 0)
        first = CostTracker()
        first.record("claude-3-opus", 1000, 0)
        assert ct.total_cost() == pytest.approx(first.total_cost() * 2)

    def test_breakdown_by_model_has_entry(self):
        ct = CostTracker()
        ct.record("claude-3-opus", 1000, 500)
        breakdown = ct.breakdown_by_model()
        assert "claude-3-opus" in breakdown

    def test_breakdown_cost_positive(self):
        ct = CostTracker()
        ct.record("claude-3-opus", 1000, 500)
        assert ct.breakdown_by_model()["claude-3-opus"] > 0

    def test_multiple_models_tracked_separately(self):
        ct = CostTracker()
        ct.record("claude-3-opus", 1000, 0)
        ct.record("claude-3-haiku", 1000, 0)
        breakdown = ct.breakdown_by_model()
        assert "claude-3-opus" in breakdown
        assert "claude-3-haiku" in breakdown

    def test_negative_tokens_in_raises(self):
        ct = CostTracker()
        with pytest.raises(ValueError):
            ct.record("claude-3-opus", -1, 0)

    def test_negative_tokens_out_raises(self):
        ct = CostTracker()
        with pytest.raises(ValueError):
            ct.record("claude-3-opus", 0, -1)

    def test_zero_tokens_zero_cost(self):
        ct = CostTracker()
        ct.record("claude-3-opus", 0, 0)
        assert ct.total_cost() == 0.0


# ---------------------------------------------------------------------------
# Reference dataclass
# ---------------------------------------------------------------------------

class TestReference:
    def test_fields_stored(self):
        r = Reference(type="Section", identifier="3.1", position=10)
        assert r.type == "Section"
        assert r.identifier == "3.1"
        assert r.position == 10

    def test_confidence_defaults_one(self):
        r = Reference(type="Figure", identifier="2", position=5)
        assert r.confidence == 1.0

    def test_explicit_confidence(self):
        r = Reference(type="Table", identifier="1", position=0, confidence=0.8)
        assert r.confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# REFERENCE_PATTERNS
# ---------------------------------------------------------------------------

class TestReferencePatterns:
    def test_is_dict(self):
        assert isinstance(REFERENCE_PATTERNS, dict)

    def test_has_tech_docs(self):
        assert "tech_docs" in REFERENCE_PATTERNS

    def test_has_legal_docs(self):
        assert "legal_docs" in REFERENCE_PATTERNS

    def test_has_academic_docs(self):
        assert "academic_docs" in REFERENCE_PATTERNS

    def test_each_entry_is_list(self):
        for key, patterns in REFERENCE_PATTERNS.items():
            assert isinstance(patterns, list), f"{key} should be a list"

    def test_each_pattern_is_compiled(self):
        for key, patterns in REFERENCE_PATTERNS.items():
            for p in patterns:
                assert hasattr(p, "finditer"), f"Pattern in {key} is not compiled"


# ---------------------------------------------------------------------------
# detect_references
# ---------------------------------------------------------------------------

class TestDetectReferences:
    def test_returns_list(self):
        result = detect_references("See Section 3.1 for details.")
        assert isinstance(result, list)

    def test_detects_section(self):
        refs = detect_references("See Section 3.1 for details.")
        assert any(r.identifier == "3.1" for r in refs)

    def test_detects_figure(self):
        refs = detect_references("As shown in Figure 2.4.")
        assert any(r.identifier == "2.4" for r in refs)

    def test_detects_table(self):
        refs = detect_references("Results in Table 1.2.")
        assert any(r.identifier == "1.2" for r in refs)

    def test_detects_appendix(self):
        refs = detect_references("Refer to Appendix B.")
        assert any(r.identifier == "B" for r in refs)

    def test_empty_text_no_refs(self):
        refs = detect_references("")
        assert refs == []

    def test_no_refs_in_plain_text(self):
        refs = detect_references("The quick brown fox jumps over the lazy dog.")
        assert refs == []

    def test_legal_docs_article(self):
        refs = detect_references("Under Article 5 of the agreement.", "legal_docs")
        assert any(r.identifier == "5" for r in refs)

    def test_legal_docs_exhibit(self):
        refs = detect_references("See Exhibit A attached.", "legal_docs")
        assert any(r.identifier == "A" for r in refs)

    def test_academic_chapter(self):
        refs = detect_references("As explained in Chapter 4.", "academic_docs")
        assert any(r.identifier == "4" for r in refs)

    def test_position_is_int(self):
        refs = detect_references("Section 2.1 applies.")
        for r in refs:
            assert isinstance(r.position, int)

    def test_abbreviation_sec(self):
        refs = detect_references("See Sec. 4.2 for examples.")
        assert len(refs) >= 1

    def test_case_insensitive_section(self):
        refs = detect_references("refer to section 1.0 below.")
        assert len(refs) >= 1

    def test_unknown_doc_type_no_refs(self):
        # Unknown doc type returns empty patterns → no refs found
        refs = detect_references("See Section 1.0.", "unknown_type")
        assert refs == []

    def test_multiple_refs_in_text(self):
        refs = detect_references("See Section 1.1 and Figure 3.2 and Table 4.1.")
        assert len(refs) >= 3


# ---------------------------------------------------------------------------
# resolve_reference
# ---------------------------------------------------------------------------

class TestResolveReference:
    def test_returns_dict_when_found(self):
        ref = Reference(type="Section", identifier="3.1", position=0)
        index = {"doc1.txt": "See Section 3.1 for more details."}
        result = resolve_reference(ref, index)
        assert isinstance(result, dict)

    def test_found_has_document_id(self):
        ref = Reference(type="Section", identifier="3.1", position=0)
        index = {"doc1.txt": "Section 3.1 covers this."}
        result = resolve_reference(ref, index)
        assert result["document_id"] == "doc1.txt"

    def test_found_has_match_text(self):
        ref = Reference(type="Section", identifier="3.1", position=0)
        index = {"doc1.txt": "Section 3.1 covers this."}
        result = resolve_reference(ref, index)
        assert "match_text" in result

    def test_not_found_returns_none(self):
        ref = Reference(type="Section", identifier="99.99", position=0)
        index = {"doc1.txt": "No such section here."}
        result = resolve_reference(ref, index)
        assert result is None

    def test_empty_index_returns_none(self):
        ref = Reference(type="Section", identifier="1.0", position=0)
        result = resolve_reference(ref, {})
        assert result is None

    def test_searches_multiple_docs(self):
        ref = Reference(type="Section", identifier="5", position=0)
        index = {
            "a.txt": "No match here.",
            "b.txt": "See Section 5 for this.",
        }
        result = resolve_reference(ref, index)
        assert result is not None
        assert result["document_id"] == "b.txt"
